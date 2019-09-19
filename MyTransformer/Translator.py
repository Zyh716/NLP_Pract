import torch
import torch.nn as nn
import torch.nn.functional as F

from Module import Transformer
from Beam import Beam

class Translator(object):
    def __init__(self, opt):
        #定义属性
        #建立模型
        #储存
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = Transformer(model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
            emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded')

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)
        self.model = model

        self.model.eval()




    def translate_batch(self, src_seq, src_pos):
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def beam_decoder_step(inst_dec_beams, len_dec, src_seq, enc_output, indx2position_map, n_bm):

            def prepare_beam_dec_seq(inst_dec_beams, len_dec):
                #就是将所有未完成的beam里已有的输出以（bh*n_bm, len_dec）的格式返回
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec)

                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec, n_active_insts, n_bm):
                dec_partial_pos = torch.arange(1, len_dec+1. dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_insts*n_bm, 1)

                return dec_partial_pos
            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_insts, n_bm):
                dec_output, *_ = self.model.encoder(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]
                word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)
                word_prob = dec_output.view(n_active_insts, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_dec_beams, word_prob, indx2position_map):
                active_insts_list = []
                for idx, pos in indx2position_map:
                    is_done = inst_dec_beams[idx].advance(word_prob[pos])
                    if not is_done:
                        active_insts_list += [idx]

                return active_insts_list

            n_active_insts = len(indx2position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec)
            dec_pos = prepare_beam_dec_pos(len_dec, n_active_insts, n_bm)
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_insts, n_bm)
            activate_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, indx2position_map)

            return activate_inst_idx_list

        def collect_active_part(beam_tensor, active_pos_list, n_pre_active_insts, n_bm):
            _, *dim = beam_tensor.size()
            n_current_active_insts = len(active_pos_list)
            new_shape = (n_current_active_insts * n_bm, *dim)

            beam_tensor = beam_tensor.view(n_pre_active_insts, -1)
            beam_tensor = beam_tensor.index_select(0, active_pos_list)
            beam_tensor = beam_tensor.view(*new_shape)

            return beam_tensor

        def collate_active_info(src_seq, enc_output, indx2position_map, activate_inst_list):
            n_pre_active_insts = len(indx2position_map)
            active_pos_list = [indx2position_map[idx] for idx in activate_inst_list]
            active_pos_list = torch.LongTensor(active_pos_list).to(self.device)

            src_seq = collect_active_part(src_seq, active_pos_list, n_pre_active_insts, n_bm)
            enc_output = collect_active_part(enc_output, active_pos_list, n_pre_active_insts, n_bm)
            indx2position_map = get_inst_idx_to_tensor_position_map(activate_inst_list)

            return src_seq, enc_output, indx2position_map

        def collate_hyp_and_scores(inst_dec_beams, n_best):
            all_hyp, all_score = [], []
            for idx in range(len(inst_dec_beams)):
                score, tail = inst_dec_beams[idx].sort_scores()
                all_score += [score[:n_best]]
                hyps = [inst_dec_beams.get_hyp_stream_from_one_final_scores(i) for i in tail[:n_best]]
                all_hyp += [hyps]

            return all_hyp, all_score

        with torch.no_grad():
            #改变设备、获得编码器输出
            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            enc_output, *_ = self.model.encoder(src_seq, src_pos)

            #repeat data for beam search
            n_bm = self.opt.beam_size
            n_inst, len_s, d_h = enc_output.size()
            src_seq.repeat(1, n_bm).view(n_inst*n_bm, len_s)
            enc_output.repeat(1, n_bm, 1).view(n_inst*n_bm, len_s, d_h)
            '''a.repeat(1,2).view(4,3)
                tensor([[ 1,  2,  3],
                [ 1,  2,  3],
                [45,  6,  7],
                [45,  6,  7]])
               '''
            #准备beams
            inst_dec_beams = [Beam(self.opt.n_bm, self.device) for i in range(n_inst)]

            #设置list，储存还未获得全部输出的seq
            activate_inst_list = list(range(n_inst))
            indx2position_map = get_inst_idx_to_tensor_position_map(activate_inst_list)

            #decoder：以一个seq（句子）最大单词数为循环次数，循环体为一个beam——step，并且更新上一步设置的list
            for len_dec in range(1, self.opt.max_token_seq_len+1):
                activate_inst_list = beam_decoder_step(
                    inst_dec_beams, len_dec, src_seq, enc_output, indx2position_map, n_bm)
                if not activate_inst_list:
                    break
                src_seq, enc_output, indx2position_map = collate_active_info(src_seq, enc_output, indx2position_map, activate_inst_list)

        #获得预测值和分数
        batch_hyp, batch_scores = collate_hyp_and_scores(inst_dec_beams, self.opt.n_best)
        return batch_hyp, batch_scores
