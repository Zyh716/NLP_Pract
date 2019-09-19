import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import Constants as Constants


def get_non_pad_mask(seq):
    # 可以替代
    # size of seq: bh, lens
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)
    #size of return: bh, lens

def get_attn_key_pad_mask(seq_k, seq_q):
    # 可以替代
    len_q = seq_q.size(1)
    mask = seq_k.eq(Constants.PAD)
    mask = mask.unsqueeze(1).expand(-1, len_q, -1)

    return mask
    #size of return: bh, lenq. lenk

def get_subsequent_mask(seq):
    # 可以替代
    #size of seq: bh, lens
    bh, lens = seq.size()
    mask = torch.triu(torch.ones((lens, lens), device=seq.device, dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(bh, -1, -1)

    return mask
    #size of return: bh, lens, lens

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    #pass
    def cal_angele(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    def get_posi_angle_vec(position):
        return [cal_angele(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class ScaledDotProductAttention(nn.Module):
    #可以替代
    def __init__(self, scale, dropout_rate=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None, maskval=-np.inf):
        #size of q : bh, lens, n*dk
        #size of mask ： bh, lens, lens
        #这个mask是上三角（不含主对角线的mask）
        attn = torch.bmm(q, k.transpose(1,2))
        attn = attn / self.scale

        if mask:
            attn = attn.mask_fill(mask, maskval)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
        #size of output : bh, lens, n*dk  #size of attn: bh, lens, lens

class MultiHeadAttention(nn.Module):
    #pass
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout_rate = dropout

        self.fc_mh_q = nn.Linear(d_model, n_head * d_k)
        self.fc_mh_k = nn.Linear(d_model, n_head * d_k)
        self.fc_mh_v = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.fc_mh_q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.fc_mh_k.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.fc_mh_v.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.self_attention = ScaledDotProductAttention(np.power(d_k, 0.5), dropout)

        self.fc_back_v = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc_back_v.weight)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask):
        # size of q : bh, lens, dmodel  #size of mask ： bh, lens, lens
        # size of mask: bh, lens, lens
        batch_size, len_seq_q, _ = q.size()
        batch_size, len_seq_k, _ = k.size()
        batch_size, len_seq_v, _ = v.size()
        res = q

        q = self.fc_mh_q(q).view(batch_size, len_seq_q, self.n_head, self.d_k)  # bh, lens, n*dk
        k = self.fc_mh_k(k).view(batch_size, len_seq_k, self.n_head, self.d_k)
        v = self.fc_mh_v(v).view(batch_size, len_seq_v, self.n_head, self.d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_head, len_seq_q, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_head, len_seq_k, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_head, len_seq_v, self.d_k)

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)
        output, attn = self.self_attention(q, k, v, mask)  # size of output: n*bh, lens, dk

        output = output.contiguous().view(self.n_head, batch_size, len_seq_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_seq_q, -1)
        # print(output.size()[2], self.fc_back_v.)
        output = self.dropout(self.fc_back_v(output))
        output = self.layer_norm(output + res)

        return output, attn  # size of output = q  #size of attn: bh*n, lens, lens

class PositionwiseFeedForward(nn.Module):
    #可以替代
    def __init__(self, dim_in, dim_hid, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.fc2 = nn.Linear(dim_hid, dim_in)
        self.layer_norm = nn.LayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        #size of input: bh, lens, dmodel
        res = x
        output = self.fc1(x)
        output = self.fc2(F.relu(output))
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output   #size equal to x

class EncoderLayer(nn.Module):
    #pass
    def __init__(self, d_model, dim_hid, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.MultiHead = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, dim_hid, dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        #non_pad_mask 将enc_input原本是pad的地方在计算后的数据那里置为0, slf_attn_mask 是上三角矩阵，实施masked xxxx，
        #size of x: bh, lens, dmodel

        enc_output, attn = self.MultiHead(enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, attn

class DecoderLayer(nn.Module):
    #pass
    def __init__(self, d_model, dim_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.MaskedMultiHead = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.MultiHead = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, dim_hid, dropout)


    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.MaskedMultiHead(dec_input, dec_input, dec_input, mask = slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.MultiHead(dec_input, enc_output, enc_output, mask = dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn



class Encoder(nn.Module):
    # pass
    def __init__(self, n_src_vocab, len_max_seq, d_word_vec,
                 n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, Constants.PAD)

        self.position_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, 0), freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attn=False):
        enc_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(src_seq, src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        enc_output = self.src_word_emb(src_seq) + self.position_emb(src_pos)
        #print("enc_output in encoder size is ", enc_output.size()) #normal
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, non_pad_mask, slf_attn_mask)
            # print("enc_output in encoder size is ", enc_output.size())  # normal
            if return_attn:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attn:
            return enc_output, enc_slf_attn_list
        # print("enc_output in encoder size is ", enc_output.size())  # normal
        # #add
        # global enc_output_c_list
        # enc_output_c_list += [enc_output]
        return enc_output,


class Decoder(nn.Module):
    # pass
    def __init__(self, n_tgt_vocab, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 dropout):
        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, Constants.PAD)
        self.position_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, 0), freeze=True)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_word_vec, n_head, d_k, d_v, dropout) for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attn=False):
        # print("enc_input size in decoder1 is ", enc_output.size())
        dec_slf_attn_list = []
        dec_enc_attn_list = []

        triu_mask = get_subsequent_mask(tgt_seq)
        slf_attn_mask = (get_attn_key_pad_mask(tgt_seq, tgt_seq) + triu_mask).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(src_seq, tgt_seq)
        # print("dec_enc_attn_mask size in decoder is ",dec_enc_attn_mask.size())
        non_pad_mask = get_non_pad_mask(tgt_seq)

        # def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_emb(tgt_pos)

        for dec_layer in self.layer_stack:
            # print("enc_input size in decoder2 is ", enc_output.size())
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
            if return_attn:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attn:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    #pass
    def __init__(self, n_src_vocab, n_tgt_vocab, len_max_seq,
                 d_word_vec=512, d_model=512, d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
                 tgt_emb_prj_weight_sharing=True,
                 emb_src_tgt_weight_sharing=True):
        super().__init__()
        self.encoder = Encoder(n_src_vocab, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.decoder = Decoder(n_tgt_vocab, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec

        if tgt_emb_prj_weight_sharing:
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = d_model ** -0.5
        else:
            self.x_logit_scale = 1 #这是啥

        if emb_src_tgt_weight_sharing:
            assert n_src_vocab == n_tgt_vocab
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]   #为什么不包含最后一个

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
        #size of return : bh*lens, tgt_vocab