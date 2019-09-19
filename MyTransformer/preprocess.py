import argparse
import torch
import Constants as Constants

def read_instances_from_file(inst_file, max_sent_len, keep_case):
    sent_list = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            word_list = sent.split()
            #这里实现的不一样
            if len(word_list) > max_sent_len:
                word_list = word_list[:max_sent_len]
                trimmed_sent_count += 1
            if word_list:
                sent_list += [[Constants.BOS_WORD] + word_list + [Constants.EOS_WORD]]
            else:
                sent_list += [None]
    print('[Info] Get {} instances(sentences) from {}'.format(len(sent_list), inst_file))

    if trimmed_sent_count>0:
        print('[Warring] {} instances are trimed to the max sentence length {}'.format(trimmed_sent_count, max_sent_len))
    return sent_list

def build_vocab_idx(word_insts, min_word_count):
    #获得所有word
    #建立word2idx
    #统计词频
    #过滤频率太低的词
    #返回word2idx
    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))
    
    word2idx = {
        Constants.EOS_WORD: Constants.EOS,
        Constants.BOS_WORD: Constants.BOS,
        Constants.UNK_WORD: Constants.UNK,
        Constants.PAD_WORD: Constants.PAD
    }
    word_count = {w:0 for w in full_vocab}
    for sent in word_insts:
        for w in sent:
            word_count[w] += 1

    ignored_word_count = 0
    for w in full_vocab:
        if word_count[w] > min_word_count:
            word2idx[w] = len(word2idx)
        else:
            ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    
    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_token_seq_len + 2
    
    #set train_word_insts
    #read src and tgt word
    #让两者句子数一样
    #remove empty sentence
    train_src_word_insts = read_instances_from_file((opt.train_src, opt.max_token_seq_len, opt.keep_case))
    train_tgt_word_insts = read_instances_from_file(opt.train_tgt, opt.max_token_seq_len, opt.keep_case)
    
    if len(train_src_word_insts) != len(train_tgt_word_insts):
        min_insts_len = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_insts_len]
        train_tgt_word_insts = train_tgt_word_insts[:min_insts_len]
    
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (i, j) for i,j in zip(train_src_word_insts, train_tgt_word_insts) if i and j
        ]))
    
    #set valid word insts
    valid_src_word_insts = read_instances_from_file((opt.valid_src, opt.max_token_seq_len, opt.keep_case))
    valid_tgt_word_insts = read_instances_from_file(opt.valid_tgt, opt.max_token_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        min_insts_len = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_insts_len]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_insts_len]

    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (i, j) for i, j in zip(valid_src_word_insts, valid_tgt_word_insts) if i and j
    ]))
    # Build vocabulary/ 建立词库的word2idx
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['src']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(train_src_word_insts+train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)
    #建立数据集的word2idx
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)

    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    #把insts、word2idx储存起来
    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx
        },
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts
        },
        'valid':{
            'src': valid_src_insts,
            'tgt': valid_tgt_insts
        }
    }
    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()