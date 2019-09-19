import torch
import numpy as np
import Constants as Constants

class Beam():
    def __init__(self, size, device=False):
        self.size = size
        self._done = False

        self.cur_scores = torch.zeros((size, ), dtype=torch.float, device=device)
        self.all_scores = []

        self.score_father = []

        self.all_word_id = [torch.full((size,), Constants.PAD, dtype=torch.long, device=device)]
        self.all_word_id [0][0] = Constants.BOS

    def get_current_state(self):
        return self.get_tentative_hypothesis()

    def get_current_father(self):
        return self.score_father[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        '''size of word_prob: (size, vocab) if 已经进行了至少一次搜索（获得了size个搜索结果）
                              (1, vocab) if 是第一次搜索'''
        n_vocab = word_prob.size(1)

        if len(self.all_scores) > 0:
            new_beam_score = word_prob + self.cur_scores.unsqueeze(1).expand_as(word_prob)
        else:
            new_beam_score = word_prob[0]
        new_beam_score_flat = new_beam_score.view(-1)

        best_size_scores, best_size_scores_id = new_beam_score_flat.topk(self.size, 0, True, True)

        self.all_scores.append(self.cur_scores)
        self.cur_scores = best_size_scores

        beam_id = best_size_scores_id / n_vocab
        self.score_father.append(beam_id)
        self.next_ys.append(best_size_scores_id - beam_id*n_vocab)

        if self.all_word_id[-1][0] == Constants.EOS:
            self._done = True
            self.all_scores.append(self.cur_scores)

        return self._done

    def sort_scores(self):
        return torch.sort(self.cur_scores, 0, True)

    def get_the_best_score_and_idx(self):
        pass
        #感觉有问题：1 没被使用过 2 原版代码函数实现好像不太对

    def get_tentative_hypothesis(self):
        if len(self.all_word_id) == 1:
            hyp_stream_list = self.all_word_id[0].unsqueeze(1)
        else:
            _, ids = self.sort_scores()
            hyp_stream_list = [self.get_hyp_stream_from_one_final_scores(k) for k in ids]
            hyp_stream_list = [[Constants.BOS] + hsl for hsl in hyp_stream_list]
            hyp_stream_list = torch.LongTensor(hyp_stream_list)

        return hyp_stream_list

    def get_hyp_stream_from_one_final_scores(self, k):
        hyp_stream = []
        for i in range(len(self.all_word_id) -1, -1, -1):
            hyp_stream.append(self.all_word_id[i+1][k])
            k = self.score_father[i][k]

        return hyp_stream
