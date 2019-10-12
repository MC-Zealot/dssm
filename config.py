#!/usr/bin/env python
# encoding=utf-8


def load_vocab(file_path):
    word_dict = {}
    with open(file_path, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict


class Config(object):
    def __init__(self):
        self.vocab_map = load_vocab(self.vocab_path)
        self.nwords = len(self.vocab_map)

    unk = '[UNK]'
    pad = '[PAD]'
    #vocab_path = '/Users/Zealot/yizhou/yizhou/git/dssm/data/vocab.txt'
    vocab_path = './data/vocab.txt'
    # vocab_path = './data/vocab_filtered.txt'
    # file_train = './data/oppo_round1_train_20180929_mini_test.txt'
    file_train = './data/comment/dataset20190101_no_extend_no_left.txt'
    # file_train = './data/oppo_round1_train_20180929.txt'
    #file_train = './data/oppo_round1_train_20180929_2.txt'
    # file_vali = './data/oppo_round1_vali_20180929_mini.txt'
    file_vali = './data/comment/dataset20190101_no_extend_no_left_vali.txt'
    # file_vali = './data/oppo_round1_vali_20180929.txt'
    max_seq_len = 10
    hidden_size_rnn = 100
    use_stack_rnn = False
    learning_rate = 0.01
    # max_steps = 8000
    num_epoch = 4
    summaries_dir = './Summaries/'
    gpu = 0
    # negative sample
    NEG = 4
    # query batch size
    query_BS = 256
    query_mid_vector_file = r'data/y_mid_vector.txt'
    doc_pos_y_mid_vector_file = r'data/doc_pos_y_mid_vector.txt'
    doc_neg_y_mid_vector_file = r'data/doc_neg_y_mid_vector.txt'


if __name__ == '__main__':
    conf = Config()
    print(len(conf.vocab_map))
    pass
