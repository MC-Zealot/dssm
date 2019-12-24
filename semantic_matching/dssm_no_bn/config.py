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
        self.vocab_path = '../../data/vocab.txt'
        self.file_train = '../../data/comment/trainset_repeat_20190508_20190514_shuffle.txt'
        # self.file_train = '../../data/dataset_20190508_20190514_2w.txt'
        # self.file_vali = '../../data/dataset_vali_20190515_20190515_5k.txt'
        self.file_vali = '../../data/comment/testset_repeat_20190515_20190515_shuffle.txt'
        self.L1_N = 1000
        self.L2_N = 300
        self.query_BS = 512
        self.learning_rate = 0.00001
        self.num_epoch = 40
        self.summaries_dir = './Summaries/'
        self.gpu = 0
        # negative sample
        self.NEG = 4
        # query batch size

        self.query_mid_vector_file = r'output/y_mid_vector.txt'
        self.doc_pos_y_mid_vector_file = r'output/doc_pos_y_mid_vector.txt'
        self.doc_neg_y_mid_vector_file = r'output/doc_neg_y_mid_vector.txt'
        self.max_seq_len = 10
        self.hidden_size_rnn = 100
        self.use_stack_rnn = False


if __name__ == '__main__':
    conf = Config()
    print(len(conf.vocab_map))
    pass
