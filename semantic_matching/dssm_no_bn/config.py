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

    vocab_path = '../../data/vocab.txt'
    file_train = '../../data/dataset_20190508_20190514_2w.txt'
    file_vali = '../../data/dataset_vali_20190515_20190515_5k.txt'

    L1_N = 1000
    L2_N = 300
    query_BS = 512
    learning_rate = 0.1
    num_epoch = 10
    summaries_dir = './Summaries/'
    gpu = 0
    # negative sample
    NEG = 4
    # query batch size

    query_mid_vector_file = r'output/y_mid_vector.txt'
    doc_pos_y_mid_vector_file = r'output/doc_pos_y_mid_vector.txt'
    doc_neg_y_mid_vector_file = r'output/doc_neg_y_mid_vector.txt'
    max_seq_len = 10
    hidden_size_rnn = 100
    use_stack_rnn = False


if __name__ == '__main__':
    conf = Config()
    print(len(conf.vocab_map))
    pass
