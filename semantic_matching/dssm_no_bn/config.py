#!/usr/bin/env python
# encoding=utf-8


class Config(object):
    def to_string(self):
        print("conf params: ")
        hyper_params = self.__dict__
        for key in hyper_params:
            print(str(key) + ": " + str(hyper_params[key]))

    def __init__(self):
        self.vocab_path = '../../data/vocab.txt'
        self.file_train = '../../data/comment/trainset_repeat_20190508_20190514_shuffle.txt'
        #self.file_train = '../../data/dataset_20190508_20190514_2w.txt'
        #self.file_vali = '../../data/dataset_vali_20190515_20190515_5k.txt'
        self.file_vali = '../../data/comment/testset_repeat_20190515_20190515_shuffle.txt'
        self.L1_N = 1000
        self.L2_N = 300
        self.query_BS = 256
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
        self.stopwords_path = '../../data/stopwords.txt'
        self.to_string()


if __name__ == '__main__':
    conf = Config()
    print(len(conf.vocab_map))
    pass
