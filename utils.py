#!/usr/bin/env python
# encoding=utf-8
import json
from config import Config
import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf

# 配置文件
conf = Config()


def gen_word_set(file_path, out_path='./data/words.txt'):
    word_set = set()
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            cur_arr = [prefix, title]
            query_pred = json.loads(query_pred)
            for w in prefix:
                word_set.add(w)
            for each in query_pred:
                for w in each:
                    word_set.add(w)
    with open(word_set, 'w', encoding='utf8') as o:
        for w in word_set:
            o.write(w + '\n')
    pass


def convert_word2id(query, vocab_map):
    """
    从生成好的字典里找到字的id
    :param query:
    :param vocab_map:
    :return:
    """
    ids = []
    for w in query:
        if w in vocab_map:
            ids.append(vocab_map[w])
        else:
            ids.append(vocab_map[conf.unk])
    while len(ids) < conf.max_seq_len: #如果query少于max_seq_len，则填充0，满足长度
        ids.append(vocab_map[conf.pad])
    ids = np.array(ids)
    return ids[:conf.max_seq_len]


def convert_word2id_for_dssm(query, vocab_map):
    """
    从生成好的字典里找到字的id,用在dssm
    :param query:
    :param vocab_map:
    :return:
    """
    ids = []
    ids = np.array(ids)
    for w in query:
        if w in vocab_map:
            ids.append(vocab_map[w])
        else:
            ids.append(vocab_map[conf.unk])
    while len(ids) < conf.max_seq_len: #如果query少于max_seq_len，则填充0，满足长度
        ids.append(vocab_map[conf.pad])
    return ids[:conf.max_seq_len]

def get_data_by_dssm(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
    """
    data_map = {'query': [], 'query_len': [], 'doc_pos': [],  'doc_pos_len': [], 'doc_neg': [], 'doc_neg_len': []}
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            cur_arr, cur_len = [], []
            query_pred = json.loads(query_pred)
            # only 4 negative sample
            for each in query_pred: #从预测的query中，找4个负例
                if each == title:
                    continue
                cur_arr.append(convert_word2id(each, conf.vocab_map))
                each_len = len(each) if len(each) < conf.max_seq_len else conf.max_seq_len
                cur_len.append(each_len)
            if len(cur_arr) >= 4:
                data_map['query'].append(convert_word2id(prefix, conf.vocab_map))
                data_map['query_len'].append(len(prefix) if len(prefix) < conf.max_seq_len else conf.max_seq_len)
                data_map['doc_pos'].append(convert_word2id(title, conf.vocab_map))  #点击的query当做正例
                data_map['doc_pos_len'].append(len(title) if len(title) < conf.max_seq_len else conf.max_seq_len)
                data_map['doc_neg'].extend(cur_arr[:4])  #只取前4个负例
                data_map['doc_neg_len'].extend(cur_len[:4])
            # print("query_in shape....: ", np.shape(data_map['query']))
            pass

    data_map['query'] = csr_matrix(data_map['query'],(np.shape(data_map['query'])[0], conf.max_seq_len), dtype=np.float32)
    data_map['doc_pos'] = csr_matrix(data_map['doc_pos'],(np.shape(data_map['doc_pos'])[0], conf.max_seq_len), dtype=np.float32)
    data_map['doc_neg'] = csr_matrix(data_map['doc_neg'],(np.shape(data_map['doc_neg'])[0], conf.max_seq_len), dtype=np.float32)

    return data_map

def get_data(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
    """
    data_map = {'query': [], 'query_len': [], 'doc_pos': [],  'doc_pos_len': [], 'doc_neg': [], 'doc_neg_len': []}
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            cur_arr, cur_len = [], []
            query_pred = json.loads(query_pred)
            # only 4 negative sample
            for each in query_pred: #从预测的query中，找4个负例
                if each == title:
                    continue
                cur_arr.append(convert_word2id(each, conf.vocab_map))
                each_len = len(each) if len(each) < conf.max_seq_len else conf.max_seq_len
                cur_len.append(each_len)
            if len(cur_arr) >= 4:
                data_map['query'].append(convert_word2id(prefix, conf.vocab_map))
                data_map['query_len'].append(len(prefix) if len(prefix) < conf.max_seq_len else conf.max_seq_len)
                data_map['doc_pos'].append(convert_word2id(title, conf.vocab_map))  #点击的query当做正例
                data_map['doc_pos_len'].append(len(title) if len(title) < conf.max_seq_len else conf.max_seq_len)
                data_map['doc_neg'].extend(cur_arr[:4])  #只取前4个负例
                data_map['doc_neg_len'].extend(cur_len[:4])
            # print("query_in shape....: ", np.shape(data_map['query']))
            pass
    return data_map


# def GetActDat_v2(FileName):
#     query = []
#     doc = []
#     with open(FileName, 'r') as f:
#         for line in f:
#             s_line = line.strip().split("\t")
#
#             if len(s_line) != 4:
#                 continue
#
#             query.append(s_line[3])
#             doc.append(s_line[1])
#
#     return query, doc


def GetActDat(FileName):
    # Suppose the file stores the InterActDat
    # in the form of ad_mid - sep_words - bhv_mid - sep_words
    # where fields are separated by '\t'
    # This function returns the formatted data for CounterVectorizer input
    query = []
    doc = []

    with open(FileName, 'r') as f:
        for line in f:
            s_line = line.strip().split('\t')

            if len(s_line) != 4:
                continue

            query_i = []
            for i in s_line[3].split('\001'):  # query are bhv_mid
                if len(i.split('/')) == 3:
                    query_i.append(i.split('/')[0])
            doc_i = []
            for i in s_line[1].split('\001'):  # doc are ad_mid
                if len(i.split('/')) == 3:
                    doc_i.append(i.split('/')[0] + '\t')
            if query_i and doc_i:
                query.append('\t'.join(query_i))
                doc.append('\t'.join(doc_i))

    return query, doc



def convert_sparse_matrix_to_sparse_tensor(X):
	coo = X.tocoo()
	indices = np.mat([coo.row, coo.col]).transpose()
	return tf.SparseTensorValue(indices, coo.data, coo.shape)


# def pull_batch(query_data, doc_data, batch_idx, BS, query_batch, doc_batch):
# 	print ("pull batch %d-%d" % (batch_idx * BS, (batch_idx + 1) * BS))
# 	query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
# 	print("query_in shape: ", np.shape(query_in))
# 	doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
# 	query_in = convert_sparse_matrix_to_sparse_tensor(query_in)
# 	print ("query_in[0]: ", query_in[0], ", query_in[1]: ", query_in[1], ",query_in[2]: ", query_in[2])
# 	return {query_batch: query_in, doc_batch: convert_sparse_matrix_to_sparse_tensor(doc_in)}


def pull_batch(on_training,query_data, doc_data,doc_neg_data, batch_idx, BS, query_batch, doc_pos_batch,doc_neg_batch,on_train_batch):
    # print ("pull batch %d-%d" % (batch_idx * BS, (batch_idx + 1) * BS))
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_pos_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    print("query_in shape: ", np.shape(query_in))
    query_in = convert_sparse_matrix_to_sparse_tensor(query_in)
    doc_neg_in = doc_neg_data[batch_idx * BS * conf.NEG:(batch_idx + 1) * BS * conf.NEG, :]

    doc_pos_in = convert_sparse_matrix_to_sparse_tensor(doc_pos_in)
    doc_neg_in = convert_sparse_matrix_to_sparse_tensor(doc_neg_in)
    print ("query_in[0]: ", query_in[0], ", query_in[1]: ", query_in[1], ",query_in[2]: ", query_in[2])
    return {query_batch: query_in, doc_pos_batch: doc_pos_in, doc_neg_batch: doc_neg_in,on_train_batch: on_training}


def GetActDat_v2(FileName):
    query = []
    doc = []
    doc_neg=[]
    with open(FileName, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            query.append(prefix)
            doc.append(title)
            cur_arr = []
            query_pred = json.loads(query_pred)
            # only 4 negative sample
            for each in query_pred:  # 从预测的query中，找4个负例
                if each == title:
                    continue
                cur_arr.append(each)
            if len(cur_arr) >= 4:
                query.append(prefix)
                doc.append(title)
                doc_neg.extend(cur_arr[:conf.NEG])


    return query, doc,doc_neg

if __name__ == '__main__':
    # prefix, query_prediction, title, tag, label
    # query_prediction 为json格式。
    file_train = './data/oppo_round1_train_20180929.txt'
    file_vali = './data/oppo_round1_vali_20180929.txt'
    # data_train = get_data(file_train)
    # data_train = get_data(file_vali)
    # print(len(data_train['query']), len(data_train['doc_pos']), len(data_train['doc_neg']))
    pass
