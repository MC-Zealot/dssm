#!/usr/bin/env python
# encoding=utf-8
import json
from config import Config
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import ast

# 配置文件
conf = Config()


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data, coo.shape)


def pull_batch(on_training, query_data, doc_data, doc_neg_data, batch_idx, BS, query_batch, doc_pos_batch,
               doc_neg_batch, on_train_batch):
    # print("batch_idx: ",batch_idx, "doc_data shape",np.shape(doc_data))
    # print("batch_idx: ",batch_idx, "doc_neg_data shape",np.shape(doc_neg_data))
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_pos_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_neg_in = doc_neg_data[batch_idx * BS * conf.NEG:(batch_idx + 1) * BS * conf.NEG, :]
    # print("batch_idx: ",batch_idx, "query_in shape: ", np.shape(query_in))
    # print("batch_idx: ",batch_idx, "doc_neg_in shape: ", np.shape(doc_neg_in))
    query_in = convert_sparse_matrix_to_sparse_tensor(query_in)
    doc_pos_in = convert_sparse_matrix_to_sparse_tensor(doc_pos_in)
    doc_neg_in = convert_sparse_matrix_to_sparse_tensor(doc_neg_in)
    # print ("query_in[0]: ", query_in[0], ", query_in[1]: ", query_in[1], ",query_in[2]: ", query_in[2])
    # print ("doc_pos_in[0]: ", doc_pos_in[0], ", doc_pos_in[1]: ", doc_pos_in[1], ",doc_pos_in[2]: ", doc_pos_in[2])
    # print ("doc_neg_in[0]: ", doc_neg_in[0], ", doc_neg_in[1]: ", doc_neg_in[1], ",doc_neg_in[2]: ", doc_neg_in[2])
    return {query_batch: query_in, doc_pos_batch: doc_pos_in, doc_neg_batch: doc_neg_in, on_train_batch: on_training}


def GetActDat_v2(FileName):
    """

    :param FileName:
    :return:query, doc, doc_neg
    """
    query = []
    doc = []
    doc_neg = []
    with open(FileName, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            # query.append(prefix)
            # doc.append(title)
            cur_arr = []
            query_pred = json.loads(query_pred)
            # only 4 negative sample
            for each in query_pred:  # 从预测的query中，找4个负例
                if each == title:
                    continue
                each = [i for i in each]
                each = " ".join(each)
                # print ("each: ", each)
                cur_arr.append(each)
            if len(cur_arr) >= 4:
                prefix = [i for i in prefix]
                title = [i for i in title]
                prefix = " ".join(prefix)
                title = " ".join(title)
                query.append(prefix)
                doc.append(title)
                doc_neg.extend(cur_arr[:conf.NEG])

    return query, doc, doc_neg


def test1():
    # prefix, query_prediction, title, tag, label
    # query_prediction 为json格式。
    file_train = './data/oppo_round1_train_20180929_mini.txt.bak'
    bhv_act, ad_act, ac_act_neg = GetActDat_v2(file_train)
    print("len: ", len(bhv_act), ", bhv_act: ", bhv_act)
    bhv_act = set(bhv_act)
    print("len: ", len(bhv_act), ", bhv_act: ", bhv_act)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    # vectorizer = CountVectorizer()
    vectorizer.fit(bhv_act)
    feature_names = vectorizer.get_feature_names()
    TRIGRAM_D = len(feature_names)  # 词库大小，aka 稀疏矩阵列数
    print("feature_names: ", feature_names)
    print("TRIGRAM_D: ", TRIGRAM_D)


def save_vectorizer(vectorizer,path='data/vectorizer_data'):
    """
    保存字典文件
    :param vectorizer:
    :param path:
    :return:
    """
    modelFileSave = open(path, 'wb')
    pickle.dump(vectorizer, modelFileSave)
    modelFileSave.close()


def load_vectorizer(path='data/vectorizer_data'):
    """
    加载字典文件
    :param path:
    :return:
    """
    modelFileLoad = open(path, 'rb')
    vec = pickle.load(modelFileLoad)
    return vec


def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)


def cosine_similarity(vector_map_1, vector_map_2):
    vector_map_1 = "{" + vector_map_1 + "}"
    vector_map_2 = "{" + vector_map_2 + "}"
    vector_map_1 = ast.literal_eval(vector_map_1)
    vector_map_2 = ast.literal_eval(vector_map_2)

    dot_product = 0.0
    normA = 0.0
    normB = 0.0

    for index,value in vector_map_1:
        if vector_map_2[index] is not None:
            dot_product += vector_map_1[index] * vector_map_2[index]
            normA += vector_map_1[index] ** 2
            normB += vector_map_2[index] ** 2

    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)




if __name__ == '__main__':
    print("hello")
    print (ast.literal_eval("{'0' : '0.041', '2' : '0.837'}"))