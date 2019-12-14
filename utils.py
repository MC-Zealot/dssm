#!/usr/bin/env python
# encoding=utf-8
import json
from config import Config
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import json
import random
import math
import sys
import re

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

def GetActDat(FileName):
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
                # if each == title:
                #     continue
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


def get_data_set(FileName):
    """
1、查看query结构，字符串，句子每个字以空格为分隔符，uni gram
2、查看doc正例与负例结构
3、先计算正例（保持不变）
4、再通过正例，随机选择NEG个当做负例。
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
            prefix = [i for i in prefix]
            title = [i for i in title]
            prefix = " ".join(prefix)
            title = " ".join(title)
            query.append(prefix)
            doc.append(title)
            # doc_neg.extend(cur_arr[:conf.NEG])
    size = len(doc)
    for i in range(size):
        # print(doc[i])
        for j in range(conf.NEG):
            r = random.random()
            r = int(r * size)
            doc_neg.append(doc[r])

    return query, doc, doc_neg


def test1():
    # prefix, query_prediction, title, tag, label
    # query_prediction 为json格式。
    file_train = './data/oppo_round1_train_20180929_mini.txt.bak'
    bhv_act, ad_act, ad_act_neg = GetActDat_v2(file_train)
    print("len: ", len(bhv_act), ", bhv_act: ", bhv_act)
    # bhv_act = set(bhv_act)
    print("len: ", len(bhv_act), ", bhv_act: ", bhv_act)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    # vectorizer = CountVectorizer()
    # vectorizer.fit(ad_act + bhv_act + ad_act_neg)
    vectorizer.fit(bhv_act)
    feature_names = vectorizer.get_feature_names()
    TRIGRAM_D = len(feature_names)  # 词库大小，aka 稀疏矩阵列数
    print("feature_names: ", feature_names)
    print("TRIGRAM_D: ", TRIGRAM_D)
    query_train_dat = vectorizer.transform(bhv_act)
    i=0

    # print(field for field in bhv_act[i])
    print( "bhv_act["+str(i)+"]",bhv_act[i], "query_train_dat["+str(i)+"]",query_train_dat[i])


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

    print("query vector_map: ", vector_map_2)
    print("doc vector_map: ", vector_map_1)
    vector_map_1 = str_to_dict(vector_map_1)
    vector_map_2 = str_to_dict(vector_map_2)

    dot_product = 0.0
    normA = 0.0
    normB = 0.0

    for index,value in vector_map_1.items():
        if index in vector_map_2:
            dot_product += vector_map_1[index] * vector_map_2[index]
            normA += vector_map_1[index] ** 2
            normB += vector_map_2[index] ** 2

    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)


def str_to_dict(str):
    # print("str: ",str)
    dict = {}
    if str is None or str == "":
        return dict
    fields = str.split(",")
    for field in fields:
        tmp = field.split(":")
        dict[tmp[0]] = float(tmp[1])
    return dict


def test_case_for_cal_similarity():
    # 1、打开query文件，加载数据到list[dict]中，
    #     # 2、打开doc_neg文件，加载数据到list[dict]中，
    #     # 3、根据index选择query-docs
    #     # 4、分别计算相似度并且打分，打印出来
    query_list = []
    doc_list = []
    query_file_name = conf.query_mid_vector_file
    doc_neg_y_mid_vector_file_name = conf.doc_neg_y_mid_vector_file
    with open(query_file_name, encoding='utf8') as f:
        for line in f.readlines():
            query_str, norm, query_vec = line.strip().split('\t')
            query_list.append((query_str, query_vec))
            # print (query_str,": ",query_vec)

    with open(doc_neg_y_mid_vector_file_name, encoding='utf8') as f:
        for line in f.readlines():
            doc_str, doc_vec = line.strip().split('\t')
            doc_list.append((doc_str, doc_vec))

    print("query_list len:", len(query_list), ", shape: ", np.shape(query_list))
    print("doc_list len:", len(doc_list), ", shape: ", np.shape(doc_list))

    index = 35
    query = query_list[index]
    docs = doc_list[index * conf.NEG:index * conf.NEG + conf.NEG]
    print("query len:", len(query), ", shape: ", query)
    print("docs len:", len(docs), ", shape: ", docs)
    print("docs[1]", docs[1])
    # exit(0)
    print("===================================================================")
    for i in range(len(docs)):
        doc = docs[i]
        doc_index_vec = doc[1]
        doc_index_str = doc[0]
        query_index_vec = query[1]
        query_index_str = query[0]

        score = cosine_similarity(doc_index_vec, query_index_vec)
        print("query_index_str:", query_index_str)
        print("doc_index_str:", doc_index_str)
        print("score: ", score)
        print("=====================")


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()
    print()


def get_data_set_comment(FileName):
    """
    评论流数据
1、查看query（正文博文）结构，字符串，句子每个字以空格为分隔符，uni gram
2、查看doc（广告博文）正例与负例结构
3、先计算正例（保持不变）
4、再通过正例，随机选择NEG个当做负例。
    :param FileName:
    :return:query, doc, doc_neg
    """
    query = []
    doc = []
    doc_neg = []
    with open(FileName, encoding='utf8') as f:
        for line in f.readlines():

            spline = line.split('\t')
            if len(spline) < 3:
                continue
            prefix,  title, label, mid, feed_id = spline
            if label != '1':
                print("wrong label:", line)
                continue
            prefix = pre_process(prefix)
            title = pre_process(title)

            prefix = [i for i in prefix]
            title = [i for i in title]
            prefix = " ".join(prefix)
            title = " ".join(title)

            query.append(prefix)
            doc.append(title)
            # doc_neg.extend(cur_arr[:conf.NEG])
    size = len(doc)
    for i in range(size):
        # print(doc[i])
        j = 0
        pos_content = doc[i]
        doc_neg_list=[]#负样本NEG个
        doc_neg_list=set(doc_neg_list)
        while j < conf.NEG:
            r = random.random()

            r = int(r * size)
            neg_content = doc[r] #随机选择NEG个负样本，如果和正样本ad相同，或者query相同，则pass
            if pos_content != neg_content and query[i] != query[r] and neg_content not in doc_neg_list:
                # doc_neg.append(neg_content)
                doc_neg_list.add(neg_content)
                j += 1
        doc_neg.extend(doc_neg_list)

    return query, doc, doc_neg


def pre_process(line):
    if line is None:
        return line
    #去掉两端空白
    line = line.strip()
    #判断是否有短链，如果有则去掉
    reg_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配模式
    urls = re.findall(reg_pattern, line)
    if len(urls) != 0:
        line = line.replace(urls[0], "")
        for index in range(len(urls)):
            line = line.replace(urls[index], "")
    line = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", line)
    return line


if __name__ == '__main__':
    print("hello")
    # vectorizer=load_vectorizer()
    # print(vectorizer.get_feature_names())
    # pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配模式

    # str = 'Its after 12 noon, do you know where your rooftops are? '
    # str = '高速车震……速看！！！！http://t.cn/RSNVzD8 ​​	  [馋嘴]玩了几天，这个月的花贝都还上了[弗莱见钱眼开]缺零花钱的可以来玩一下[来] http://t.cn/Eo3OKsf [弗莱见钱眼开][弗莱见钱眼开][弗莱见钱眼开][弗莱见钱眼开]玩每天赚点新闻赚钱辣么多零花钱随便你赚啦[嘻嘻][嘻嘻]	1	4373771843879773	4370362685017799 '
    # print(str)
    # url = re.findall(pattern, str)
    # print(url,len(url))
    # str=str.replace(url[0],"")
    # print(pre_process(str))
    # test_case_for_cal_similarity()
    file_train = './data/comment/trainset_20190515_20190521_mini.txt'
    query, doc, doc_neg = get_data_set_comment(file_train)
    idx = 0
    print("query: ",query[idx])
    print("doc: ",doc[idx])
    print("doc_neg: ",doc_neg[idx][0])
    print("doc_neg: ",doc_neg[idx][1])
    print("doc_neg: ",doc_neg[idx][2])
    print("doc_neg: ",doc_neg[idx][3])

