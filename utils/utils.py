#!/usr/bin/env python
# encoding=utf-8
# from semantic_matching.dssm.config import Config
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import json
import random
import math
import sys
import re
import jieba
import io

# 配置文件
# conf = Config()


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()

    return tf.SparseTensorValue(indices, coo.data, coo.shape)


def pull_batch_drop_out(on_training, query_data, doc_data, doc_neg_data, batch_idx, BS, query_batch, doc_pos_batch,
               doc_neg_batch, on_train_batch,conf,keep_prob,conf_keep_prob=1.0):
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
    return {query_batch: query_in, doc_pos_batch: doc_pos_in, doc_neg_batch: doc_neg_in, on_train_batch: on_training,
            keep_prob: conf_keep_prob}

def pull_batch(on_training, query_data, doc_data, doc_neg_data, batch_idx, BS, query_batch, doc_pos_batch,
               doc_neg_batch, on_train_batch,conf):
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
    return {query_batch: query_in, doc_pos_batch: doc_pos_in, doc_neg_batch: doc_neg_in, on_train_batch: on_training
            }


def pull_batch_rnn(on_training, query_data, doc_data, doc_neg_data, batch_idx, BS, query_batch, doc_pos_batch,
               doc_neg_batch, on_train_batch, query_seq_length, neg_seq_length,pos_seq_length, TRIGRAM_D,conf):
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_pos_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_neg_in = doc_neg_data[batch_idx * BS * conf.NEG:(batch_idx + 1) * BS * conf.NEG, :]
    # pos_seq_len = doc_data[batch_idx * BS:(batch_idx + 1) * BS]
    # neg_seq_len = doc_data[batch_idx * BS * conf.NEG:(batch_idx + 1) * BS * conf.NEG]

    # print("batch_idx: ",batch_idx, "query_in shape: ", np.shape(query_in))
    # print("batch_idx: ",batch_idx, "doc_neg_in shape: ", np.shape(doc_neg_in))
    # query_in = convert_sparse_matrix_to_sparse_tensor(query_in)
    # doc_pos_in = convert_sparse_matrix_to_sparse_tensor(doc_pos_in)
    # doc_neg_in = convert_sparse_matrix_to_sparse_tensor(doc_neg_in)

    print("query_in type: ", type(query_in.indices), ", type: ", query_in.indices)
    print("query_in shape: ", query_in._shape)
    print("query_in values: ", query_in.data)
    query_in_dense = tf.sparse_to_dense(query_in.indices, query_in._shape, query_in.data)
    doc_pos_in = tf.sparse_to_dense(doc_pos_in.indices, doc_pos_in._shape, doc_pos_in.data)
    doc_neg_in = tf.sparse_to_dense(doc_neg_in.indices, doc_neg_in._shape, doc_neg_in.data)
    # query_in = tf.sparse.to_dense(query_in)
    # doc_pos_in = tf.sparse.to_dense(doc_pos_in)
    # doc_neg_in = tf.sparse.to_dense(doc_neg_in)
    query_len = query_in_dense._shape[0]
    query_seq_len = [conf.max_seq_len] * query_len
    pos_seq_len = [conf.max_seq_len] * query_len
    neg_seq_len = [conf.max_seq_len] * query_len * conf.NEG
    return {query_batch: query_in_dense, doc_pos_batch: doc_pos_in, doc_neg_batch: doc_neg_in, on_train_batch: on_training,
            query_seq_length: query_seq_len,
            neg_seq_length: neg_seq_len,
            pos_seq_length: pos_seq_len}


def GetActDat_v2(FileName,conf):
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


def GetActDat(FileName,conf):
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


def get_data_set(FileName, conf):
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


def save_vectorizer(vectorizer,path='output/vectorizer_data'):
    """
    保存字典文件
    :param vectorizer:
    :param path:
    :return:
    """
    modelFileSave = open(path, 'wb')
    pickle.dump(vectorizer, modelFileSave)
    modelFileSave.close()


def load_vectorizer(path='output/vectorizer_data'):
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


def test_case_for_cal_similarity(conf):
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


def get_data_set_comment(FileName, conf):
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
def pre_process_blank(line):
    if line is None:
        return line
    #去掉两端空白
    # line = line.strip()
    #判断是否有短链，如果有则去掉
    reg_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配模式
    urls = re.findall(reg_pattern, line)
    if len(urls) != 0:
        line = line.replace(urls[0], "")
        for index in range(len(urls)):
            line = line.replace(urls[index], "")
    #line = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", line)
    return line

def get_data(file_path,conf):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
    """
    # conf = Config()
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
                cur_arr.append(convert_word2id(each, conf.vocab_map, conf))
                each_len = len(each) if len(each) < conf.max_seq_len else conf.max_seq_len
                cur_len.append(each_len)
            if len(cur_arr) >= 4:
                data_map['query'].append(convert_word2id(prefix, conf.vocab_map, conf))
                data_map['query_len'].append(len(prefix) if len(prefix) < conf.max_seq_len else conf.max_seq_len)
                data_map['doc_pos'].append(convert_word2id(title, conf.vocab_map, conf))  #点击的query当做正例
                data_map['doc_pos_len'].append(len(title) if len(title) < conf.max_seq_len else conf.max_seq_len)
                data_map['doc_neg'].extend(cur_arr[:4])  #只取前4个负例
                data_map['doc_neg_len'].extend(cur_len[:4])
            # print("query_in shape....: ", np.shape(data_map['query']))
            pass
    return data_map

def convert_word2id(query, vocab_map, conf):
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


# 创建停用词list
def stopwords_list(filepath):
    stopwords = [line.strip() for line in io.open(filepath, 'r',encoding='utf-8').readlines()]
    return stopwords


#切词
def cut_words(line,conf):
    words = jieba.cut(line, cut_all=False)
    # 去停用词
    stopwords = stopwords_list(conf.stopwords_path)
    # for key in words:
    #      print key.word,key.flag
    after_tingyongci_words = []
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    for word in words:
        word = re.sub(r, '', word).strip()
        if word not in stopwords:
            if word != '\t' and len(word)>0:
                after_tingyongci_words.append(word)
    return " ".join(after_tingyongci_words)


def get_data_set_comment_cut_words(FileName, conf):
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

            prefix = pre_process_blank(prefix)
            title = pre_process_blank(title)

            prefix = cut_words(prefix, conf)
            title = cut_words(title, conf)

            # prefix = [i for i in prefix]
            # title = [i for i in title]
            # prefix = " ".join(prefix)
            # title = " ".join(title)

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
    # file_train = './data/comment/trainset_20190515_20190521_mini.txt'
    # from semantic_matching.dssm.config import Config
    # conf = Config()
    # query, doc, doc_neg = get_data_set_comment(file_train, conf)
    # idx = 0
    # print("query: ",query[idx])
    # print("doc: ",doc[idx])
    # print("doc_neg: ",doc_neg[idx][0])
    # print("doc_neg: ",doc_neg[idx][1])
    # print("doc_neg: ",doc_neg[idx][2])
    # print("doc_neg: ",doc_neg[idx][3])
    sess = tf.InteractiveSession()

    embedding = tf.Variable(np.identity(6, dtype=np.int32))
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
    input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

    sess.run(tf.global_variables_initializer())
    print(sess.run(embedding))
    print(sess.run(input_embedding, feed_dict={input_ids: [4, 0, 2, 4, 5, 1, 3, 0]}))

