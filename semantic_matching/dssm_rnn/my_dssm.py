# coding=utf8
"""
python=3.6
TensorFlow=1.6
无bn版本
"""
import sys
sys.path.append("../../")
from utils.utils import *
import random
import time
import numpy as np
import tensorflow as tf

from semantic_matching.dssm_rnn.config import Config
from sklearn.feature_extraction.text import CountVectorizer

start = time.time()

# 读取数据
conf = Config()
query_BS = conf.query_BS

L1_N = conf.L1_N
L2_N = conf.L2_N

# The part below shouldn't be commented for everyday training
# utilize the CountVectorizer() object to transform the successfully-interacted bhv & ad words as raw vectors

bhv_act, ad_act, ad_act_neg = get_data_set_comment(conf.file_train, conf)
# bhv_act, ad_act, ad_act_neg = utils.GetActDat_v2(conf.file_train)
# exit(0)
bhv_act_test, ad_act_test, ad_act_neg_test = get_data_set_comment(conf.file_vali, conf)
# bhv_act_test, ad_act_test, ad_act_neg_test  = utils.GetActDat_v2(conf.file_vali)
print ("data_train['query'] len: ", np.shape(bhv_act))
## Establish Vectorizer and transform the raw word input into sparse matrix
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
vectorizer.fit(ad_act + bhv_act + ad_act_neg)
save_vectorizer(vectorizer)

query_train_dat = vectorizer.transform(bhv_act)
print (type(query_train_dat))
doc_train_dat = vectorizer.transform(ad_act)
doc_neg_train_dat = vectorizer.transform(ad_act_neg)
query_vali_dat = vectorizer.transform(bhv_act_test)
doc_vali_dat = vectorizer.transform(ad_act_test)
doc_neg_vali_dat = vectorizer.transform(ad_act_neg_test)
TRIGRAM_D = len(vectorizer.get_feature_names()) # 词库大小，aka 稀疏矩阵列数

train_epoch_steps = int(query_train_dat.shape[0] / query_BS) - 1 # = number of samples / batch_size
print ("train_epoch_steps:", train_epoch_steps)
vali_epoch_steps = int(query_vali_dat.shape[0] / query_BS) - 1 # = number of samples / batch_size
print ("vali_epoch_steps:", vali_epoch_steps)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)




with tf.name_scope('input'):
    # 预测时只用输入query即可，将其embedding为向量。
    print ("TRIGRAM_D: ",TRIGRAM_D)
    #定义数据结构，类型、shape
    # query_batch = tf.placeholder(tf.int32, shape=[None, TRIGRAM_D], name='query_batch')
    # doc_positive_batch = tf.placeholder(tf.int32, shape=[None, TRIGRAM_D], name='doc_positive_batch')
    # doc_negative_batch = tf.placeholder(tf.int32, shape=[None, TRIGRAM_D], name='doc_negative_batch')
    # on_train = tf.placeholder(tf.bool, name='on_train')

    query_batch = tf.placeholder(tf.int32, shape=[None, None], name='query_batch')
    doc_pos_batch = tf.placeholder(tf.int32, shape=[None, None], name='doc_positive_batch')
    doc_neg_batch = tf.placeholder(tf.int32, shape=[None, None], name='doc_negative_batch')
    query_seq_length = tf.placeholder(tf.int32, shape=[None], name='query_sequence_length')
    pos_seq_length = tf.placeholder(tf.int32, shape=[None], name='pos_seq_length')
    neg_seq_length = tf.placeholder(tf.int32, shape=[None], name='neg_sequence_length')
    on_train = tf.placeholder(tf.bool)


K = 500

with tf.name_scope('word_embeddings_layer'):
    _word_embedding = tf.get_variable(name="word_embedding_arr", dtype=tf.float32, shape=[TRIGRAM_D, K])
    query_embed = tf.nn.embedding_lookup(_word_embedding, query_batch, name='query_batch_embed')
    doc_pos_embed = tf.nn.embedding_lookup(_word_embedding, doc_pos_batch, name='doc_positive_embed')
    doc_neg_embed = tf.nn.embedding_lookup(_word_embedding, doc_neg_batch, name='doc_negative_embed')


with tf.name_scope('RNN'):
    # Abandon bag of words, use GRU, you can use stacked gru
    # query_l1 = add_layer(query_batch, TRIGRAM_D, L1_N, activation_function=None)  # tf.nn.relu()
    # doc_positive_l1 = add_layer(doc_positive_batch, TRIGRAM_D, L1_N, activation_function=None)
    # doc_negative_l1 = add_layer(doc_negative_batch, TRIGRAM_D, L1_N, activation_function=None)
    cell_fw = tf.contrib.rnn.GRUCell(conf.hidden_size_rnn, reuse=tf.AUTO_REUSE)
    cell_bw = tf.contrib.rnn.GRUCell(conf.hidden_size_rnn, reuse=tf.AUTO_REUSE)
    # query
    (_, _), (query_output_fw, query_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, query_embed, sequence_length=query_seq_length, dtype=tf.float32)
    query_rnn_output = tf.concat([query_output_fw, query_output_bw], axis=-1)
    # query_rnn_output = tf.nn.dropout(query_rnn_output, drop_out_prob)
    # doc_pos
    (_, _), (doc_pos_output_fw, doc_pos_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, doc_pos_embed, sequence_length=pos_seq_length, dtype=tf.float32)
    doc_pos_rnn_output = tf.concat([doc_pos_output_fw, doc_pos_output_bw], axis=-1)
    # doc_pos_rnn_output = tf.nn.dropout(doc_pos_rnn_output, drop_out_prob)
    # doc_neg
    (_, _), (doc_neg_output_fw, doc_neg_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, doc_neg_embed, sequence_length=neg_seq_length, dtype=tf.float32)
    doc_neg_rnn_output = tf.concat([doc_neg_output_fw, doc_neg_output_bw], axis=-1)
    # doc_neg_rnn_output = tf.nn.dropout(doc_neg_rnn_output, drop_out_prob)

with tf.name_scope('Merge_Negative_Doc'):
    # 合并负样本，tile可选择是否扩展负样本。
    # doc_y = tf.tile(doc_positive_y, [1, 1])
    doc_y = tf.tile(doc_pos_rnn_output, [1, 1])

    for i in range(conf.NEG):
        for j in range(query_BS):
            # slice(input_, begin, size)切片API
            # doc_y = tf.concat([doc_y, tf.slice(doc_negative_y, [j * NEG + i, 0], [1, -1])], 0)
            doc_y = tf.concat([doc_y, tf.slice(doc_neg_rnn_output, [j * conf.NEG + i, 0], [1, -1])], 0)

        print("doc_y ", i, ": ", doc_y.shape)



with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    # query_norm = sqrt(sum(each x^2))
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_rnn_output), 1, True)), [conf.NEG + 1, 1])
    # doc_norm = sqrt(sum(each x^2))
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    prod = tf.reduce_sum(tf.multiply(tf.tile(query_rnn_output, [conf.NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    # cos_sim_raw = query * doc / (||query|| * ||doc||)
    cos_sim_raw = tf.truediv(prod, norm_prod)
    # gamma = 20
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [conf.NEG + 1, query_BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    # 转化为softmax概率矩阵。
    prob = tf.nn.softmax(cos_sim)#1、输出一下结构，2、修改结构，变成n*1，当做out
    # 只取第一列，即正样本列概率。
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob)) / query_BS

    # loss = -tf.reduce_sum(tf.log(tf.clip_by_value(hit_prob, 1e-8, 1.0)))#防止nan

    tf.summary.scalar('loss', loss)


with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.AdamOptimizer(conf.learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# with tf.name_scope('Auc'):
#     # insert by trobr
#     indices = tf.squeeze(tf.where(tf.less_equal(label_tensor, 2 - 1)), 1)
#     label_tensor = tf.cast(tf.gather(label_tensor, indices), tf.int32)
#     predictions = tf.gather(cos_sim_raw, indices)
#     # end of insert
#     auc_value, auc_op = tf.metrics.auc(label_tensor, predictions, num_thresholds=2000)
#     tf.summary.scalar('auc', auc_value)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)

with tf.name_scope('Train'):
    train_average_loss = tf.placeholder(tf.float32)
    train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)

config = tf.ConfigProto()
#config.log_device_placement=True
config.gpu_options.allow_growth = True


# 创建一个Saver对象，选择性保存变量或者模型。
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
#with tf.InteractiveSession(config=config) as sess:
    sess.run(tf.global_variables_initializer()) #变量声明
    sess.run(tf.local_variables_initializer())
    train_writer = tf.summary.FileWriter(conf.summaries_dir + '/train', sess.graph)

    start = time.time()
    for epoch in range(conf.num_epoch):
        view_bar("processing image of ", epoch + 1, conf.num_epoch)
        batch_ids = [i for i in range(train_epoch_steps)]
        # print ("batch_ids: ", batch_ids)
        random.shuffle(batch_ids)
        for batch_id in batch_ids:
            sess.run(train_step, feed_dict=pull_batch_rnn(
                True, query_train_dat, doc_train_dat,doc_neg_train_dat,
                batch_id, query_BS, query_batch, doc_pos_batch,doc_neg_batch, on_train,
                query_seq_length, neg_seq_length, pos_seq_length,TRIGRAM_D,
                conf))
        end = time.time()
        # train loss下边是来计算损失，打印结果，不参与模型训练
        epoch_loss = 0
        epoch_auc = 0
        for i in range(train_epoch_steps):

            loss_v = sess.run(loss, feed_dict=pull_batch_rnn(False, query_train_dat, doc_train_dat,doc_neg_train_dat, i, query_BS, query_batch, doc_pos_batch, doc_neg_batch,on_train, query_seq_length, neg_seq_length, pos_seq_length,TRIGRAM_D,conf))
            epoch_loss += loss_v

            # sess.run(auc_op, feed_dict=pull_batch(False, query_train_dat, doc_train_dat,doc_neg_train_dat, i, query_BS, query_batch, doc_positive_batch, doc_negative_batch,on_train))
            # auc_v=sess.run(auc_value, feed_dict=pull_batch(False, query_train_dat, doc_train_dat,doc_neg_train_dat, i, query_BS, query_batch, doc_positive_batch, doc_negative_batch,on_train))
            # epoch_auc += auc_v

            # print("train_loss epoch:", epoch, ", i: ", i, "loss_v: ", loss_v)
        #    print("epoch: ", epoch,", train_epoch_steps: ", i,", train_loss: ", loss_v,", auc: ", auc_v)


        epoch_loss /= (train_epoch_steps)
        epoch_auc /= (train_epoch_steps)
        train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
        train_writer.add_summary(train_loss, epoch + 1)
        print("\nEpoch #%d | Train Loss: %-4.3f | Train Auc: %-4.3f | PureTrainTime: %-3.3fs" % (epoch, epoch_loss,epoch_auc, end - start))

        # test loss
        start = time.time()
        epoch_loss = 0
        epoch_auc = 0
        for index in range(vali_epoch_steps):
            # print("test batch_id:", batch_id,", i: ",i)
            loss_v = sess.run(loss, feed_dict=pull_batch_rnn(False, query_vali_dat, doc_vali_dat, doc_neg_vali_dat, index,
                                                         query_BS, query_batch, doc_pos_batch, doc_neg_batch,
                                                         on_train, query_seq_length, neg_seq_length, pos_seq_length,TRIGRAM_D,conf))
            # print("test_loss epoch:", epoch, ", index: ", index,"loss_v: ",loss_v)
            epoch_loss += loss_v

            # sess.run(auc_op,feed_dict=pull_batch(False, query_vali_dat, doc_vali_dat, doc_neg_vali_dat, index, query_BS, query_batch, doc_positive_batch, doc_negative_batch, on_train))
            # auc_v = sess.run(auc_value, feed_dict=pull_batch(False, query_vali_dat, doc_vali_dat, doc_neg_vali_dat, index, query_BS, query_batch, doc_positive_batch, doc_negative_batch, on_train))
            # epoch_auc += auc_v

            # print("train_loss epoch:", epoch, ", i: ", i, "loss_v: ", loss_v)
#            print("epoch: ", epoch, ", test_epoch_steps: ", index, ", test_loss: ", loss_v, ", auc: ", auc_v)
        epoch_loss /= (vali_epoch_steps)
        epoch_auc /= (vali_epoch_steps)
        test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
        train_writer.add_summary(test_loss, epoch + 1)
        # test_writer.add_summary(test_loss, step + 1)
        print("Epoch #%d | Test  Loss: %-4.3f | Test Auc: %-4.3f| Calc_LossTime: %-3.3fs" % (epoch, epoch_loss,epoch_auc, start - end))

    # 保存模型
    save_path = saver.save(sess, "model/model_1.ckpt")
    print("Model saved in file: ", save_path)
