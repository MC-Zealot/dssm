# coding=utf8
"""
python=3.5
TensorFlow=1.2.1
"""
from scipy import sparse
import random
import time
import numpy as np
import tensorflow as tf
import data_input
from config import Config

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', 'Summaries', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 80000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 2000, "Number of steps in one epoch.")
flags.DEFINE_integer('pack_size', 2000, "Number of batches in one pickle pack.")
flags.DEFINE_integer('test_pack_size', 200, "Number of batches in one pickle pack.")
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")


start = time.time()

TRIGRAM_D = 21128
# TRIGRAM_D = 100
# negative sample
NEG = 4
# query batch size
query_BS = 100
# batch size
BS = query_BS * NEG
L1_N = 400
L2_N = 120

# 读取数据
conf = Config()
data_train = data_input.get_data(conf.file_train)
# print ("data_train['query'] len: ", len(data_train['query']),", data: ", data_train['query'])
print ("data_train['query'] len: ", len(data_train['query']))
data_vali = data_input.get_data(conf.file_vali)
# print ("data_vali['query'] len: ", len(data_vali['query']),", data: ", data_vali['query'])
print ("data_vali['query'] len: ", len(data_vali['query']))
# print(len(data_train['query']), query_BS, len(data_train['query']) / query_BS)
train_epoch_steps = int(len(data_train['query']) / query_BS) - 1
vali_epoch_steps = int(len(data_vali['query']) / query_BS) - 1
print ("train_epoch_steps: ", train_epoch_steps)
print ("vali_epoch_steps: ", vali_epoch_steps)


def batch_normalization(x, phase_train, out_size):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        out_size:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[out_size]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_size]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


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
    query_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='query_batch')
    doc_positive_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='doc_positive_batch')
    doc_negative_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='doc_negative_batch')
    on_train = tf.placeholder(tf.bool)
    # drop_out_prob = tf.placeholder(tf.float32, name='drop_out_prob')
    # query_seq_length = tf.placeholder(tf.int32, shape=[None], name='query_sequence_length')
    # pos_seq_length = tf.placeholder(tf.int32, shape=[None], name='pos_seq_length')
    # neg_seq_length = tf.placeholder(tf.int32, shape=[None], name='neg_sequence_length')


with tf.name_scope('FC1'):
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
    weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    variable_summaries(weight1, 'L1_weights')
    variable_summaries(bias1, 'L1_biases')

    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
    doc_positive_l1 = tf.sparse_tensor_dense_matmul(doc_positive_batch, weight1) + bias1
    doc_negative_l1 = tf.sparse_tensor_dense_matmul(doc_negative_batch, weight1) + bias1


with tf.name_scope('BN1'):
    query_l1 = batch_normalization(query_l1, on_train, L1_N)
    doc_l1 = batch_normalization(tf.concat([doc_positive_l1, doc_negative_l1], axis=0), on_train, L1_N)
    doc_positive_l1 = tf.slice(doc_l1, [0, 0], [query_BS, -1])
    doc_negative_l1 = tf.slice(doc_l1, [query_BS, 0], [-1, -1])
    query_l1_out = tf.nn.relu(query_l1)
    doc_positive_l1_out = tf.nn.relu(doc_positive_l1)
    doc_negative_l1_out = tf.nn.relu(doc_negative_l1)



with tf.name_scope('FC2'):
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(weight2, 'L2_weights')
    variable_summaries(bias2, 'L2_biases')

    query_l2 = tf.matmul(query_l1_out, weight2) + bias2
    doc_positive_l2 = tf.matmul(doc_positive_l1_out, weight2) + bias2
    doc_negative_l2 = tf.matmul(doc_negative_l1_out, weight2) + bias2

    query_l2 = batch_normalization(query_l2, on_train, L2_N)


with tf.name_scope('BN2'):
    doc_l2 = batch_normalization(tf.concat([doc_positive_l2, doc_negative_l2], axis=0), on_train, L2_N)
    doc_positive_l2 = tf.slice(doc_l2, [0, 0], [query_BS, -1])
    doc_negative_l2 = tf.slice(doc_l2, [query_BS, 0], [-1, -1])

    query_y = tf.nn.relu(query_l2)
    doc_positive_y = tf.nn.relu(doc_positive_l2)
    doc_negative_y = tf.nn.relu(doc_negative_l2)
    # query_y = tf.contrib.slim.batch_norm(query_l2, activation_fn=tf.nn.relu)

with tf.name_scope('Merge_Negative_Doc'):
    # 合并负样本，tile可选择是否扩展负样本。
    doc_y = tf.tile(doc_positive_y, [1, 1])
    # doc_y = tf.tile(doc_pos_rnn_output, [1, 1])

    for i in range(NEG):
        for j in range(query_BS):
            # slice(input_, begin, size)切片API
            doc_y = tf.concat([doc_y, tf.slice(doc_negative_y, [j * NEG + i, 0], [1, -1])], 0)
            # doc_y = tf.concat([doc_y, tf.slice(doc_neg_rnn_output, [j * NEG + i, 0], [1, -1])], 0)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    # query_norm = sqrt(sum(each x^2))
    # query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_rnn_output), 1, True)), [NEG + 1, 1])
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    # doc_norm = sqrt(sum(each x^2))
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    # prod = tf.reduce_sum(tf.multiply(tf.tile(query_rnn_output, [NEG + 1, 1]), doc_y), 1, True)
    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    # cos_sim_raw = query * doc / (||query|| * ||doc||)
    cos_sim_raw = tf.truediv(prod, norm_prod)
    # gamma = 20
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, query_BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    # 转化为softmax概率矩阵。
    prob = tf.nn.softmax(cos_sim)
    # 只取第一列，即正样本列概率。
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob))
    tf.summary.scalar('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.AdamOptimizer(conf.learning_rate).minimize(loss)

# with tf.name_scope('Accuracy'):
#     correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)

with tf.name_scope('Train'):
    train_average_loss = tf.placeholder(tf.float32)
    train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)


def pull_all(query_in, doc_positive_in, doc_negative_in):
    query_in = sparse.coo_matrix(query_in)
    doc_positive_in = sparse.coo_matrix(doc_positive_in)
    doc_negative_in = sparse.coo_matrix(doc_negative_in)

    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array(query_in.shape, dtype=np.int64))
    doc_positive_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_positive_in.row, dtype=np.int64), np.array(doc_positive_in.col, dtype=np.int64)]),
        np.array(doc_positive_in.data, dtype=np.float),
        np.array(doc_positive_in.shape, dtype=np.int64))
    doc_negative_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_negative_in.row, dtype=np.int64), np.array(doc_negative_in.col, dtype=np.int64)]),
        np.array(doc_negative_in.data, dtype=np.float),
        np.array(doc_negative_in.shape, dtype=np.int64))

    return query_in, doc_positive_in, doc_negative_in


def pull_batch(data_map, batch_id):
    query_in_tmp=np.mat(data_map['query'])
    doc_positive_in_tmp=np.mat(data_map['doc_pos'])
    doc_negative_in_tmp=np.mat(data_map['doc_neg'])
    query_in = query_in_tmp[batch_id * query_BS:(batch_id + 1) * query_BS, :]
    doc_positive_in = doc_positive_in_tmp[batch_id * query_BS:(batch_id + 1) * query_BS, :]
    doc_negative_in = doc_negative_in_tmp[batch_id * query_BS * NEG:(batch_id + 1) * query_BS * NEG, :]

    query_in, doc_positive_in, doc_negative_in = pull_all(query_in, doc_positive_in, doc_negative_in)

    return query_in, doc_positive_in, doc_negative_in


def feed_dict(on_training, Train, batch_id):
    """
    和定义的输入的数据结构相对应，互相绑定
    :param on_training:
    :param Train:
    :param batch_id:
    :return:
    """
    if Train:
        # batch_id = int(random.random() * (FLAGS.epoch_steps - 1))
        query_in, doc_positive_in, doc_negative_in = pull_batch(data_train, batch_id)
    else:
        query_in, doc_positive_in, doc_negative_in = pull_batch(data_vali, batch_id)
    return {query_batch: query_in,
            doc_positive_batch: doc_positive_in,
            doc_negative_batch: doc_negative_in,
            on_train: on_training}

# config = tf.ConfigProto()  # log_device_placement=True)
# config.gpu_options.allow_growth = True
# if not config.gpu:
# config = tf.ConfigProto(device_count= {'GPU' : 0})


config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1,
                log_device_placement=True)


# 创建一个Saver对象，选择性保存变量或者模型。
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #变量声明
    train_writer = tf.summary.FileWriter(conf.summaries_dir + '/train', sess.graph)

    start = time.time()
    for epoch in range(conf.num_epoch):
        batch_ids = [i for i in range(train_epoch_steps)]
        print ("batch_ids: ", batch_ids)
        random.shuffle(batch_ids)
        for batch_id in batch_ids:
            print("train batch_id:", batch_id)
            sess.run(train_step, feed_dict=feed_dict(True,True, batch_id))#模型训练
        end = time.time()
        # train loss下边是来计算损失，打印结果，不参与模型训练
        epoch_loss = 0
        for i in range(train_epoch_steps):
            print("train 2 batch_id:", batch_id,", i: ",i)
            loss_v = sess.run(loss, feed_dict=feed_dict(False, True, i))
            epoch_loss += loss_v

        epoch_loss /= (train_epoch_steps)
        train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
        train_writer.add_summary(train_loss, epoch + 1)
        print("\nEpoch #%d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" % (epoch, epoch_loss, end - start))

        # test loss
        start = time.time()
        epoch_loss = 0
        for i in range(vali_epoch_steps):
            print("test batch_id:", batch_id,", i: ",i)
            loss_v = sess.run(loss, feed_dict=feed_dict(False, False, i))
            epoch_loss += loss_v
        epoch_loss /= (vali_epoch_steps)
        test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
        train_writer.add_summary(test_loss, epoch + 1)
        # test_writer.add_summary(test_loss, step + 1)
        print("Epoch #%d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
              (epoch, epoch_loss, start - end))

    # 保存模型
    save_path = saver.save(sess, "model/model_1.ckpt")
    print("Model saved in file: ", save_path)
