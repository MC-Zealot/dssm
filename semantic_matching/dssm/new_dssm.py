# coding=utf8
"""
python=3.6
TensorFlow=1.6
"""
import sys
sys.path.append("../utils")
import time
from utils.utils import *
from semantic_matching.dssm.config import Config
from sklearn.feature_extraction.text import CountVectorizer

start = time.time()

conf = Config()
print("conf: ",conf.__dict__)
print("conf: ", conf)
query_BS = conf.query_BS
L1_N = conf.L1_N
L2_N = conf.L2_N

# The part below shouldn't be commented for everyday training
# utilize the CountVectorizer() object to transform the successfully-interacted bhv & ad words as raw vectors
# 读取数据
bhv_act, ad_act, ad_act_neg = get_data_set_comment(conf.file_train)
# bhv_act, ad_act, ad_act_neg = GetActDat_v2(conf.file_train)
# exit(0)
bhv_act_test, ad_act_test, ad_act_neg_test  = get_data_set_comment(conf.file_vali)
# bhv_act_test, ad_act_test, ad_act_neg_test  = GetActDat_v2(conf.file_vali)
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
# data_train = data_input.get_data_by_dssm(conf.file_train)
# print ("data_train['query'] len: ", data_train['query'].shape[0])
# data_vali = data_input.get_data_by_dssm(conf.file_vali)
# print ("data_vali['query'] len: ", data_vali['query'].shape[0])
# train_epoch_steps = int( data_train['query'].shape[0] / query_BS) - 1
# vali_epoch_steps = int(data_vali['query'].shape[0] / query_BS) - 1
# print ("train_epoch_steps: ", train_epoch_steps)
# print ("vali_epoch_steps: ", vali_epoch_steps)


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

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
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
    # query_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='query_batch')
    # doc_positive_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='doc_positive_batch')
    # doc_negative_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='doc_negative_batch')
    query_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='query_batch')
    # print ("query_batch shape: ",query_batch.shape)
    doc_positive_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='doc_positive_batch')
    doc_negative_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='doc_negative_batch')
    on_train = tf.placeholder(tf.bool, name='on_train')


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


with tf.name_scope('BN2'):
    query_l2 = batch_normalization(query_l2, on_train, L2_N)
    doc_l2 = batch_normalization(tf.concat([doc_positive_l2, doc_negative_l2], axis=0), on_train, L2_N)
    doc_positive_l2 = tf.slice(doc_l2, [0, 0], [query_BS, -1])
    doc_negative_l2 = tf.slice(doc_l2, [query_BS, 0], [-1, -1])

    query_y = tf.nn.relu(query_l2,name='embedding_query_y')
    doc_positive_y = tf.nn.relu(doc_positive_l2,name='embedding_doc_positive_y')
    doc_negative_y = tf.nn.relu(doc_negative_l2,name='embedding_doc_negative_y')

with tf.name_scope('Merge_Negative_Doc'):
    #获取正样本
    doc_y = tf.tile(doc_positive_y, [1, 1])#这句话因为tile（mul）的结构是[1，1]，所以觉得tile没啥必要，直接赋值就行了
    label_pos = [1]*query_BS
    label_neg=[0]*query_BS*conf.NEG
    label=label_pos+label_neg
    label_tensor = tf.convert_to_tensor(label)
    print("doc_positive_y shape: ",doc_positive_y.shape,", doc_y: ",doc_y.shape)
    # 在正样本上合并负样本，tile可选择是否扩展负样本。
    for i in range(conf.NEG):
        # print ("i: ",i)
        for j in range(query_BS):
            # slice(input_, begin, size)切片API
            doc_y = tf.concat(
                [doc_y, tf.slice(
                    doc_negative_y,
                    [j * conf.NEG + i, 0],
                    [1, -1] #If `size[i]` is -1, all remaining elements in dimension i are included in the slice
                )],
                0)
        print("doc_y ", i, ": ", doc_y.shape)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    # query_norm = sqrt(sum(each x^2))
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [conf.NEG + 1, 1], name='query_norm') #包括了query的embedding
    query_norm_single = tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True), name='query_norm_single') #包括了query的embedding
    # doc_norm = sqrt(sum(each x^2))
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True), name='doc_norm')  #doc_y  shape: (500,120)，包括了正例的embedding

    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [conf.NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    # cos_sim_raw = query * doc / (||query|| * ||doc||)
    cos_sim_raw = tf.truediv(prod, norm_prod)#1、输出一下结构，2、修改结构，变成n*1，当做out
    # gamma = 20
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [conf.NEG + 1, query_BS])) * 20
    print ("cos_sim shape: ", cos_sim.shape)
    print ("cos_sim [0]: ", cos_sim[0])

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

with tf.name_scope('Auc'):
    # insert by trobr
    indices = tf.squeeze(tf.where(tf.less_equal(label_tensor, 2 - 1)), 1)
    label_tensor = tf.cast(tf.gather(label_tensor, indices), tf.int32)
    predictions = tf.gather(cos_sim_raw, indices)
    # end of insert
    auc_value, auc_op = tf.metrics.auc(label_tensor, cos_sim_raw, num_thresholds=2000)
    tf.summary.scalar('auc', auc_value)

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
           # print("train batch_id:", batch_id)
            # sess.run(train_step, feed_dict=feed_dict(True,True, batch_id))#模型训练
            cos_s_r=sess.run(cos_sim_raw,feed_dict=pull_batch(True, query_train_dat, doc_train_dat,doc_neg_train_dat, batch_id, query_BS, query_batch, doc_positive_batch,doc_negative_batch,on_train))
            # print("cos_sim_raw shape",cos_sim_raw.shape,"cos_sim_raw[0:12]: ",cos_s_r[0:10])
            # exit(0)
            sess.run(train_step, feed_dict=pull_batch(True, query_train_dat, doc_train_dat,doc_neg_train_dat, batch_id, query_BS, query_batch, doc_positive_batch,doc_negative_batch,on_train))
        end = time.time()
        # train loss下边是来计算损失，打印结果，不参与模型训练
        epoch_loss = 0
        epoch_auc = 0
        for i in range(train_epoch_steps):

            # loss_v = sess.run(loss, feed_dict=feed_dict(False, True, i))
            loss_v = sess.run(loss, feed_dict=pull_batch(False, query_train_dat, doc_train_dat,doc_neg_train_dat, i, query_BS, query_batch, doc_positive_batch, doc_negative_batch,on_train))
            epoch_loss += loss_v

            sess.run(auc_op, feed_dict=pull_batch(False, query_train_dat, doc_train_dat,doc_neg_train_dat, i, query_BS, query_batch, doc_positive_batch, doc_negative_batch,on_train))
            auc_v=sess.run(auc_value, feed_dict=pull_batch(False, query_train_dat, doc_train_dat,doc_neg_train_dat, i, query_BS, query_batch, doc_positive_batch, doc_negative_batch,on_train))
            epoch_auc += auc_v

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
            # loss_v = sess.run(loss, feed_dict=feed_dict(False, False, i))
            loss_v = sess.run(loss, feed_dict=pull_batch(False, query_vali_dat, doc_vali_dat, doc_neg_vali_dat, index, query_BS, query_batch, doc_positive_batch, doc_negative_batch,on_train))
            # print("test_loss epoch:", epoch, ", index: ", index,"loss_v: ",loss_v)
            epoch_loss += loss_v

            sess.run(auc_op, feed_dict=pull_batch(False, query_vali_dat, doc_vali_dat, doc_neg_vali_dat, index, query_BS,
                                                query_batch, doc_positive_batch, doc_negative_batch, on_train))
            auc_v = sess.run(auc_value, feed_dict=pull_batch(False, query_vali_dat, doc_vali_dat, doc_neg_vali_dat, index,
                                                        query_BS, query_batch, doc_positive_batch, doc_negative_batch,
                                                        on_train))
            epoch_auc += auc_v

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
