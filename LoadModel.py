import utils
from utils import *

# 读取数据
conf = Config()
sess=tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('./model/model_1.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model'))


# #得到该网络中，所有可以加载的参数
# variables = tf.contrib.framework.get_variables_to_restore()
#
# #删除output层中的参数
# variables_to_resotre = [v for v in variables if v.name.split('/')[0]!='Merge_Negative_Doc'
#                                             or v.name.split('/')[0]!='Cosine_Similarity'
#                                             or v.name.split('/')[0]!='Loss'
#                                             or v.name.split('/')[0] != 'Accuracy'
#                                             or v.name.split('/')[0] != 'Training'
#                                             or v.name.split('/')[0] != 'Test'
#                                             or v.name.split('/')[0] != 'Train'
#                         ]
# #构建这部分参数的saver
# saver = tf.train.Saver(variables_to_resotre)
# saver.restore(sess,'./model/model_1.ckpt.meta')




vectorizer = utils.load_vectorizer()#字典
TRIGRAM_D = len(vectorizer.get_feature_names()) # 词库大小，aka 稀疏矩阵列数
bhv_act_test, ad_act_test, ad_act_neg_test  = utils.GetActDat_v2(conf.file_vali)
print("bhv_act_test len: ",len(bhv_act_test))
# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
# print ("node: ",[n.name for n in graph.as_graph_def().node])
# for n in graph.as_graph_def().node:
#     print ("node: ",n.name)

query_y = graph.get_tensor_by_name("BN2/embedding_query_y:0")
on_train = graph.get_tensor_by_name("input/on_train:0")
# query_batch = graph.get_tensor_by_name("input/query_batch:0")
query_batch_indices = graph.get_tensor_by_name("input/query_batch/indices:0")
query_batch_values = graph.get_tensor_by_name("input/query_batch/values:0")
query_batch_shape = graph.get_tensor_by_name("input/query_batch/shape:0")
# query_batch = tf.sparse_placeholder(tf.float32, name='query_batch')
# query_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='query_batch')


Bhv_in = convert_sparse_matrix_to_sparse_tensor(vectorizer.transform(bhv_act_test))

print("Bhv_in len: ",len(Bhv_in))
# print("Bhv_in[0]: ",Bhv_in[0])
# print("Bhv_in[1]: ",Bhv_in[1])

y = sess.run(query_y, feed_dict={query_batch_indices: Bhv_in[0],query_batch_values: Bhv_in[1],query_batch_shape: Bhv_in[2], on_train: False})
print("y:" ,y[0],", len: ", len(y),", d: ",len(y[0]))