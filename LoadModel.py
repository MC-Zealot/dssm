import utils
from utils import *

# 读取数据
conf = Config()
sess = tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('./model/model_1.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model'))

vectorizer = utils.load_vectorizer()#字典
TRIGRAM_D = len(vectorizer.get_feature_names()) # 词库大小，aka 稀疏矩阵列数
query_test, doc_test, doc_neg_test = utils.GetActDat(conf.file_vali)
query_test = query_test[:conf.query_BS]
doc_test = doc_test[:conf.query_BS]
doc_neg_test = doc_neg_test[:conf.query_BS * conf.NEG]

print("bhv_act_test len: ", len(query_test))
print("ad_act_test len: ", len(doc_test))
print("ad_act_neg_test len: ", len(doc_neg_test))
# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
# print ("node: ",[n.name for n in graph.as_graph_def().node])
# for n in graph.as_graph_def().node:
#     print ("node: ",n.name)

query_y = graph.get_tensor_by_name("BN2/embedding_query_y:0")
doc_positive_y = graph.get_tensor_by_name("BN2/embedding_doc_positive_y:0")
doc_negative_y = graph.get_tensor_by_name("BN2/embedding_doc_negative_y:0")
query_norm_single = graph.get_tensor_by_name("Cosine_Similarity/query_norm_single:0")
on_train                   = graph.get_tensor_by_name("input/on_train:0")
# query_batch = graph.get_tensor_by_name("input/query_batch:0")
query_batch_indices        = graph.get_tensor_by_name("input/query_batch/indices:0")
query_batch_values         = graph.get_tensor_by_name("input/query_batch/values:0")
query_batch_shape          = graph.get_tensor_by_name("input/query_batch/shape:0")

doc_positive_batch_indices = graph.get_tensor_by_name("input/doc_positive_batch/indices:0")
doc_positive_batch_values  = graph.get_tensor_by_name("input/doc_positive_batch/values:0")
doc_positive_batch_shape   = graph.get_tensor_by_name("input/doc_positive_batch/shape:0")

doc_negative_batch_indices = graph.get_tensor_by_name("input/doc_negative_batch/indices:0")
doc_negative_batch_values  = graph.get_tensor_by_name("input/doc_negative_batch/values:0")
doc_negative_batch_shape   = graph.get_tensor_by_name("input/doc_negative_batch/shape:0")


Bhv_in = convert_sparse_matrix_to_sparse_tensor(vectorizer.transform(query_test))
ad_in = convert_sparse_matrix_to_sparse_tensor(vectorizer.transform(doc_test))
ad_neg_in = convert_sparse_matrix_to_sparse_tensor(vectorizer.transform(doc_neg_test))
# ad_neg_in = convert_sparse_matrix_to_sparse_tensor(vectorizer.transform(ad_act_neg_test))

print("Bhv_in len: ",len(Bhv_in))
print("ad_in len: ",len(Bhv_in))
print("ad_neg_in len: ",len(ad_neg_in))
# print("Bhv_in[0]: ",Bhv_in[0])
# print("Bhv_in[1]: ",Bhv_in[1])

y = sess.run(query_y, feed_dict={
    query_batch_indices: Bhv_in[0],
    query_batch_values: Bhv_in[1],
    query_batch_shape: Bhv_in[2],

    doc_positive_batch_indices: ad_in[0],
    doc_positive_batch_values: ad_in[1],
    doc_positive_batch_shape: ad_in[2],

    doc_negative_batch_indices: ad_neg_in[0],
    doc_negative_batch_values: ad_neg_in[1],
    doc_negative_batch_shape: ad_neg_in[2],
    on_train: False})
# doc_pos_y = sess.run(doc_positive_y, feed_dict={doc_positive_batch_indices: ad_in[0] ,doc_positive_batch_values: ad_in[1], doc_positive_batch_shape: ad_in[2],  on_train: False})
doc_pos_y = sess.run(doc_positive_y, feed_dict={
    query_batch_indices: Bhv_in[0],
    query_batch_values: Bhv_in[1],
    query_batch_shape: Bhv_in[2],

    doc_positive_batch_indices: Bhv_in[0],
    doc_positive_batch_values: Bhv_in[1],
    doc_positive_batch_shape: Bhv_in[2],

    doc_negative_batch_indices: ad_neg_in[0],
    doc_negative_batch_values: ad_neg_in[1],
    doc_negative_batch_shape: ad_neg_in[2],
    on_train: False})
doc_neg_y = sess.run(doc_negative_y, feed_dict={
    query_batch_indices: Bhv_in[0],
    query_batch_values: Bhv_in[1],
    query_batch_shape: Bhv_in[2],
    doc_positive_batch_indices: ad_in[0],
    doc_positive_batch_values: ad_in[1],
    doc_positive_batch_shape: ad_in[2],

    doc_negative_batch_indices: ad_neg_in[0],
    doc_negative_batch_values: ad_neg_in[1],
    doc_negative_batch_shape: ad_neg_in[2],
    on_train: False})
query_norm_single = sess.run(query_norm_single, feed_dict={query_batch_indices: Bhv_in[0],query_batch_values: Bhv_in[1],query_batch_shape: Bhv_in[2], on_train: False})
# query_norm_single = sess.run(query_norm_single, feed_dict={query_batch_indices: Bhv_in[0],query_batch_values: Bhv_in[1],query_batch_shape: Bhv_in[2], on_train: False})
print("y:", y[0][:10], ", len: ", len(y), ", d: ", len(y[0]))
print("doc_pos_y:", doc_pos_y[0][:10], ", len: ", len(doc_pos_y), ", d: ", len(doc_pos_y[0]))
print("doc_neg_y:", doc_neg_y[0][:10], ", len: ", len(doc_neg_y), ", d: ", len(doc_neg_y[0]))

print("query_norm_single:", query_norm_single[0], ", len: ", len(query_norm_single), ", d: ", len(query_norm_single[0]))
# exit(0)

y_mid_vector_file = conf.y_mid_vector_file
with open(y_mid_vector_file, 'a+') as f:

    for i in range(len(y)):
        s = []
        index = 0
        for j in y[i].tolist():
            j_s = str(j)
            if j_s != "0.0" and j > 0.0001:
                s.append(str(index) + ":" + j_s[0:6])
            index += 1
        # print()
        line= query_test[i].replace(" ","") + "\t" + str(query_norm_single[i][0]) + "\t" + ",".join(s)
        f.write(line + '\n')  # 加\n换行显示

doc_pos_y_mid_vector_file = conf.doc_pos_y_mid_vector_file
with open(doc_pos_y_mid_vector_file, 'a+') as f:

    for i in range(len(doc_pos_y)):
        s = []
        index = 0
        for j in doc_pos_y[i].tolist():
            j_s = str(j)
            if j_s != "0.0" and j > 0.0001:
                s.append(str(index) + ":" + j_s[0:6])
            index += 1
        # print()
        line = doc_test[i].replace(" ","") + "\t" + ",".join(s)
        f.write(line + '\n')  # 加\n换行显示

doc_neg_y_mid_vector_file = conf.doc_neg_y_mid_vector_file
with open(doc_neg_y_mid_vector_file, 'a+') as f:

    for i in range(len(doc_neg_y)):
        s = []
        index = 0
        for j in doc_neg_y[i].tolist():
            j_s = str(j)
            if j_s != "0.0" and j > 0.0001:
                s.append(str(index) + ":" + j_s[0:6])
            index += 1
        # print()
        line = doc_neg_test[i].replace(" ","") + "\t" + ",".join(s)
        f.write(line + '\n')  # 加\n换行显示