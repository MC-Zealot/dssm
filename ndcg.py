import math
import tensorflow as tf
import numpy as np
import metrics


'''
ref: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
'''


def calculate_dcg(items, k):
    items = items[0:k]
    dcg = 0
    i = 0
    for item in items:
        i += 1
        dcg += item / math.log(i + 1, 2)
    return dcg


def calc_ndcg(relevances, ideal=None, k=5):
    dcg = calculate_dcg(relevances, k)
    print("dcg:", dcg)
    if ideal is None:
        ideal = list(relevances)
    ideal.sort(reverse=True)
    optimal_dcg = calculate_dcg(ideal, k)
    print("optimal_dcg: ", optimal_dcg)
    ndcg = dcg / optimal_dcg
    return ndcg


rel_scores = [3, 2, 3, 0, 1, 2, 3, 2]
rel_scores = [[3, 0], [2, 0], [3, 0], [0, 0], [1, 0], [2, 0], [3, 0], [2, 0]]
ideal = None
# class_res = calc_ndcg(rel_scores, ideal,6)
# print (class_res)

# from scipy import sparse
# tmp = sparse.csr_matrix(rel_scores)
# coo = tmp.tocoo()
# print("coo type: ", type(coo))
# indices = np.mat([coo.row, coo.col]).transpose()
# rel_scores_tensor = tf.SparseTensorValue(indices, coo.data, coo.shape)
#
# rel_scores=[[3,0],[2,0],[3,0],[0,0],[1,0],[2,0],[3,0],[2,0]]
tensor_rel_scores = tf.convert_to_tensor(rel_scores)
tensor_rel_scores_float = tf.cast(tensor_rel_scores, tf.float32)

ndcg_out=metrics.normalized_discounted_cumulative_gain(tensor_rel_scores_float, tensor_rel_scores_float)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # a = sess.run(ndcg_out)
    # sess.run(tf.local_variables_initializer())
    # print ("rel_scores_tensor: ", sess.run(rel_scores_tensor))

    # print('...', tensor_rel_scores)

    # print('...', tensor_rel_scores)
    # tensor_rel_scores = tf.expand_dims(tensor_rel_scores, axis=1)
    # print('...', tensor_rel_scores)
    print("aaaaaa: ", np.shape(ndcg_out))
    print("aaaaaa: ", np.shape(ndcg_out[0]))
    print("aaaaaa: ", np.shape(ndcg_out[1]))
