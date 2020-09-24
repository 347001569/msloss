import os
import sys
import numpy as np
import tensorflow as tf
import keras.backend as K

#alpha=2.0, beta=50.0, lamb=1.0, eps=0.1
def ms_loss(labels, embeddings, alpha=2.0, beta=50.0, lamb=0.5, eps=1.5, ms_mining=True):
    '''
    ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    official codes: https://github.com/MalongTech/research-ms-loss
    '''
    # make sure emebedding should be l2-normalized

    embeddings = tf.nn.l2_normalize(embeddings, axis=1)

    print(embeddings.get_shape().as_list())
    labels = tf.reshape(labels, [-1, 1])

    batch_size =12
    # if embeddings.get_shape().as_list()[0] ==None:
    #
    # else:
    #     batch_size = embeddings.get_shape().as_list()[0]
    #     print(batch_size)


    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)

    mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
    mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

    sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
    sim_mat = tf.maximum(sim_mat, 0.0)
    print(sim_mat)
    print(mask_pos)
    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    pos_exp = tf.exp(-alpha * (pos_mat - lamb))
    pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

    neg_exp = tf.exp(beta * (neg_mat - lamb))
    neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

    pos_term = tf.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
    neg_term = tf.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

    loss = tf.reduce_mean(pos_term + neg_term)

    return loss

def loss_ms(labels,feats):
    thresh = 0.5
    margin = 0.1

    scale_pos =2.0
    scale_neg =40.0

    batch_size=feats.get_shape().as_list()[0]
    sim_mat = tf.matmul(feats,  tf.transpose(feats))
    epsilon = 1e-5
    loss = list()
    for i in range(batch_size):
        pos_pair_ = sim_mat[i][labels == labels[i]]
        pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
        neg_pair_ = sim_mat[i][labels != labels[i]]

        neg_pair = neg_pair_[neg_pair_ + margin > min(pos_pair_)]
        pos_pair = pos_pair_[pos_pair_ - margin < max(neg_pair_)]

        if len(neg_pair) < 1 or len(pos_pair) < 1:
            continue

        # weighting step
        pos_loss = 1.0 / scale_pos * tf.log(
            1 + tf.reduce_sum(tf.exp(-scale_pos * (pos_pair - thresh))))
        neg_loss = 1.0 / scale_neg * tf.log(
            1 + tf.reduce_sum(tf.exp(scale_neg * (neg_pair - thresh))))
        loss.append(pos_loss + neg_loss)
    if len(loss) == 0:
        return tf.zeros([], requires_grad=True)

    loss = sum(loss) / batch_size


    return loss