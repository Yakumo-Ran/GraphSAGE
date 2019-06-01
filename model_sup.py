#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import time
from aggregator_new import *
import networkx as nx
from sklearn.linear_model import LogisticRegression
import os


class Graphsage_sup:
    def __init__(self):
        self.cfg = cfg
        self.features = tf.Variable(self.cfg.features,dtype=tf.float32,trainable=False)

        if self.cfg.aggregator == 'mean':
            self.aggregator = mean_aggregator
        elif self.cfg.aggregator == 'pooling':
            self.aggregator = pooling_aggreagtor
        elif self.cfg.aggregator == 'lstm':
            self.aggregator = lstm_aggregator
        else:
            raise(Exception,"Invalid aggregator!")

        if self.cfg.gcn:
            neigh_size = self.cfg.sample_num + 1
        else:
            neigh_size = self.cfg.sample_num
        self.batch_nodes = tf.placeholder(shape=(None),dtype=tf.int32)
        self.s1_neighs = tf.placeholder(shape=(None,neigh_size),dtype=tf.int32)
        if self.cfg.depth == 2:
            self.s2_neighs = tf.placeholder(shape=(None,neigh_size,neigh_size),dtype=tf.int32)
        self.labels = tf.placeholder(shape=(None), dtype=tf.int32)





        if self.cfg.depth == 1:
            node_fea = tf.nn.embedding_lookup(self.features, self.batch_nodes)
            neigh_1_fea = tf.nn.embedding_lookup(self.features, self.s1_neighs)
            self.agg_result = self.aggregator(node_fea, neigh_1_fea, self.cfg.dims, 'agg_1st')
        else:
            node_fea = tf.nn.embedding_lookup(self.features, self.batch_nodes)
            neigh_1_fea = tf.nn.embedding_lookup(self.features, self.s1_neighs)
            agg_node = self.aggregator(node_fea, neigh_1_fea, self.cfg.dims, 'agg_1st')
            neigh_2_fea = tf.nn.embedding_lookup(self.features, self.s2_neighs)
            agg_negh1 = self.aggregator(neigh_1_fea, neigh_2_fea, self.cfg.dims, 'agg_1st')
            self.agg_result = self.aggregator(agg_node, agg_negh1, self.cfg.dims, 'agg_2nd')

        self.pred = tf.layers.dense(self.agg_result, units=self.cfg.num_classes, activation=None)
        self.label = tf.one_hot(self.labels, depth=self.cfg.num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.pred)
        self.accuray = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred, -1), tf.argmax(self.label, -1)), tf.float32))
        self.opt = tf.train.AdamOptimizer(self.cfg.lr).minimize(self.loss)






