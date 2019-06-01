#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import time
from aggregator_new import *
import networkx as nx
from sklearn.linear_model import LogisticRegression
import os


class Graphsage_unsup:
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

        self.ui_nodes = tf.placeholder(shape=(None),dtype=tf.int32)
        self.ui_s1_neighs = tf.placeholder(shape=(None,neigh_size),dtype=tf.int32)
        if self.cfg.depth == 2:
            self.ui_s2_neighs = tf.placeholder(shape=(None,neigh_size,neigh_size),dtype=tf.int32)

        self.uj_nodes = tf.placeholder(shape=(None), dtype=tf.int32)
        self.uj_s1_neighs = tf.placeholder(shape=(None, neigh_size), dtype=tf.int32)
        if self.cfg.depth == 2:
            self.uj_s2_neighs = tf.placeholder(shape=(None, neigh_size, neigh_size), dtype=tf.int32)

        self.unsup_label = tf.placeholder(shape=(None), dtype=tf.float32)

        if self.cfg.depth == 1:
            ui_fea = tf.nn.embedding_lookup(self.features, self.ui_nodes)
            ui_1_fea = tf.nn.embedding_lookup(self.features, self.ui_s1_neighs)
            self.ui_result = self.aggregator(ui_fea, ui_1_fea, self.cfg.dims, 'agg_1st')
            uj_fea = tf.nn.embedding_lookup(self.features, self.uj_nodes)
            uj_1_fea = tf.nn.embedding_lookup(self.features, self.uj_s1_neighs)
            self.uj_result = self.aggregator(uj_fea, uj_1_fea, self.cfg.dims, 'agg_1st')
        else:
            ui_fea = tf.nn.embedding_lookup(self.features, self.ui_nodes)
            ui_1_fea = tf.nn.embedding_lookup(self.features, self.ui_s1_neighs)
            ui_agg_node = self.aggregator(ui_fea, ui_1_fea, self.cfg.dims, 'agg_1st')
            ui_2_fea = tf.nn.embedding_lookup(self.features, self.ui_s2_neighs)
            ui_agg_negh1 = self.aggregator(ui_1_fea, ui_2_fea, self.cfg.dims, 'agg_1st')
            self.ui_result = self.aggregator(ui_agg_node, ui_agg_negh1, self.cfg.dims, 'agg_2nd')
            uj_fea = tf.nn.embedding_lookup(self.features, self.uj_nodes)
            uj_1_fea = tf.nn.embedding_lookup(self.features, self.uj_s1_neighs)
            uj_agg_node = self.aggregator(uj_fea, uj_1_fea, self.cfg.dims, 'agg_1st')
            uj_2_fea = tf.nn.embedding_lookup(self.features, self.uj_s2_neighs)
            uj_agg_negh1 = self.aggregator(uj_1_fea, uj_2_fea, self.cfg.dims, 'agg_1st')
            self.uj_result = self.aggregator(uj_agg_node, uj_agg_negh1, self.cfg.dims, 'agg_2nd')

            self.ui_result = tf.nn.l2_normalize(self.ui_result, 1)
            self.uj_result = tf.nn.l2_normalize(self.uj_result, 1)

        #unsup
        self.inner_product = tf.reduce_mean(self.ui_result * self.uj_result, axis=1)
        self.loss_unsup = -tf.reduce_mean(tf.log_sigmoid(self.unsup_label * self.inner_product))
        self.unsup_op = tf.train.AdamOptimizer(self.cfg.lr).minimize(self.loss_unsup)
