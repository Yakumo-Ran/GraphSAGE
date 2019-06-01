#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import time
from generator import Generator
from model_sup import Graphsage_sup
import networkx as nx
from config_new import cfg
import os

def train():
    gen = Generator()
    model = Graphsage_sup()
    tt = time.ctime().replace(' ', '-')
    tt = tt.replace(':', '-')
    path = 'grasage_sup' + '-' + tt
    fout = open(path + "-log.txt", "w")
    rand_indices = np.random.permutation(cfg.num_nodes)
    test = list(rand_indices[:1000])
    val = list(rand_indices[1000:1200])
    train = list(rand_indices[1200:])
    val_s1, val_s2 = gen.fetch_batch(val)
    val_label = cfg.labels[val]
    feed_dict_val={model.batch_nodes: val, model.s1_neighs: val_s1, model.s2_neighs: val_s2, model.labels: val_label}
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(cfg.epochs):
            t = 0
            for b in range(0, len(train), cfg.batchsize):
                s = time.time()
                start = b
                end = min(b+cfg.batchsize, len(train))
                batchnodes = train[start: end]
                s1,s2 = gen.fetch_batch(batchnodes)
                label = cfg.labels[batchnodes]
                feed_dict_train = {model.batch_nodes: batchnodes, model.s1_neighs: s1, model.s2_neighs: s2,
                                   model.labels: label}

                _, ls_train, acc_train = sess.run([model.opt, model.loss, model.accuray], feed_dict=feed_dict_train)
                ls_val, acc_val = sess.run([model.loss, model.accuray], feed_dict=feed_dict_val)
                e = time.time()
                t = e - s
                print(' Epoch = {:d} TrainLoss = {:.5f} TrainAccuracy = {:.3f} ValLoss = {:.5f} ValAccuracy = {:.3f} Time = {:.3f}\n'.format(
                        i + 1, ls_train, acc_train, ls_val, acc_val, t))
        start = 0
        loss_list = []
        accu_list = []
        while start < len(test):
            end = min(start + cfg.batchsize, len(test))
            batchnodes = test[start:end]
            s1,s2 = gen.fetch_batch(batchnodes)
            test_label = cfg.labels[batchnodes]
            feed_dict_test = {model.batch_nodes: batchnodes, model.s1_neighs: s1, model.s2_neighs: s2,
                                   model.labels: test_label}
            ls_test, acc_test = sess.run([model.loss, model.accuray], feed_dict=feed_dict_test)
            loss_list.append(ls_test * (end - start))
            accu_list.append(acc_test * (end - start))
            start = end
        print('TestLoss = ', sum(loss_list) / len(test), ' TestAccuracy = ', sum(accu_list) / len(test))

if __name__ == '__main__':
    train()