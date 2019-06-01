#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import time
from generator import Generator
from model_unsup import Graphsage_unsup
import networkx as nx
import random
from config_new import cfg
from sklearn.linear_model import LogisticRegression
import os

def train():
    gen = Generator()
    model = Graphsage_unsup()
    num_batchs = 10
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(num_batchs):
                s = time.time()
                u_i,u_j,unsup_label = gen.deepwalk_walk()
                u_s1,u_s2 = gen.fetch_batch(u_i)

                j_s1,j_s2 = gen.fetch_batch(u_j)

                feed_dict_unsup = {model.ui_nodes: u_i, model.uj_nodes: u_j, model.ui_s1_neighs: u_s1,
                                   model.ui_s2_neighs: u_s2, model.uj_s1_neighs: j_s1, model.uj_s2_neighs: j_s2,
                                   model.unsup_label: unsup_label}
                _, ls_train = sess.run([model.unsup_op, model.loss_unsup], feed_dict=feed_dict_unsup)
                e = time.time()
                t = e - s
                print(' num of batchs = {:d} TrainLoss = {:.5f} Time = {:.3f}\n'.format(
                        i + 1, ls_train, t))

        #test
        start = 0
        embedding = np.zeros((cfg.num_nodes, cfg.dims))
        while (start < cfg.num_nodes):
            end = min(start + cfg.batchsize, cfg.num_nodes)
            unique_nodes = list(range(start, end))
            samp_neighs_1st,samp_neighs_2nd = gen.fetch_batch(unique_nodes)
            x = sess.run(model.ui_result, feed_dict={
                model.ui_nodes: unique_nodes,
                model.ui_s1_neighs: samp_neighs_1st,
                model.ui_s2_neighs: samp_neighs_2nd
            })
            embedding[unique_nodes] = x
            start = end
        print(embedding.shape)
        X, Y = [i for i in range(cfg.num_nodes)], [int(cfg.labels[i]) for i in range(cfg.num_nodes)]
        state = random.getstate()
        random.shuffle(X)
        random.setstate(state)
        random.shuffle(Y)
        index = int(cfg.num_nodes * cfg.clf_ratio)
        X_train = embedding[X[0:index]]
        Y_train = Y[0:index]
        X_test = embedding[X[index:]]
        Y_test = Y[index:]
        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial').fit(X_train, Y_train)
        print('TestAccuracy = ', clf.score(X_test, Y_test))


if __name__ == '__main__':
    train()