#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import random
from config_new import cfg
import networkx as nx
import os

class Generator:
    def __init__(self):
      self.cfg = cfg
      self.s_neighs = []
      self.s1_neighs = []
      self.s2_neighs = []
      self.u_i = []
      self.u_j = []

    def deepwalk_walk(self, walk_length=6, num_walks=20):
        G = nx.read_edgelist('./cora/cora.cites', create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
        nodes = G.nodes()
        degrees = [G.degree(x) for x in list(nodes)]
        p = np.array(degrees) / sum(degrees)
        self.u_i = []
        self.u_j = []
        self.label = []
        for i in range(num_walks):
            neg = 0
            start_node_id = np.random.choice(self.cfg.num_nodes)
            start_node = self.cfg.id_map[start_node_id]
            walk = [start_node]
            walk_index = [start_node_id]
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = list(G.neighbors(cur))
                if len(cur_nbrs) > 0:
                    walk_node = np.random.choice(cur_nbrs)
                    walk.append(walk_node)
                    walk_index.append(self.cfg.node_map[walk_node])
                    self.u_i.append(start_node_id)
                    self.u_j.append(self.cfg.node_map[walk_node])
                    self.label.append(1.)
                else:
                    break
            while neg < self.cfg.neg_num:
                x = np.random.choice(nodes, p=p)
                a = self.cfg.node_map[x]
                if a not in walk_index:
                    self.u_i.append(start_node_id)
                    self.u_j.append(a)
                    self.label.append(-1.)
                    neg = neg + 1

        return self.u_i, self.u_j, self.label

    def sample(self,nodes):
        self.s_neighs = []
        for node in nodes:
            neighs = list(cfg.adj_lists[int(node)])
            if self.cfg.sample_num > len(neighs):
                nei = list(np.random.choice(neighs,cfg.sample_num,replace=True))
            else:
                nei = list(np.random.choice(neighs, cfg.sample_num, replace=False))
            if self.cfg.gcn:
                nei.append(node)
            self.s_neighs.append(nei)
        return self.s_neighs

    def fetch_batch(self,nodes):
        self.s1_neighs = []
        self.s2_neighs = []
        self.s1_neighs = self.sample(nodes)
        for neigh in self.s1_neighs:
            self.s2_neighs.append(self.sample(neigh))

        return self.s1_neighs,self.s2_neighs
if __name__ == '__main__':
    gen = Generator()
    u_i,u_j,label = gen.deepwalk_walk()
    print(u_i)
    print('\n---------------\n')
    print(u_j)
    print('\n---------------\n')
    print(label)