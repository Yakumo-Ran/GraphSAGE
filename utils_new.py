from collections import defaultdict
import numpy as np
import networkx as nx
def load_data():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes), dtype=np.int64)
    node_map = {}
    label_map = {}
    id_map = {}
    with open('./cora/cora.content') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            # feat_data[i,:] = map(float, info[1:-1])
            feat_data[i, :] = [float(x) for x in info[1:-1]]
            node_map[info[0]] = i
            id_map[i] = info[0]
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open('./cora/cora.cites') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists, node_map, id_map

if __name__ == '__main__':
    feat_data, labels, adj_lists, node_map, id_map = load_data()
    print(node_map['20584'])