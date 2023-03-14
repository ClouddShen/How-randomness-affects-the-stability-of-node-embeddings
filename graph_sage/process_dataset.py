
import random
import torch
from torch_geometric.utils import to_dense_adj, add_self_loops, to_undirected
import numpy as np
import networkx as nx

def create_train_test_split(src, dst, split_ratio=0.1):
    G = nx.Graph()
    edges = zip(src, dst)
    G.add_edges_from(edges)
    largest_cc = max(nx.connected_components(G), key=len)
    G_max = nx.Graph(G.subgraph(largest_cc)) # get largest connected components

    num_edges = G_max.number_of_edges()
    num_edges_to_remove = int(num_edges * split_ratio)
    edge_list = list(G_max.edges())
    random.seed(0)
    random.shuffle(edge_list)

    # remove 10% edges from Gmax yet make it still connected
    removed_edges = []
    for i in range(num_edges_to_remove):
        G_max.remove_edge(*edge_list[i])

        if not nx.is_connected(G):
            removed_edges.append(edge_list[i])
            continue

    src_train = [e[0] for e in G_max.edges()]
    dst_train = [e[1] for e in G_max.edges()]
    src_test = [e[0] for e in removed_edges]
    dst_test = [e[1] for e in removed_edges]

    train_edges = torch.tensor([src_train, dst_train]) 
    test_edges = torch.tensor([src_test, dst_test]) 

    train_edges_undirected = to_undirected(train_edges, G_max.number_of_nodes())
    test_edges_undirected = to_undirected(test_edges, G_max.number_of_nodes())

    return train_edges_undirected, test_edges_undirected



def generate_test_neg(train_edges_undirected, test_edges_undirected):
    train_pos_edges = train_edges_undirected
    test_pos_edges = test_edges_undirected
    edges_index = torch.cat((train_edges_undirected, test_edges_undirected), dim=1)
    edges_index_self_loops = add_self_loops(edges_index)[0]
    adj_matrix = to_dense_adj(edges_index_self_loops).squeeze(dim=0)
    zero_indices = torch.where(adj_matrix == 0)
    num = 506
    neg_indices = torch.randperm(zero_indices[0].shape[0])[0:num]
    test_neg_sourse = zero_indices[0][neg_indices].unsqueeze(0)
    test_neg_target = zero_indices[1][neg_indices].unsqueeze(0)
    test_neg_edges = to_undirected(torch.cat((test_neg_sourse, test_neg_target), dim=0))

    print(train_pos_edges.shape)
    print(test_pos_edges.shape)
    print(test_neg_edges.shape)
    np.save('./cora/data/train_pos_edges.npy', train_pos_edges.numpy())
    np.save('./cora/data/test_pos_edges.npy', test_pos_edges.numpy())
    np.save('./cora/data/test_neg_edges.npy', test_neg_edges.numpy())
    return train_pos_edges, test_pos_edges, test_neg_edges
