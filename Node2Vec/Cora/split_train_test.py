from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import numpy as np

cora_pyg = Planetoid(root='Cora', name='Cora', split="full")
cora_data = cora_pyg[0]

# load source nodes and dst nodes for edges of Cora
src = cora_data.edge_index[0].numpy()
dst = cora_data.edge_index[1].numpy()

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
  cnt = 0
  index = 0
  while cnt < num_edges_to_remove:
    G_tmp = G_max.copy()
    G_tmp.remove_edge(*edge_list[index])
    if nx.is_connected(G_tmp):
      G_max.remove_edge(*edge_list[index])
      removed_edges.append(edge_list[index])
      cnt+=1
    index += 1

  src_train = [e[0] for e in G_max.edges()]
  dst_train = [e[1] for e in G_max.edges()]
  src_test = [e[0] for e in removed_edges]
  dst_test = [e[1] for e in removed_edges]

  train_edges = torch.tensor([src_train, dst_train]) 
  test_edges = torch.tensor([src_test, dst_test]) 

  train_edges_undirected = to_undirected(train_edges, G_max.number_of_nodes())
  test_edges_undirected = to_undirected(test_edges, G_max.number_of_nodes())

  return train_edges_undirected, test_edges_undirected

train_edges_undirected, test_edges_undirected = create_train_test_split(src, dst)
np.save("train_split.npy", train_edges_undirected)
np.save("test_split.npy", test_edges_undirected)