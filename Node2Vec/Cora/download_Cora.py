from torch_geometric.datasets import Planetoid
import os.path as osp

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), dataset)
dataset = Planetoid(path, dataset)