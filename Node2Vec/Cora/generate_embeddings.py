import argparse

import torch
# from torch_geometric.nn import Node2Vec
from pyg_node2vec import Node2Vec

from torch_geometric.utils import to_undirected

from ogb.linkproppred import PygLinkPropPredDataset

import os
import numpy as np

def save_embedding(model, filename):
    torch.save(model.embedding.weight.data.cpu(), filename)
    
def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (Node2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default = 0)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    
    model = Node2Vec(train_edges_undirected, args.embedding_dim, args.walk_length,
                         args.context_size, args.walks_per_node,
                         sparse=True, seed=args.seed).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True,
                          num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    model.train()
    save_dir = "embeddings"
    save_dir_dim = os.path.join(save_dir, str(args.embedding_dim))
    os.makedirs(save_dir_dim, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')
        
    save_dir_dim_seed = os.path.join(save_dir_dim, str(args.seed) + ".pt")
    save_embedding(model, save_dir_dim_seed)
if __name__ == "__main__":
    main()