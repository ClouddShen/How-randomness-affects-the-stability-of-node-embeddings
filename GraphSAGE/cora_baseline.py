
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch.utils.data import DataLoader
from torch_geometric.nn import GraphSAGE
from predictor import MLPPredictor
from fixed_seed import fix_seed

def train(model, predictor, node_features, edge_index, optimizer, batch_size):
    model.train()
    predictor.train()

    num_nodes = node_features.shape[0]
    source_edge = edge_index[0].to(node_features.device)
    target_edge = edge_index[1].to(node_features.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(source_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        h = model(node_features, edge_index)

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, node_features, pos_edge_index, neg_edge_index, batch_size):
    predictor.eval()

    num_correct = 0

    size = pos_edge_index.shape[1]
    pos_source_index = pos_edge_index[0].to(node_features.device)
    pos_target_index = pos_edge_index[1].to(node_features.device)
    neg_source_index = neg_edge_index[0].to(node_features.device)
    neg_target_index = neg_edge_index[1].to(node_features.device)

    for perm in DataLoader(range(size), batch_size, shuffle=True):

        h = model(node_features, pos_edge_index)
        # print("h", h.shape)

        pos_src, pos_dst = pos_source_index[perm], pos_target_index[perm]
        pos_pred = predictor(h[pos_src], h[pos_dst])
        correct_pos_index = torch.where(pos_pred > 0.5)
        correct_pos_num = correct_pos_index[0].shape[0]

        neg_src, neg_dst = neg_source_index[perm], neg_target_index[perm]
        neg_pred = predictor(h[neg_src], h[neg_dst])
        correct_neg_index = torch.where(neg_pred < 0.5)
        correct_neg_num = correct_neg_index[0].shape[0]

        num_correct += correct_pos_num + correct_neg_num

    
    acc = num_correct / (size * 2)
    return acc


def save_embedding_data(node_embbeding, embed_dim, seed):
    file_path = f'./cora/embedding_data/{str(embed_dim)}/{str(seed)}'
    print("save node embeddings in ", file_path)
    np.save(file_path, node_embbeding.detach().numpy())
    return

def main(seed, embed_dim):
    cora_pyg = Planetoid(root='/tmp/Cora', name='Cora', split="full")
    cora_data = cora_pyg[0]

    print(cora_data)

    device = 0
    num_features = cora_data.x.shape[1]
    hidden_channels = 256
    num_layers = 2
    dropout = 0
    lr = 0.0005
    epochs = 10
    batch_size = 256

    # embed_dim = 8
    seed = seed

    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    cora_data = cora_data.to(device)
    train_pos_edges = torch.from_numpy(np.load('./cora/data/train_pos_edges.npy')).to(device)
    test_pos_edges = torch.from_numpy(np.load('./cora/data/test_pos_edges.npy')).to(device)
    test_neg_edges = torch.from_numpy(np.load('./cora/data/test_neg_edges.npy')).to(device)
    print(train_pos_edges.shape)
    print(test_pos_edges.shape)
    print(test_neg_edges.shape)

    fix_seed(seed)

    model = GraphSAGE(num_features, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=embed_dim).to(device)
    print(model)
    predictor = MLPPredictor(embed_dim).to(device)
    print(predictor)

    model.reset_parameters()
    predictor.reset_parameters()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr)

    loss_list = []
    acc_list = []

    for epoch in range(1, 1 + epochs):
        loss = train(model, predictor, cora_data.x, train_pos_edges, optimizer, batch_size)
        loss_list.append(loss)
        print(epoch, "loss:", loss)
        acc = test(model, predictor, cora_data.x, test_pos_edges, test_neg_edges, batch_size)
        acc_list.append(acc)
        print(epoch, "acc:", acc)

    print(loss_list)
    print(acc_list)
    model_path = f'./cora/model/{embed_dim}/GraphSAGE_embed_{embed_dim}_seed_{seed}.pt'
    torch.save(model.state_dict(), model_path)
    print("Saving GraphSage in: ", model_path)

    predictor_path = f'./cora/model/{embed_dim}/MLP_embed_{embed_dim}_seed_{seed}.pt'
    torch.save(predictor.state_dict(), predictor_path)
    print("Saving MLP in: ", predictor_path)
    
    embeded_nodes = model(cora_data.x, train_pos_edges)
    print(embeded_nodes.shape)
    save_embedding_data(embeded_nodes, embed_dim, seed)


if __name__ == '__main__':
    embed_dim_list = [8, 16, 32, 64, 128, 256]
    for embed_dim in embed_dim_list:
        for i in range(10):
            main(i, embed_dim)