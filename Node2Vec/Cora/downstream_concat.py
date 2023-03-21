from mlp import MLPPredictor
import torch
import numpy as np
from torch.utils.data import DataLoader
import os.path as osp
import os
from tqdm import tqdm
from utils import predict_downstream_result, calculate_two_model_downstream_result_similarity, calculate_one_dim_downstream_result_similarity
from utils import calcaulate_overlap_for_all_seeds, calculate_acc_mean_std_for_one_dim, calculate_one_dim_downstream_result_similarity_all_true
import pandas as pd

device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


def train(model, x, split_edge, optimizer, batch_size):
    model.train()
    src_node = split_edge[0].to(x.device)
    dst_node = split_edge[1].to(x.device)
    total_loss = total_examples = 0
    for perm in DataLoader(range(src_node.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        src, dst = src_node[perm], dst_node[perm]
        pos_out = model(x[src], x[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        dst_neg = torch.randint(0, x.size(0), src.size(), dtype=torch.long,
                                device=x.device)
        neg_out = model(x[src], x[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def test(model, x, pos_edge, neg_edge, batch_size):
    src_pos_node = pos_edge[0].to(x.device)
    dst_pos_node = pos_edge[1].to(x.device)
    src_neg_node = neg_edge[0].to(x.device)
    dst_neg_node = neg_edge[1].to(x.device)
    
    correct = 0
    
    for perm in DataLoader(range(src_pos_node.size(0)), batch_size):
        src_pos, dst_pos = src_pos_node[perm], dst_pos_node[perm]
        correct += sum(model(x[src_pos], x[dst_pos]).squeeze().cpu() >= 0.5)
        src_neg, dst_neg = src_neg_node[perm], dst_neg_node[perm]
        correct += sum(model(x[src_neg], x[dst_neg]).squeeze().cpu() < 0.5)
    return correct/(pos_edge.size(1) * 2)
    

def run_and_save_model(src_dir="", dst_dir="", seed_id=0, dim=8):
    train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    embedding_path = os.path.join(src_dir, str(dim), str(seed_id) + ".pt")
    x_emb = torch.load(embedding_path, map_location='cpu')
    x_origin = torch.from_numpy(np.load("x_origin.npy"))
    x = torch.cat((x_emb, x_origin), dim=-1)
    x = x.to(device)
    model = MLPPredictor(x.size(-1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(1, 101):
        loss = train(model, x, train_edges_undirected, optimizer, 100)
        test(model, x, test_edges_undirected, test_neg_edges_undirected, 100)
    model_save_path = os.path.join(dst_dir, str(dim), str(seed_id)+".pth")
    torch.save(model.state_dict(), model_save_path)

def save_all_models(src_dir="embeddings", dst_dir="models_concat", seed_num=10, dims=[8, 16, 32, 64, 128, 256]):
    os.makedirs(dst_dir, exist_ok=True)
    for dim in dims:
        os.makedirs(os.path.join(dst_dir, str(dim)), exist_ok=True)
        for seed_id in tqdm(range(seed_num)):
            run_and_save_model(src_dir, dst_dir, seed_id, dim)

            


def main():
    
#     train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
#     test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
#     test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    
    
#     embedding_path = "/hy-tmp/cora_experiment/embeddings/64/0.pt"
#     x_emb = torch.load(embedding_path, map_location='cpu')
#     x_origin = torch.from_numpy(np.load("x_origin.npy"))
#     x = torch.cat((x_emb, x_origin), dim=-1)
#     x = x.to(device)
    
#     model = MLPPredictor(x.size(-1)).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    
#     for epoch in range(1, 11):
#         loss = train(model, x, train_edges_undirected, optimizer, 100)
#         test(model, x, test_edges_undirected, test_neg_edges_undirected, 100)
#         print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    # save_all_models()
    

    # ############ calculate_overlap_for_pair_seeds #############
    # train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    # test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    # result_dir = "result/concat"
    # os.makedirs(result_dir, exist_ok=True)
    # df = pd.DataFrame(columns=["pos_mean_overlap_rate", "neg_mean_overlap_rate", "all_mean_overlap_rate"])
    # model_src_dir = "models_concat"
    # embedding_src_dir = "embeddings"
    # node_feature_path = "x_origin.npy"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_mean_overlap_rate, neg_mean_overlap_rate, all_mean_overlap_rate = calculate_one_dim_downstream_result_similarity(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, node_feature_path, dim=dim, seed_num=10)
    #     tmp_df = pd.DataFrame({"pos_mean_overlap_rate":[pos_mean_overlap_rate], "neg_mean_overlap_rate":[neg_mean_overlap_rate], "all_mean_overlap_rate":[all_mean_overlap_rate]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "overlap_for_pair_seeds.csv")
    # df.to_csv(save_path)
        
        
    #         ############ calculate_overlap_for_pair_seeds_all_true #############
    # train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    # test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    # result_dir = "result/concat"
    # os.makedirs(result_dir, exist_ok=True)
    # df = pd.DataFrame(columns=["pos_mean_overlap_rate", "neg_mean_overlap_rate", "all_mean_overlap_rate"])
    # model_src_dir = "models_concat"
    # embedding_src_dir = "embeddings"
    # node_feature_path = "x_origin.npy"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_mean_overlap_rate, neg_mean_overlap_rate, all_mean_overlap_rate = calculate_one_dim_downstream_result_similarity_all_true(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, node_feature_path, dim=dim, seed_num=10)
    #     tmp_df = pd.DataFrame({"pos_mean_overlap_rate":[pos_mean_overlap_rate], "neg_mean_overlap_rate":[neg_mean_overlap_rate], "all_mean_overlap_rate":[all_mean_overlap_rate]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "overlap_for_pair_seeds_all_true.csv")
    # df.to_csv(save_path)
    
    # ############calcaulate_overlap_for_all_seeds #############
    # train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    # test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    # result_dir = "result/concat"
    # os.makedirs(result_dir, exist_ok=True)
    # df = pd.DataFrame(columns=["pos_overlap_rate", "neg_overlap_rate", "all_overlap_rate"])
    # model_src_dir = "models_concat"
    # embedding_src_dir = "embeddings"
    # node_feature_path = "x_origin.npy"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_overlap_rate, neg_overlap_rate, all_overlap_rate = calcaulate_overlap_for_all_seeds(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, node_feature_path, dim=dim, seed_num=10)
    #     tmp_df = pd.DataFrame({"pos_overlap_rate":[pos_overlap_rate], "neg_overlap_rate":[neg_overlap_rate], "all_overlap_rate":[all_overlap_rate]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "overlap_for_all_seeds.csv")
    # df.to_csv(save_path)
    
        ############ calculate_acc_mean_std_for_one_dim #############
    train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    result_dir = "result/concat"
    df = pd.DataFrame(columns=["pos_acc_mean", "pos_acc_std", "neg_acc_mean", "neg_acc_std", "all_acc_mean", "all_acc_std"])
    model_src_dir = "models_concat"
    embedding_src_dir = "embeddings"
    node_feature_path = "x_origin.npy"
    for dim in [8, 16, 32, 64, 128, 256]:
        print(f"dim is {dim}")
        pos_acc_mean, pos_acc_std, neg_acc_mean, neg_acc_std, all_acc_mean, all_acc_std = calculate_acc_mean_std_for_one_dim(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, node_feature_path, dim=dim, seed_num=10)
        tmp_df = pd.DataFrame({"pos_acc_mean":[pos_acc_mean], "pos_acc_std":[pos_acc_std], "neg_acc_mean":[neg_acc_mean], "neg_acc_std":[neg_acc_std], "all_acc_mean":[all_acc_mean], "all_acc_std":[all_acc_std]})
        df = pd.concat([df, tmp_df])
    df.index = [8, 16, 32, 64, 128, 256]
    save_path = os.path.join(result_dir, "acc_mean_std.csv")
    df.to_csv(save_path)
    
    
if __name__ == '__main__':
    main()