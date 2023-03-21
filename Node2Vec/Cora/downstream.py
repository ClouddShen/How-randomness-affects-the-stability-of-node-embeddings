from mlp import MLPPredictor
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
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
    model.eval()
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

def accuracy_sensitivity(src_dir="", seed_num=10, dim = 8):
    train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    acc_list = []
    for i in tqdm(range(seed_num)):
        embedding_path = os.path.join(src_dir, str(dim), str(i) + ".pt")
        x = torch.load(embedding_path, map_location='cpu')
        x = x.to(device)
        model = MLPPredictor(x.size(-1)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        for epoch in range(1, 101):
            loss = train(model, x, train_edges_undirected, optimizer, 100)
            test(model, x, test_edges_undirected, test_neg_edges_undirected, 100)
        acc = test(model, x, test_edges_undirected, test_neg_edges_undirected, 100)
        acc_list.append(acc)
    acc_array = np.array(acc_list)
    acc_mean = np.mean(acc_array)
    acc_std = np.std(acc_array)
    return acc_mean, acc_std

def test_accuracy_sensitivity():
    src_dir = "embeddings"
    for dim in [8, 16, 32, 64, 128, 256]:
        print(f"dim {dim}")
        acc_mean, acc_std = accuracy_sensitivity(src_dir, seed_num=10, dim=dim)
        print(f"Acc_mean {acc_mean}, Acc_std {acc_std}")

        
def run_and_save_model(src_dir="", dst_dir="", seed_id=0, dim=8):
    train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    embedding_path = os.path.join(src_dir, str(dim), str(seed_id) + ".pt")
    x = torch.load(embedding_path, map_location='cpu')
    x = x.to(device)
    model = MLPPredictor(x.size(-1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(1, 101):
        loss = train(model, x, train_edges_undirected, optimizer, 100)
        test(model, x, test_edges_undirected, test_neg_edges_undirected, 100)
    model_save_path = os.path.join(dst_dir, str(dim), str(seed_id)+".pth")
    torch.save(model.state_dict(), model_save_path)

def save_all_models(src_dir="embeddings", dst_dir="models", seed_num=10, dims=[8, 16, 32, 64, 128, 256]):
    os.makedirs(dst_dir, exist_ok=True)
    for dim in dims:
        os.makedirs(os.path.join(dst_dir, str(dim)), exist_ok=True)
        for seed_id in tqdm(range(seed_num)):
            run_and_save_model(src_dir, dst_dir, seed_id, dim)

        
def main():
#     train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
#     test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
#     test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    
    
#     embedding_path = "/hy-tmp/cora_experiment/embeddings/256/0.pt"
#     x = torch.load(embedding_path, map_location='cpu')
#     x = x.to(device)
    
#     model = MLPPredictor(x.size(-1)).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    
#     for epoch in range(1, 101):
#         loss = train(model, x, train_edges_undirected, optimizer, 100)
#         test(model, x, test_edges_undirected, test_neg_edges_undirected, 100)


    # save_all_models()
    
    
    ############test predict_downstream_result method #############
    # train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    # test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    # embedding_path = "/hy-tmp/cora_experiment/embeddings/256/0.pt"
    # x = torch.load(embedding_path, map_location='cpu')
    # x = x.to(device)
    # model = MLPPredictor(x.size(-1)).to(device)
    # model.load_state_dict(torch.load('models/256/0.pth'))
    # pos_pred, neg_pred = predict_downstream_result(model, x, test_edges_undirected, test_neg_edges_undirected)
    # print(sum(pos_pred))
    # print(len(pos_pred))
    # print(sum(neg_pred))
    
    
#     ############test calculate_two_model_downstream_result_similarity method #############
#     train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
#     test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
#     test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    
#     embedding_path1 = "/hy-tmp/cora_experiment/embeddings/256/0.pt"
#     x1 = torch.load(embedding_path1, map_location='cpu')
#     x1 = x1.to(device)
    
#     embedding_path2 = "/hy-tmp/cora_experiment/embeddings/256/1.pt"
#     x2 = torch.load(embedding_path2, map_location='cpu')
#     x2 = x2.to(device)
    
#     model1 = MLPPredictor(x1.size(-1)).to(device)
#     model1.load_state_dict(torch.load('models/256/0.pth'))
#     model2 = MLPPredictor(x2.size(-1)).to(device)
#     model2.load_state_dict(torch.load('models/256/1.pth'))
#     print(calculate_two_model_downstream_result_similarity(model1, model2, x1, x2, test_edges_undirected, test_neg_edges_undirected))


    # ############test calculate_one_dim_downstream_result_similarity method #############
    # train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    # test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    # model_src_dir = "models"
    # embedding_src_dir = "embeddings"
    # print(calculate_one_dim_downstream_result_similarity(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, dim=8, seed_num=10))
    
    
    # ############calcaulate_overlap_for_all_seeds #############
    # result_dir = "result"
    # df = pd.DataFrame(columns=["pos_overlap_rate", "neg_overlap_rate", "all_overlap_rate"])
    # train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    # test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    # model_src_dir = "models"
    # embedding_src_dir = "embeddings"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_overlap_rate, neg_overlap_rate, all_overlap_rate = calcaulate_overlap_for_all_seeds(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, dim=dim, seed_num=10)
    #     tmp_df = pd.DataFrame({"pos_overlap_rate":[pos_overlap_rate], "neg_overlap_rate":[neg_overlap_rate], "all_overlap_rate":[all_overlap_rate]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "overlap_for_all_seeds.csv")
    # df.to_csv(save_path)
    
    
    
    
    # ############ calculate_overlap_for_pair_seeds #############
    # train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    # test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    # result_dir = "result"
    # df = pd.DataFrame(columns=["pos_mean_overlap_rate", "neg_mean_overlap_rate", "all_mean_overlap_rate"])
    # model_src_dir = "models"
    # embedding_src_dir = "embeddings"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_mean_overlap_rate, neg_mean_overlap_rate, all_mean_overlap_rate = calculate_one_dim_downstream_result_similarity(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, dim=dim, seed_num=10)
    #     tmp_df = pd.DataFrame({"pos_mean_overlap_rate":[pos_mean_overlap_rate], "neg_mean_overlap_rate":[neg_mean_overlap_rate], "all_mean_overlap_rate":[all_mean_overlap_rate]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "overlap_for_pair_seeds.csv")
    # df.to_csv(save_path)
    
    #     ############ calculate_overlap_for_pair_seeds_all_true #############
    # train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    # test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    # result_dir = "result"
    # df = pd.DataFrame(columns=["pos_mean_overlap_rate", "neg_mean_overlap_rate", "all_mean_overlap_rate"])
    # model_src_dir = "models"
    # embedding_src_dir = "embeddings"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_mean_overlap_rate, neg_mean_overlap_rate, all_mean_overlap_rate = calculate_one_dim_downstream_result_similarity_all_true(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, dim=dim, seed_num=10)
    #     tmp_df = pd.DataFrame({"pos_mean_overlap_rate":[pos_mean_overlap_rate], "neg_mean_overlap_rate":[neg_mean_overlap_rate], "all_mean_overlap_rate":[all_mean_overlap_rate]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "overlap_for_pair_seeds_all_true.csv")
    # df.to_csv(save_path)
        
    
    
    # ############ calculate_acc_mean_std_for_one_dim #############
    # train_edges_undirected = torch.from_numpy(np.load("train_split.npy"))
    # test_edges_undirected = torch.from_numpy(np.load("test_split.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edges.npy"))
    # result_dir = "result"
    # df = pd.DataFrame(columns=["pos_acc_mean", "pos_acc_std", "neg_acc_mean", "neg_acc_std", "all_acc_mean", "all_acc_std"])
    # model_src_dir = "models"
    # embedding_src_dir = "embeddings"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_acc_mean, pos_acc_std, neg_acc_mean, neg_acc_std, all_acc_mean, all_acc_std = calculate_acc_mean_std_for_one_dim(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, dim=dim, seed_num=10)
    #     tmp_df = pd.DataFrame({"pos_acc_mean":[pos_acc_mean], "pos_acc_std":[pos_acc_std], "neg_acc_mean":[neg_acc_mean], "neg_acc_std":[neg_acc_std], "all_acc_mean":[all_acc_mean], "all_acc_std":[all_acc_std]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "acc_mean_std.csv")
    # df.to_csv(save_path)
    
if __name__ == '__main__':
    main()