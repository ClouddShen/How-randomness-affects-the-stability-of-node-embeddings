from mlp import LinkPredictor
import torch
import numpy as np
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm
import os
from utils import predict_downstream_result, calculate_two_model_downstream_result_similarity, calculate_one_dim_downstream_result_similarity
from utils import calcaulate_overlap_for_all_seeds, calculate_acc_mean_std_for_one_dim, calculate_one_dim_downstream_result_similarity_all_true
import pandas as pd




def main():
    device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    
    #    ############ calculate_overlap_for_pair_seeds #############
    # test_edges_undirected = torch.from_numpy(np.load("test_pos_edge.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edge.npy"))
    # result_dir = "result/concat"
    # os.makedirs(result_dir, exist_ok=True)
    # df = pd.DataFrame(columns=["pos_mean_overlap_rate", "neg_mean_overlap_rate", "all_mean_overlap_rate"])
    # model_src_dir = "models"
    # embedding_src_dir = "embeddings"
    # node_feature_path = "x_origin.npy"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_mean_overlap_rate, neg_mean_overlap_rate, all_mean_overlap_rate = calculate_one_dim_downstream_result_similarity(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, node_feature_path, dim=dim, seed_num=5)
    #     tmp_df = pd.DataFrame({"pos_mean_overlap_rate":[pos_mean_overlap_rate], "neg_mean_overlap_rate":[neg_mean_overlap_rate], "all_mean_overlap_rate":[all_mean_overlap_rate]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "overlap_for_pair_seeds.csv")
    # df.to_csv(save_path)
    
           ############ calculate_overlap_for_pair_seeds_all_true #############
    # test_edges_undirected = torch.from_numpy(np.load("test_pos_edge.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edge.npy"))
    # result_dir = "result/concat"
    # os.makedirs(result_dir, exist_ok=True)
    # df = pd.DataFrame(columns=["pos_mean_overlap_rate", "neg_mean_overlap_rate", "all_mean_overlap_rate"])
    # model_src_dir = "models"
    # embedding_src_dir = "embeddings"
    # node_feature_path = "x_origin.npy"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_mean_overlap_rate, neg_mean_overlap_rate, all_mean_overlap_rate = calculate_one_dim_downstream_result_similarity_all_true(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, node_feature_path, dim=dim, seed_num=5)
    #     tmp_df = pd.DataFrame({"pos_mean_overlap_rate":[pos_mean_overlap_rate], "neg_mean_overlap_rate":[neg_mean_overlap_rate], "all_mean_overlap_rate":[all_mean_overlap_rate]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "overlap_for_pair_seeds_all_true.csv")
    # df.to_csv(save_path)
    
    
        ############calcaulate_overlap_for_all_seeds #############
    test_edges_undirected = torch.from_numpy(np.load("test_pos_edge.npy"))
    test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edge.npy"))
    result_dir = "result/concat"
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame(columns=["pos_overlap_rate", "neg_overlap_rate", "all_overlap_rate"])
    model_src_dir = "models"
    embedding_src_dir = "embeddings"
    node_feature_path = "x_origin.npy"
    for dim in [8, 16, 32, 64, 128, 256]:
        print(f"dim is {dim}")
        pos_overlap_rate, neg_overlap_rate, all_overlap_rate = calcaulate_overlap_for_all_seeds(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, node_feature_path, dim=dim, seed_num=5)
        tmp_df = pd.DataFrame({"pos_overlap_rate":[pos_overlap_rate], "neg_overlap_rate":[neg_overlap_rate], "all_overlap_rate":[all_overlap_rate]})
        df = pd.concat([df, tmp_df])
    df.index = [8, 16, 32, 64, 128, 256]
    save_path = os.path.join(result_dir, "overlap_for_all_seeds.csv")
    df.to_csv(save_path)
    
    
    #     ############ calculate_acc_mean_std_for_one_dim #############
    # test_edges_undirected = torch.from_numpy(np.load("test_pos_edge.npy"))
    # test_neg_edges_undirected = torch.from_numpy(np.load("test_neg_edge.npy"))
    # result_dir = "result/concat"
    # df = pd.DataFrame(columns=["pos_acc_mean", "pos_acc_std", "neg_acc_mean", "neg_acc_std", "all_acc_mean", "all_acc_std"])
    # model_src_dir = "models"
    # embedding_src_dir = "embeddings"
    # node_feature_path = "x_origin.npy"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim is {dim}")
    #     pos_acc_mean, pos_acc_std, neg_acc_mean, neg_acc_std, all_acc_mean, all_acc_std = calculate_acc_mean_std_for_one_dim(model_src_dir, embedding_src_dir, test_edges_undirected, test_neg_edges_undirected, device, node_feature_path, dim=dim, seed_num=5)
    #     tmp_df = pd.DataFrame({"pos_acc_mean":[pos_acc_mean], "pos_acc_std":[pos_acc_std], "neg_acc_mean":[neg_acc_mean], "neg_acc_std":[neg_acc_std], "all_acc_mean":[all_acc_mean], "all_acc_std":[all_acc_std]})
    #     df = pd.concat([df, tmp_df])
    # df.index = [8, 16, 32, 64, 128, 256]
    # save_path = os.path.join(result_dir, "acc_mean_std.csv")
    # df.to_csv(save_path)
    
if __name__ == '__main__':
    main()