import numpy as np
import torch
from utils import procruste_similarity
from utils import knn_jaccard_similarity
from utils import second_ord_cos_similarity
import os
import pandas as pd
import csv

def compute_seed_sensitivity(metric, src_dir="", seed_num=10, dim = 8): # metric is a function handler
    sim_list = []
    for i in range(seed_num):
        embedding_path = os.path.join(src_dir, str(dim), str(i) + ".pt")
        embedding1 = torch.load(embedding_path)
        for j in range(i + 1, seed_num):
            embedding_path = os.path.join(src_dir, str(dim), str(j) + ".pt")
            embedding2 = torch.load(embedding_path)
            sim_list.append(metric(embedding1, embedding2))
    sim_array = np.array(sim_list)
    sim_mean = np.mean(sim_array)
    sim_std = np.std(sim_array)
    return sim_mean, sim_std, sim_list


def compute_and_save_geometric_stability(model_name='Node2Vec', file_path='geo_result/Cora_geometric_stability.csv'):
    embed_dim = [8, 16, 32, 64, 128, 256]
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['embedding_dim', 'model_name', 'pro_sim', 'knn_sim', 'sec_sim'])
        for dim in embed_dim:
            print("-------------dim---------: ", dim)
            pro_sim_mean, pro_sim_std, pro_sim_list = compute_seed_sensitivity(procruste_similarity, src_dir="embeddings", dim=dim, seed_num=10)
            print(pro_sim_list)
            print(pro_sim_mean, pro_sim_std)
            knn_sim_mean, knn_sim_std, knn_sim_list = compute_seed_sensitivity(knn_jaccard_similarity, src_dir="embeddings", dim=dim, seed_num=10)
            print(knn_sim_list)
            print(knn_sim_mean, knn_sim_std)
            sec_sim_mean, sec_sim_std, sec_sim_list = compute_seed_sensitivity(second_ord_cos_similarity, src_dir="embeddings", dim=dim, seed_num=10)
            print(sec_sim_list)
            print(sec_sim_mean, sec_sim_std)

            dim_column = [dim] * len(pro_sim_list)
            data_name_column = [model_name] * len(pro_sim_list)
            rows = zip(dim_column, data_name_column, pro_sim_list, knn_sim_list, sec_sim_list)

            file_exists = os.path.exists(file_path)
            for row in rows:
                writer.writerow(row)

def main():
    # save_dir = "geo_result"
    # save_pro_dir = os.path.join(save_dir, "pro")
    # save_jac_dir = os.path.join(save_dir, "jac")
    # save_sec_dir = os.path.join(save_dir, "sec")
    # src_dir = "embeddings"
    # pro_df = pd.DataFrame(columns = ["sim_mean", "sim_std"])
    # jac_df = pd.DataFrame(columns = ["sim_mean", "sim_std"])
    # sec_df = pd.DataFrame(columns = ["sim_mean", "sim_std"])
    # os.makedirs(save_pro_dir, exist_ok=True)
    # os.makedirs(save_jac_dir, exist_ok=True)
    # os.makedirs(save_sec_dir, exist_ok=True)
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim {dim}")
    #     sim_mean, sim_std, sim_array = test_seed_sensitiveity(procruste_similarity, src_dir, seed_num=10, dim=dim)
    #     tmp_df = pd.DataFrame({"sim_mean":[sim_mean], "sim_std":[sim_std]})
    #     pro_df = pd.concat((pro_df, tmp_df))
    #     np.save(os.path.join(save_pro_dir, str(dim) + ".npy"), sim_array)
    #     print(f"Pro_mean {sim_mean}, Pro_std {sim_std}")
    #     sim_mean, sim_std, sim_array = test_seed_sensitiveity(knn_jaccard_similarity, src_dir, seed_num=10, dim=dim)
    #     tmp_df = pd.DataFrame({"sim_mean":[sim_mean], "sim_std":[sim_std]})
    #     jac_df = pd.concat((jac_df, tmp_df))
    #     np.save(os.path.join(save_jac_dir, str(dim) + ".npy"), sim_array)
    #     print(f"Jac_mean {sim_mean}, Jac_std {sim_std}")
    #     sim_mean, sim_std, sim_array = test_seed_sensitiveity(second_ord_cos_similarity, src_dir, seed_num=10, dim=dim)
    #     tmp_df = pd.DataFrame({"sim_mean":[sim_mean], "sim_std":[sim_std]})
    #     sec_df = pd.concat((sec_df, tmp_df))
    #     np.save(os.path.join(save_sec_dir, str(dim) + ".npy"), sim_array)
    #     print(f"Sec_mean {sim_mean}, Sec_std {sim_std}")
    # pro_df.index = [8, 16, 32, 64, 128, 256]
    # jac_df.index = [8, 16, 32, 64, 128, 256]
    # sec_df.index = [8, 16, 32, 64, 128, 256]
    # pro_df.to_csv(os.path.join(save_pro_dir, "pro.csv"))
    # jac_df.to_csv(os.path.join(save_jac_dir, "jac.csv"))
    # sec_df.to_csv(os.path.join(save_sec_dir, "sec.csv"))
    compute_and_save_geometric_stability(model_name='Node2Vec', file_path='geo_result/Cora_geometric_stability.csv')

    
if __name__ == '__main__':
    main()