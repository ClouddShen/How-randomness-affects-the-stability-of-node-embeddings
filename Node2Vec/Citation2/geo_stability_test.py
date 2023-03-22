import numpy as np
import torch
from utils import procruste_similarity
from utils import knn_jaccard_similarity
from utils import second_ord_cos_similarity
import os
import csv

def compute_seed_sensitivity_for_citation2(metric, src_dir="", seed_num=5, dim = 8, num_node=1000): # metric is a function handler
    sim_list = []
    for i in range(seed_num):
        embedding_path = os.path.join(src_dir, str(dim), str(i) + ".pt")
        embedding1 = torch.load(embedding_path)
        embedding1 = embedding1[:num_node, :]
        for j in range(i + 1, seed_num):
            embedding_path = os.path.join(src_dir, str(dim), str(j) + ".pt")
            embedding2 = torch.load(embedding_path)
            embedding2 = embedding2[:num_node, :]
            sim_list.append(metric(embedding1, embedding2))
    sim_array = np.array(sim_list)
    sim_mean = np.mean(sim_array)
    sim_std = np.std(sim_array)
    return sim_mean, sim_std, sim_list





def compute_and_save_geometric_stability_for_citation2(model_name='Node2Vec', file_path='geo_result/Citation2_geometric_stability.csv'):
#     os.makedirs("geo_result", exist_ok=True)
#     embed_dim = [8, 16, 32, 64, 128, 256]
#     num_node_list = [1000, 2000, 4000, 8000, 16000]
#     for dim in embed_dim:
#         for num_node in num_node_list:
#             print("------------- dim: ", dim, ", num_node: ", num_node, "-------------")
#             pro_sim_mean, pro_sim_std, pro_sim_list = compute_seed_sensitivity_for_citation2(procruste_similarity, src_dir="embeddings", dim=dim, seed_num=5, num_node=num_node)
#             print(pro_sim_mean, pro_sim_std)
#             knn_sim_mean, knn_sim_std, knn_sim_list = compute_seed_sensitivity_for_citation2(knn_jaccard_similarity,  src_dir="embeddings", dim=dim, seed_num=5, num_node=num_node)
#             print(knn_sim_mean, knn_sim_std)
#             sec_sim_mean, sec_sim_std, sec_sim_list = compute_seed_sensitivity_for_citation2(second_ord_cos_similarity,  src_dir="embeddings", dim=dim, seed_num=5, num_node=num_node)
#             print(sec_sim_mean, sec_sim_std)
        
#             length = len(pro_sim_list)
#             data_name_column = [model_name] * length
#             dim_column = [dim] * length
#             num_node_column = [num_node] * length
#             rows = zip(data_name_column, dim_column, num_node_column, pro_sim_list, knn_sim_list, sec_sim_list)


#             file_exists = os.path.exists(file_path)

#             with open(file_path, mode='a', newline='') as file:
#                 writer = csv.writer(file)
#                 if not file_exists:
#                     writer.writerow(['model_name', 'embedding_dim', 'num_node', 'pro_sim', 'knn_sim', 'sec_sim'])
#                 for row in rows:
#                     writer.writerow(row)

    embed_dim = [8, 16, 32, 64, 128, 256]
    num_node_list = [1000]
    for dim in embed_dim:
        for num_node in num_node_list:
            sec_sim_mean, sec_sim_std, sec_sim_list = compute_seed_sensitivity_for_citation2(second_ord_cos_similarity,  src_dir="embeddings", dim=dim, seed_num=5, num_node=num_node)

def main():
    # src_dir = "embeddings"
    # for dim in [8, 16, 32, 64, 128, 256]:
    #     print(f"dim {dim}")
    #     sim_mean, sim_std = test_seed_sensitiveity(procruste_similarity, src_dir, seed_num=5, dim=dim)
    #     print(f"Pro_mean {sim_mean}, Pro_std {sim_std}")
    #     # sim_mean, sim_std = test_seed_sensitiveity(knn_jaccard_similarity, src_dir, seed_num=5, dim=dim)
    #     # print(f"knn_mean {sim_mean}, knn_std {sim_std}")
    #     # sim_mean, sim_std = test_seed_sensitiveity(second_ord_cos_similarity, src_dir, seed_num=5, dim=dim)
    #     # print(f"second_mean {sim_mean}, second_std {sim_std}")
    
    compute_and_save_geometric_stability_for_citation2(model_name='Node2Vec', file_path='geo_result/Citation2_geometric_stability.csv')
    
    
if __name__ == '__main__':
    main()