from scipy.spatial import procrustes
import csv
import torch
import numpy as np
import numpy as np
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity


def procruste_similarity(embedding1, embedding2):
    _, _, disparity = procrustes(embedding1, embedding2)
    return 1 - disparity

def calculate_knn_indices(embedding, k=20):
    similarity_matrix = cosine_similarity(embedding)
    # print(similarity_matrix.shape, similarity_matrix[0])
    np.fill_diagonal(similarity_matrix, -np.inf)
    neighbor_indices = np.zeros((embedding.shape[0], k), dtype=np.int32)
    for i in range(embedding.shape[0]):
        indices = np.argsort(similarity_matrix[i])[::-1][:k] # top k similarity corresponding indices
        neighbor_indices[i] = indices
    return neighbor_indices, similarity_matrix
    
def knn_jaccard_similarity(embedding1, embedding2, k=20):
    mtx1, mtx2, _ = procrustes(embedding1, embedding2)
    neighbor_indices1, _ = calculate_knn_indices(mtx1, k)
    neighbor_indices2, _ = calculate_knn_indices(mtx2, k)
    intersection_num_result = []
    union_num_result = []
    for row1, row2 in zip(neighbor_indices1, neighbor_indices2):
        intersection_num = len(set(row1).intersection(row2))
        intersection_num_result.append(intersection_num)
        union_num =  2 * k - intersection_num
        union_num_result.append(union_num)
    similarity = np.mean(np.array(intersection_num_result)/np.array(union_num_result))
    return similarity

def second_ord_cos_similarity(embedding1, embedding2, k=20):
    mtx1, mtx2, _ = procrustes(embedding1, embedding2)
    neighbor_indices1, sim_matrix1 = calculate_knn_indices(mtx1, k)
    neighbor_indices2, sim_matrix2 = calculate_knn_indices(mtx2, k)
    first_sim_neighbors1 = np.zeros((embedding1.shape[0], k*2))
    first_sim_neighbors2 = np.zeros((embedding2.shape[0], k*2))
    second_sim_matrix = np.zeros((embedding2.shape[0], 1))
    for i, (indices1, indices2) in enumerate(zip(neighbor_indices1, neighbor_indices2)):
        union_indices = list(set(indices1).union(set(indices2)))
        sim1 = sim_matrix1[i][union_indices]
        first_sim_neighbors1[i][:len(sim1)] = sim1
        sim2 = sim_matrix2[i][union_indices]
        first_sim_neighbors2[i][:len(sim2)] = sim2
    second_sim_matrix = np.diag(cosine_similarity(first_sim_neighbors1, first_sim_neighbors2))
    mean_similarity = np.mean(second_sim_matrix)
    return mean_similarity

def compute_seed_sensitivity_for_cora(metric_fcn, embed_dim, seed_num):
    src_dir = f'./cora/embedding_data/{str(embed_dim)}'
    metric_list = []
    for i in range(seed_num):
        embedding_path = os.path.join(src_dir, str(i) + ".npy")
        embedding1 = np.load(embedding_path)
        for j in range(i + 1, seed_num):
            embedding_path = os.path.join(src_dir, str(j) + ".npy")
            embedding2 = np.load(embedding_path)
            print(f"Compute seed {str(i)} and seed {str(j)}...")
            metric_list.append(metric_fcn(embedding1, embedding2))
    metric_array = np.array(metric_list)
    # print(metric_array)
    metric_mean = np.mean(metric_array)
    metric_std = np.std(metric_array)
    return metric_mean, metric_std, metric_list

def compute_and_save_geometric_stability_for_cora(model_name='GraphSAGE', file_path='./visualization/cora_geometric_stability.csv'):
    embed_dim = [8, 16, 32, 64, 128, 256]
    for dim in embed_dim:
        print("-------------dim---------: ", dim)
        pro_sim_mean, pro_sim_std, pro_sim_list = compute_seed_sensitivity_for_cora(procruste_similarity, dim, 10)
        print(pro_sim_list)
        print(pro_sim_mean, pro_sim_std)
        knn_sim_mean, knn_sim_std, knn_sim_list = compute_seed_sensitivity_for_cora(knn_jaccard_similarity, dim, 10)
        print(knn_sim_list)
        print(knn_sim_mean, knn_sim_std)
        sec_sim_mean, sec_sim_std, sec_sim_list = compute_seed_sensitivity_for_cora(second_ord_cos_similarity, dim, 10)
        print(sec_sim_list)
        print(sec_sim_mean, sec_sim_std)

        dim_column = [dim] * len(pro_sim_list)
        data_name_column = [model_name] * len(pro_sim_list)
        rows = zip(dim_column, data_name_column, pro_sim_list, knn_sim_list, sec_sim_list)

        file_exists = os.path.exists(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['embedding_dim', 'model_name', 'pro_sim', 'knn_sim', 'sec_sim'])
            for row in rows:
                writer.writerow(row)


def compute_seed_sensitivity_for_citation2(metric_fcn, embed_dim, seed_num, num_node):
    src_dir = f'./citation2/embedding_data/{str(embed_dim)}'
    metric_list = []
    for i in range(seed_num):
        embedding_path = os.path.join(src_dir, str(i) + ".npy")
        embedding1 = np.load(embedding_path)
        embedding1 = embedding1[:num_node, :]
        for j in range(i + 1, seed_num):
            embedding_path = os.path.join(src_dir, str(j) + ".npy")
            embedding2 = np.load(embedding_path)
            embedding2 = embedding2[:num_node, :]
            print(f"Compute seed {str(i)} and seed {str(j)}...")
            metric_list.append(metric_fcn(embedding1, embedding2))
    metric_array = np.array(metric_list)
    # print(metric_array)
    metric_mean = np.mean(metric_array)
    metric_std = np.std(metric_array)
    return metric_mean, metric_std, metric_list


def compute_and_save_geometric_stability_for_citation2(model_name='GraphSAGE', file_path='./visualization/citation2_geometric_stability.csv'):
    embed_dim = [8, 16, 32, 64, 128, 256]
    num_node_list = [1000, 2000, 4000, 8000, 16000]
    for dim in embed_dim:
        for num_node in num_node_list:
            print("------------- dim: ", dim, ", num_node: ", num_node, "-------------")
            pro_sim_mean, pro_sim_std, pro_sim_list = compute_seed_sensitivity_for_citation2(procruste_similarity, dim, 5, num_node)
            print(pro_sim_mean, pro_sim_std)
            knn_sim_mean, knn_sim_std, knn_sim_list = compute_seed_sensitivity_for_citation2(knn_jaccard_similarity, dim, 5, num_node)
            print(knn_sim_mean, knn_sim_std)
            sec_sim_mean, sec_sim_std, sec_sim_list = compute_seed_sensitivity_for_citation2(second_ord_cos_similarity, dim, 5, num_node)
            print(sec_sim_mean, sec_sim_std)
        
            length = len(pro_sim_list)
            data_name_column = [model_name] * length
            dim_column = [dim] * length
            num_node_column = [num_node] * length
            rows = zip(data_name_column, dim_column, num_node_column, pro_sim_list, knn_sim_list, sec_sim_list)


            file_exists = os.path.exists(file_path)

            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['model_name', 'embedding_dim', 'num_node', 'pro_sim', 'knn_sim', 'sec_sim'])
                for row in rows:
                    writer.writerow(row)
    
if __name__ == '__main__':
    # compute_and_save_geometric_stability_for_cora()
    compute_and_save_geometric_stability_for_citation2()
    
