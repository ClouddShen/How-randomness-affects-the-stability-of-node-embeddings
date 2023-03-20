from scipy.spatial import procrustes
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
    for i, (indices1, indices2) in enumerate(zip(neighbor_indices1, neighbor_indices2)):
        union_indices = list(set(indices1).union(set(indices2)))
        sim1 = sim_matrix1[i][union_indices]
        first_sim_neighbors1[i][:len(sim1)] = sim1
        sim2 = sim_matrix2[i][union_indices]
        first_sim_neighbors2[i][:len(sim2)] = sim2
    second_sim_matrix = np.diag(cosine_similarity(first_sim_neighbors1, first_sim_neighbors2))
    mean_similarity = np.mean(second_sim_matrix)
    return mean_similarity


def compute_seed_sensitivity(metric_fcn, embed_dim, seed_num):
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
    print(metric_array)
    metric_mean = np.mean(metric_array)
    metric_std = np.std(metric_array)
    return metric_mean, metric_std




def main():
    embed_dim = [8, 16, 32, 64, 128, 256]
    for dim in embed_dim:
        print("-------------dim---------: ", dim)
        pro_sim_mean, pro_sim_std = compute_seed_sensitivity(procruste_similarity, dim, 10)
        knn_sim_mean, knn_sim_std = compute_seed_sensitivity(knn_jaccard_similarity, dim, 10)
        sec_sim_mean, sec_sim_std = compute_seed_sensitivity(second_ord_cos_similarity, dim, 10)
        print(pro_sim_mean, pro_sim_std)
        print(knn_sim_mean, knn_sim_std)
        print(sec_sim_mean, sec_sim_std) 
    
if __name__ == '__main__':
    # main()
    acc_8 = [0.9367588932806324, 0.9461462450592886, 0.9609683794466403, 0.9525691699604744, 0.9298418972332015, 0.950098814229249, 0.8414031620553359, 0.9807312252964426, 0.9377470355731226, 0.9491106719367589]
    acc_16 = [0.9812252964426877, 0.9115612648221344, 0.9412055335968379, 0.9071146245059288, 0.9441699604743083, 0.9841897233201581, 0.9510869565217391, 0.9392292490118577, 0.9767786561264822, 0.9496047430830039]
    acc_32 = [0.9570158102766798, 0.9446640316205533, 0.9856719367588933, 0.9382411067193676, 0.9481225296442688, 0.966897233201581, 0.983201581027668, 0.982707509881423, 0.9387351778656127, 0.9481225296442688]
    acc_64 = [0.9377470355731226, 0.9407114624505929, 0.9451581027667985, 0.9347826086956522, 0.9234189723320159, 0.9273715415019763, 0.9787549407114624, 0.9426877470355731, 0.9426877470355731, 0.9090909090909091]
    acc_128 = [0.950098814229249, 0.9570158102766798, 0.9441699604743083, 0.9456521739130435, 0.9550395256916996, 0.9407114624505929, 0.9461462450592886, 0.9540513833992095, 0.9372529644268774, 0.9431818181818182]
    acc_256 = [0.9510869565217391, 0.9426877470355731, 0.9530632411067194, 0.9703557312252964, 0.9481225296442688, 0.9515810276679841, 0.9288537549407114, 0.9491106719367589, 0.9570158102766798, 0.950098814229249]
    print(np.mean(acc_8), np.std(acc_8))
    print(np.mean(acc_16), np.std(acc_16))
    print(np.mean(acc_32), np.std(acc_32))
    print(np.mean(acc_64), np.std(acc_64))
    print(np.mean(acc_128), np.std(acc_128))
    print(np.mean(acc_256), np.std(acc_256))
