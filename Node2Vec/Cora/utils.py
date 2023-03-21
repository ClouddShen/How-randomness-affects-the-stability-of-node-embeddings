from scipy.spatial import procrustes
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
import os
from mlp import MLPPredictor


def procruste_similarity(embedding1, embedding2):
    _, _, disparity = procrustes(embedding1, embedding2)
    return 1 - disparity


def calculate_knn_indices(embedding, k=20):
    similarity_matrix = cosine_similarity(embedding)
    np.fill_diagonal(similarity_matrix, -np.inf)
    neighbor_indices = np.zeros((embedding.shape[0], k), dtype=np.int32)
    for i in range(embedding.shape[0]):
        indices = np.argsort(similarity_matrix[i])[::-1][:k]  # top k similarity corresponding indices
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
        union_num = 2 * k - intersection_num
        union_num_result.append(union_num)
    similarity = np.mean(np.array(intersection_num_result) / np.array(union_num_result))
    return similarity


def second_ord_cos_similarity(embedding1, embedding2, k=20):
    mtx1, mtx2, _ = procrustes(embedding1, embedding2)
    neighbor_indices1, sim_matrix1 = calculate_knn_indices(mtx1, k)
    neighbor_indices2, sim_matrix2 = calculate_knn_indices(mtx2, k)
    first_sim_neighbors1 = np.zeros((embedding1.shape[0], k * 2))
    first_sim_neighbors2 = np.zeros((embedding2.shape[0], k * 2))
    for i, (indices1, indices2) in enumerate(zip(neighbor_indices1, neighbor_indices2)):
        union_indices = list(set(indices1).union(set(indices2)))
        sim1 = sim_matrix1[i][union_indices]
        first_sim_neighbors1[i][:len(sim1)] = sim1
        sim2 = sim_matrix2[i][union_indices]
        first_sim_neighbors2[i][:len(sim2)] = sim2
    second_sim_matrix = np.diag(cosine_similarity(first_sim_neighbors1, first_sim_neighbors2))
    mean_similarity = np.mean(second_sim_matrix)
    return mean_similarity


def predict_downstream_result(model, x, pos_edge, neg_edge):
    model.eval()
    src_pos_node = pos_edge[0].to(x.device)
    dst_pos_node = pos_edge[1].to(x.device)
    src_neg_node = neg_edge[0].to(x.device)
    dst_neg_node = neg_edge[1].to(x.device)
    pos_prediction = model(x[src_pos_node], x[dst_pos_node]).squeeze().cpu() >= 0.5
    neg_prediction = model(x[src_neg_node], x[dst_neg_node]).squeeze().cpu() < 0.5
    return pos_prediction, neg_prediction


def calculate_two_model_downstream_result_similarity(model1, model2, x1, x2, pos_edge, neg_edge):
    pos_pred1, neg_pred1 = predict_downstream_result(model1, x1, pos_edge, neg_edge)
    pos_pred2, neg_pred2 = predict_downstream_result(model2, x2, pos_edge, neg_edge)
    pos_overlap_num = sum(pos_pred1 == pos_pred2)
    neg_overlap_num = sum(neg_pred1 == neg_pred2)
    return pos_overlap_num, neg_overlap_num


def calculate_two_model_downstream_result_similarity_all_true(model1, model2, x1, x2, pos_edge, neg_edge):
    pos_pred1, neg_pred1 = predict_downstream_result(model1, x1, pos_edge, neg_edge)
    pos_pred2, neg_pred2 = predict_downstream_result(model2, x2, pos_edge, neg_edge)
    pos_overlap_num = sum(pos_pred1 & pos_pred2)
    neg_overlap_num = sum(neg_pred1 & neg_pred2)
    return pos_overlap_num, neg_overlap_num


def calculate_one_dim_downstream_result_similarity_all_true(model_src_dir, embedding_src_dir, pos_edge, neg_edge,
                                                            device, node_feature_path="", dim=8, seed_num=10):
    pos_edge_num = pos_edge.shape[1]
    neg_edge_num = neg_edge.shape[1]
    pos_overlap_num_list = []
    neg_overlap_num_list = []
    for i in range(seed_num):
        for j in range(i + 1, seed_num):
            model1, x1 = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device,
                                                             node_feature_path, dim=dim, seed=i)
            model2, x2 = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device,
                                                             node_feature_path, dim=dim, seed=j)
            pos_overlap_num, neg_overlap_num = calculate_two_model_downstream_result_similarity_all_true(model1, model2,
                                                                                                         x1, x2,
                                                                                                         pos_edge,
                                                                                                         neg_edge)
            pos_overlap_num_list.append(pos_overlap_num)
            neg_overlap_num_list.append(neg_overlap_num)
    all_overlap_num_list = [pos + neg for pos, neg in zip(pos_overlap_num_list, neg_overlap_num_list)]

    pos_overlap_num_array = np.array(pos_overlap_num_list)
    neg_overlap_num_array = np.array(neg_overlap_num_list)
    all_overlap_num_array = np.array(all_overlap_num_list)

    pos_overlap_num_mean = np.mean(pos_overlap_num_array)
    pos_overlap_num_std = np.std(pos_overlap_num_array)
    neg_overlap_num_mean = np.mean(neg_overlap_num_array)
    neg_overlap_num_std = np.std(neg_overlap_num_array)
    all_overlap_num_mean = np.mean(all_overlap_num_array)
    all_overlap_num_std = np.std(all_overlap_num_array)

    pos_overlap_rate = pos_overlap_num_mean / pos_edge_num
    neg_overlap_rate = neg_overlap_num_mean / neg_edge_num
    all_overlap_rate = all_overlap_num_mean / (pos_edge_num + neg_edge_num)
    return pos_overlap_rate, neg_overlap_rate, all_overlap_rate


def load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, node_feature_path="", dim=8, seed=0):
    model_param_path = os.path.join(model_src_dir, str(dim), str(seed) + ".pth")
    embedding_path = os.path.join(embedding_src_dir, str(dim), str(seed) + ".pt")
    if node_feature_path == "":
        x = torch.load(embedding_path, map_location='cpu')
        x = x.to(device)
    else:
        x_emb = torch.load(embedding_path, map_location='cpu')
        x_origin = torch.from_numpy(np.load("x_origin.npy"))
        x = torch.cat((x_emb, x_origin), dim=-1)
        x = x.to(device)
    model = MLPPredictor(x.size(-1)).to(device).to(device)
    model.load_state_dict(torch.load(model_param_path))
    return model, x


def calculate_one_dim_downstream_result_similarity(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, node_feature_path="", dim=8,
                                                   seed_num=10):
    pos_edge_num = pos_edge.shape[1]
    neg_edge_num = neg_edge.shape[1]
    pos_overlap_num_list = []
    neg_overlap_num_list = []
    for i in range(seed_num):
        for j in range(i + 1, seed_num):
            model1, x1 = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, node_feature_path, dim=dim, seed=i)
            model2, x2 = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, node_feature_path, dim=dim, seed=j)
            pos_overlap_num, neg_overlap_num = calculate_two_model_downstream_result_similarity(model1, model2, x1, x2,
                                                                                                pos_edge, neg_edge)
            pos_overlap_num_list.append(pos_overlap_num)
            neg_overlap_num_list.append(neg_overlap_num)
    all_overlap_num_list = [pos + neg for pos, neg in zip(pos_overlap_num_list, neg_overlap_num_list)]

    pos_overlap_num_array = np.array(pos_overlap_num_list)
    neg_overlap_num_array = np.array(neg_overlap_num_list)
    all_overlap_num_array = np.array(all_overlap_num_list)

    pos_overlap_num_mean = np.mean(pos_overlap_num_array)
    pos_overlap_num_std = np.std(pos_overlap_num_array)
    neg_overlap_num_mean = np.mean(neg_overlap_num_array)
    neg_overlap_num_std = np.std(neg_overlap_num_array)
    all_overlap_num_mean = np.mean(all_overlap_num_array)
    all_overlap_num_std = np.std(all_overlap_num_array)

    pos_overlap_rate = pos_overlap_num_mean / pos_edge_num
    neg_overlap_rate = neg_overlap_num_mean / neg_edge_num
    all_overlap_rate = all_overlap_num_mean / (pos_edge_num + neg_edge_num)
    return pos_overlap_rate, neg_overlap_rate, all_overlap_rate


def calcaulate_overlap_for_all_seeds(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, node_feature_path="", dim=8, seed_num=10):
    pos_edge_num = pos_edge.shape[1]
    neg_edge_num = neg_edge.shape[1]
    pos_pred_overlap = np.ones(pos_edge.shape[1]).astype(np.int32).astype(bool)
    neg_pred_overlap = np.ones(neg_edge.shape[1]).astype(np.int32).astype(bool)
    all_pred_overlap = np.ones(pos_edge.shape[1] + neg_edge.shape[1]).astype(np.int32).astype(bool)
    for i in range(seed_num):
        model, x = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, node_feature_path, dim=dim, seed=i)
        pos_prediction, neg_prediction = predict_downstream_result(model, x, pos_edge, neg_edge)
        pos_pred_overlap = ~(pos_prediction.numpy().astype(bool) ^ pos_pred_overlap)
        neg_pred_overlap = ~(neg_prediction.numpy().astype(bool) ^ neg_pred_overlap)
        all_prediction = np.append(pos_prediction.numpy().astype(bool), neg_prediction.numpy().astype(bool))
        all_pred_overlap = ~(all_prediction ^ all_pred_overlap)
    return sum(pos_pred_overlap) / pos_edge_num, sum(neg_pred_overlap) / neg_edge_num, sum(all_pred_overlap) / (
                pos_edge_num + neg_edge_num)


def calculate_acc_mean_std_for_one_dim(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, node_feature_path="", dim=8,
                                       seed_num=10):
    pos_edge_num = pos_edge.shape[1]
    neg_edge_num = neg_edge.shape[1]
    pos_acc_list = []
    neg_acc_list = []
    all_acc_list = []
    for i in range(seed_num):
        model, x = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, node_feature_path, dim=dim, seed=i)
        pos_prediction, neg_prediction = predict_downstream_result(model, x, pos_edge, neg_edge)
        pos_acc = sum(pos_prediction) / pos_edge_num
        neg_acc = sum(neg_prediction) / neg_edge_num
        all_acc = (sum(pos_prediction) + sum(neg_prediction)) / (pos_edge_num + neg_edge_num)
        pos_acc_list.append(pos_acc)
        neg_acc_list.append(neg_acc)
        all_acc_list.append(all_acc)
    pos_acc_mean = np.mean(np.array(pos_acc_list))
    pos_acc_std = np.std(np.array(pos_acc_list))
    neg_acc_mean = np.mean(np.array(neg_acc_list))
    neg_acc_std = np.std(np.array(neg_acc_list))
    all_acc_mean = np.mean(np.array(all_acc_list))
    all_acc_std = np.std(np.array(all_acc_list))
    return pos_acc_mean, pos_acc_std, neg_acc_mean, neg_acc_std, all_acc_mean, all_acc_std


def main():
    # embedding1 = torch.load("embeddings/256/0.pt")
    # embedding2 = torch.load("embeddings/256/1.pt")
    # knn_jaccard_similarity(embedding1, embedding2)
    embedding1 = torch.rand((2000000, 32))
    embedding2 = torch.rand((2000000, 32))
    print(calculate_knn_indices(embedding1))


if __name__ == '__main__':
    main()