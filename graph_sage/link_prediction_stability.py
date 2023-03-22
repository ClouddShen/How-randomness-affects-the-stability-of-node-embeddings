import os
import torch
from predictor import MLPPredictor
import numpy as np
from citation2_baseline import LinkPredictor
import csv
import tqdm



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

def load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, dim=8, seed=10, data_name='cora'):
    model_param_path = os.path.join(model_src_dir, str(dim), f'MLP_embed_{str(dim)}_seed_{str(seed)}.pt')
    embedding_path = os.path.join(embedding_src_dir, str(dim), f'{str(seed)}.npy')
    x = torch.Tensor(np.load(embedding_path))
    x = x.to(device)

    if data_name == 'cora':
        model = MLPPredictor(x.size(-1)).to(device)
    elif data_name == 'citation2':
        model = LinkPredictor(x.size(-1), x.size(-1), 1, 3, 0).to(device)
    # print(model)
    model.load_state_dict(torch.load(model_param_path))
    return model, x
    

def calculate_one_dim_downstream_result_similarity(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim=8, seed_num=10, data_name='cora'):
    pos_edge_num = pos_edge.shape[1]
    neg_edge_num = neg_edge.shape[1]
    pos_overlap_num_list = []
    neg_overlap_num_list = []
    for i in range(seed_num):
        for j in range(i + 1, seed_num):
            model1, x1 = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, dim=dim, seed=i, data_name=data_name)
            model2, x2 = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, dim=dim, seed=j, data_name=data_name)
            pos_overlap_num, neg_overlap_num = calculate_two_model_downstream_result_similarity(model1, model2, x1, x2, pos_edge, neg_edge)
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


def calcaulate_overlap_for_all_seeds(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim=8, seed_num=10, data_name='cora'):
    pos_edge_num = pos_edge.shape[1]
    neg_edge_num = neg_edge.shape[1]
    for i in range(seed_num):
        model, x = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, dim=dim, seed=i, data_name=data_name)
        pos_prediction, neg_prediction = predict_downstream_result(model, x, pos_edge, neg_edge)
        pos_prediction = pos_prediction.numpy().astype(np.int32)
        neg_prediction = neg_prediction.numpy().astype(np.int32)
        if i == 0:
            pos_pred_overlap = pos_prediction.copy()
            neg_pred_overlap = neg_prediction.copy()
        else:
            for j in range(pos_edge_num):
                if pos_pred_overlap[j] == -1:
                    continue
                if pos_prediction[j] != pos_pred_overlap[j]:
                    pos_pred_overlap[j] = -1
            for j in range(neg_edge_num):
                if neg_pred_overlap[j] == -1:
                    continue
                if neg_prediction[j] != neg_pred_overlap[j]:
                    neg_pred_overlap[j] = -1
    # print(pos_pred_overlap)
    pos_num = sum(pos_pred_overlap != -1)
    # print(pos_num)
    neg_num = sum(neg_pred_overlap != -1)
    return pos_num/pos_edge_num, neg_num/neg_edge_num, (pos_num+neg_num)/(pos_edge_num+neg_edge_num)

def calculate_two_model_downstream_result_similarity_all_true(model1, model2, x1, x2, pos_edge, neg_edge):
    pos_pred1, neg_pred1 = predict_downstream_result(model1, x1, pos_edge, neg_edge)
    pos_pred2, neg_pred2 = predict_downstream_result(model2, x2, pos_edge, neg_edge)
    pos_overlap_num = sum(pos_pred1 & pos_pred2)
    neg_overlap_num = sum(neg_pred1 & neg_pred2)
    return pos_overlap_num, neg_overlap_num

def calculate_one_dim_downstream_result_similarity_all_true(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim=8, seed_num=10, data_name='cora'):
    pos_edge_num = pos_edge.shape[1]
    neg_edge_num = neg_edge.shape[1]
    pos_overlap_num_list = []
    neg_overlap_num_list = []
    for i in range(seed_num):
        for j in range(i + 1, seed_num):
            model1, x1 = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, dim=dim, seed=i, data_name=data_name)
            model2, x2 = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, dim=dim, seed=j, data_name=data_name)
            pos_overlap_num, neg_overlap_num = calculate_two_model_downstream_result_similarity_all_true(model1, model2, x1, x2, pos_edge, neg_edge)
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

def compute_and_save_link_prediction_for_cora(model_name='GraphSAGE', model_src_dir = './cora/model', embedding_src_dir = './cora/embedding_data'):
    device = torch.device('cpu')
    pos_edge = torch.from_numpy(np.load('./cora/data/test_pos_edges.npy'))
    neg_edge = torch.from_numpy(np.load('./cora/data/test_neg_edges.npy'))
    dim_list = [8, 16, 32, 64, 128, 256]
    num_seed = 10
    
    # file_path = './visualization/cora_link_prediction_overlap.csv'
    # with open(file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['model_name', 'embedding_dim', 'pos_overlap_rate', 'neg_overlap_rate', 'all_overlap_rate'])
    #     for dim in dim_list:
    #         print("embeeding dimenstion is: ", dim)
    #         pos_overlap_rate_pair, neg_overlap_rate_pair, all_overlap_rate_pair = calculate_one_dim_downstream_result_similarity(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim, num_seed)
    #         print(pos_overlap_rate_pair, neg_overlap_rate_pair, all_overlap_rate_pair)
    #         writer.writerow([model_name, dim, pos_overlap_rate_pair, neg_overlap_rate_pair, all_overlap_rate_pair])

    # file_path = './visualization/cora_link_prediction_overlap_all_true.csv'
    # with open(file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['model_name', 'embedding_dim', 'pos_overlap_rate', 'neg_overlap_rate', 'all_overlap_rate'])
    #     for dim in dim_list:
    #         print("embeeding dimenstion is: ", dim)
    #         pos_overlap_rate_pair_all_true, neg_overlap_rate_pair_all_true, all_overlap_rate_pair_all_true = calculate_one_dim_downstream_result_similarity_all_true(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim, num_seed)
    #         print(pos_overlap_rate_pair_all_true, neg_overlap_rate_pair_all_true, all_overlap_rate_pair_all_true)
    #         writer.writerow([model_name, dim, pos_overlap_rate_pair_all_true, neg_overlap_rate_pair_all_true, all_overlap_rate_pair_all_true])       
            
    file_path = './visualization/cora_link_prediction_overlap_all_seeds.csv'
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'embedding_dim', 'pos_overlap_rate', 'neg_overlap_rate', 'all_overlap_rate'])
        for dim in dim_list:
            print("embeeding dimenstion is: ", dim)
            pos_overlap_rate_all, neg_overlap_rate_all, all_overlap_rate_all = calcaulate_overlap_for_all_seeds(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim, num_seed)
            print(pos_overlap_rate_all, neg_overlap_rate_all, all_overlap_rate_all)
            writer.writerow([model_name, dim, pos_overlap_rate_all, neg_overlap_rate_all, all_overlap_rate_all])       
    
def compute_and_save_link_prediction_for_citation2(model_name='GraphSAGE', model_src_dir = './citation2/model', embedding_src_dir = './citation2/embedding_data'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_edge = torch.from_numpy(np.load('./citation2/data/test_pos_edge.npy'))
    neg_edge = torch.from_numpy(np.load('./citation2/data/test_neg_edge.npy'))
    dim_list = [8, 16, 32, 64, 128, 256]
    num_seed = 5
    
    file_path = './visualization/citation2_link_prediction_overlap_pair.csv'
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'embedding_dim', 'pos_overlap_rate', 'neg_overlap_rate', 'all_overlap_rate'])
        for dim in dim_list:
            print("embeeding dimenstion is: ", dim)
            pos_overlap_rate_pair, neg_overlap_rate_pair, all_overlap_rate_pair = calculate_one_dim_downstream_result_similarity(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim, num_seed, data_name='citation2')
            print(pos_overlap_rate_pair, neg_overlap_rate_pair, all_overlap_rate_pair)
            writer.writerow([model_name, dim, pos_overlap_rate_pair, neg_overlap_rate_pair, all_overlap_rate_pair])

    file_path = './visualization/citation2_link_prediction_overlap_pair_all_true.csv'
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'embedding_dim', 'pos_overlap_rate', 'neg_overlap_rate', 'all_overlap_rate'])
        for dim in dim_list:
            print("embeeding dimenstion is: ", dim)
            pos_overlap_rate_pair_all_true, neg_overlap_rate_pair_all_true, all_overlap_rate_pair_all_true = calculate_one_dim_downstream_result_similarity_all_true(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim, num_seed, data_name='citation2')
            print(pos_overlap_rate_pair_all_true, neg_overlap_rate_pair_all_true, all_overlap_rate_pair_all_true)
            writer.writerow([model_name, dim, pos_overlap_rate_pair_all_true, neg_overlap_rate_pair_all_true, all_overlap_rate_pair_all_true])       
            
    file_path = './visualization/citation2_link_prediction_overlap_all_seeds.csv'
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'embedding_dim', 'pos_overlap_rate', 'neg_overlap_rate', 'all_overlap_rate'])
        for dim in dim_list:
            print("embeeding dimenstion is: ", dim)
            pos_overlap_rate_all, neg_overlap_rate_all, all_overlap_rate_all = calcaulate_overlap_for_all_seeds(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim, num_seed, data_name='citation2')
            print(pos_overlap_rate_all, neg_overlap_rate_all, all_overlap_rate_all)
            writer.writerow([model_name, dim, pos_overlap_rate_all, neg_overlap_rate_all, all_overlap_rate_all])       
    return

def calculate_acc_mean_std_for_one_dim(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim=8, seed_num=10, data_name='cora'):
    pos_edge_num = pos_edge.shape[1]
    neg_edge_num = neg_edge.shape[1]
    pos_acc_list = []
    neg_acc_list = []
    all_acc_list = []
    for i in range(seed_num):
        model, x = load_model_param_and_load_embedding(model_src_dir, embedding_src_dir, device, dim=dim, seed=i, data_name=data_name)
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

def compute_and_save_accuracy_for_cora(model_name='GraphSAGE', model_src_dir = './cora/model', embedding_src_dir = './cora/embedding_data', file_path='./visualization/cora_accuracy.csv'):
    device = torch.device('cpu')
    pos_edge = torch.from_numpy(np.load('./cora/data/test_pos_edges.npy'))
    neg_edge = torch.from_numpy(np.load('./cora/data/test_neg_edges.npy'))
    dim_list = [8, 16, 32, 64, 128, 256]
    num_seed = 10

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'embedding_dim', 'pos_acc_mean', 'pos_acc_std', 'neg_acc_mean', 'neg_acc_std', 'all_acc_mean', 'all_acc_std'])
        for dim in dim_list:
            print("embeeding dimenstion is: ", dim)
            pos_acc_mean, pos_acc_std, neg_acc_mean, neg_acc_std, all_acc_mean, all_acc_std = calculate_acc_mean_std_for_one_dim(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim=dim, seed_num=num_seed, data_name='cora')
            writer.writerow([model_name, dim, pos_acc_mean, pos_acc_std, neg_acc_mean, neg_acc_std, all_acc_mean, all_acc_std])
    return

def compute_and_save_accuracy_for_citation2(model_name='GraphSAGE', model_src_dir = './citation2/model', embedding_src_dir = './citation2/embedding_data', file_path='./visualization/citation2_accuracy.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_edge = torch.from_numpy(np.load('./citation2/data/test_pos_edge.npy'))
    neg_edge = torch.from_numpy(np.load('./citation2/data/test_neg_edge.npy'))
    dim_list = [8, 16, 32, 64, 128, 256]
    num_seed = 5

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'embedding_dim', 'pos_acc_mean', 'pos_acc_std', 'neg_acc_mean', 'neg_acc_std', 'all_acc_mean', 'all_acc_std'])
        for dim in dim_list:
            print("embeeding dimenstion is: ", dim)
            pos_acc_mean, pos_acc_std, neg_acc_mean, neg_acc_std, all_acc_mean, all_acc_std = calculate_acc_mean_std_for_one_dim(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim=dim, seed_num=num_seed, data_name='citation2')
            writer.writerow([model_name, dim, pos_acc_mean, pos_acc_std, neg_acc_mean, neg_acc_std, all_acc_mean, all_acc_std])
    return




if __name__ == '__main__':
    # print("Compute and Save accuracy for cora...")
    # compute_and_save_accuracy_for_cora()

    # print("Compute link prediction stability for Cora...")
    # compute_and_save_link_prediction_for_cora()

    # print("Compute and Save accuracy for citation2...")
    # compute_and_save_accuracy_for_citation2()

    print("Compute link prediction stability for citation2...")
    compute_and_save_link_prediction_for_citation2()
        

    # print("Compute link prediction stability for Citation2...")
    # model_src_dir = './citation2/model'
    # embedding_src_dir = './citation2/embedding_data'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pos_edge = torch.from_numpy(np.load('./citation2/data/test_pos_edge.npy'))
    # neg_edge = torch.from_numpy(np.load('./citation2/data/test_neg_edge.npy'))
    # dim = 16
    # num_seed = 5
    # pos_overlap_rate_pair, neg_overlap_rate_pair, all_overlap_rate_pair = calculate_one_dim_downstream_result_similarity(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim, num_seed, data_name='citation2')
    # print(pos_overlap_rate_pair, neg_overlap_rate_pair, all_overlap_rate_pair)
    # pos_overlap_rate_all, neg_overlap_rate_all, all_overlap_rate_all = calcaulate_overlap_for_all_seeds(model_src_dir, embedding_src_dir, pos_edge, neg_edge, device, dim, num_seed, data_name='citation2')
    # print(pos_overlap_rate_all, neg_overlap_rate_all, all_overlap_rate_all)


