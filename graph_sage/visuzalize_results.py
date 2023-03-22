import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def conver_data(data):
    data_up = data.copy()
    data_down = data.copy()
    data_up["mean"] = data_up["mean"] + data_up["std"]
    data_down["mean"] = data_down["mean"] - data_down["std"]
    data = pd.concat([data_up, data_down])
    return data
    

def draw_geometric_stability(src_file_path='./results/cora/cora_geometric_stability.csv'):
    data = pd.read_csv(src_file_path)

    # Procruste Similarity
    sns.set_theme(style='whitegrid')
    g = sns.barplot(data=data, x='embedding_dim', y='pro_sim', hue='model_name', ci='sd',
                hue_order=['Node2Vec', 'GraphSAGE'], palette="dark", alpha=.6)
    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
    plt.xlabel('Node Embedding Dimension')
    plt.ylabel('Procruste Similarity')
    plt.savefig('./visualization/cora_pro_sim.png', format='png', dpi=150)
    plt.close()

    # k-NN Jaccard Similarity
    sns.set_theme(style='whitegrid')
    g = sns.barplot(data=data, x='embedding_dim', y='knn_sim', hue='model_name', ci='sd',
                hue_order=['Node2Vec', 'GraphSAGE'], palette="dark", alpha=.6)
    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
    plt.xlabel('Node Embedding Dimension')
    plt.ylabel('k-NN Jaccard Similarity')
    plt.savefig('./visualization/cora_knn_sim.png', format='png', dpi=150)
    plt.close()

    # Second-Order Cosine Similarity
    sns.set_theme(style='whitegrid')
    g = sns.barplot(data=data, x='embedding_dim', y='sec_sim', hue='model_name', ci='sd',
                hue_order=['Node2Vec', 'GraphSAGE'], palette="dark", alpha=.6)
    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
    plt.xlabel('Node Embedding Dimension')
    plt.ylabel('Second-Order Cosine Similarity')
    plt.savefig('./visualization/cora_sec_sim.png', format='png', dpi=150)
    plt.close()

def draw_link_prediction_stability(yticks, src_file_path='./results/cora/cora_link_prediction_overlap_pair.csv', des_file_path='./visualization/cora/cora_link_prediction_overlap_pair.jpg', y_label='Link Prediction Overlap Rate for Seed Pairs'):
    data = pd.read_csv(src_file_path)

    df_Node2Vec_0 = data[data['model_name'] == 'Node2Vec_0'].iloc[:, 1:]
    df_Node2Vec_1 = data[data['model_name'] == 'Node2Vec_1'].iloc[:, 1:]
    df_GraphSAGE = data[data['model_name'] == 'GraphSAGE'].iloc[:, 1:]

    fig, axs = plt.subplots(ncols=3, figsize=(18, 6))

    df_melt_Node2Vec_0 = pd.melt(df_Node2Vec_0, id_vars=['embedding_dim'], var_name='Overlap Rate', value_name='Value')
    sns.pointplot(x='embedding_dim', y='Value', hue='Overlap Rate', data=df_melt_Node2Vec_0, ax=axs[0])
    axs[0].set_title('Node2Vec without Node Features')
    axs[0].set_yticks(yticks)
    axs[0].set_xlabel('Node Embedding Dimension')
    axs[0].set_ylabel(y_label)

    df_melt_Node2Vec_1 = pd.melt(df_Node2Vec_1, id_vars=['embedding_dim'], var_name='Overlap Rate', value_name='Value')
    sns.pointplot(x='embedding_dim', y='Value', hue='Overlap Rate', data=df_melt_Node2Vec_1, ax=axs[1])
    axs[1].set_title('Node2Vec with Node Features')
    axs[1].set_yticks(yticks)
    axs[1].set_xlabel('Node Embedding Dimension')
    axs[1].set_ylabel(y_label)

    df_melt_GraphSAGE = pd.melt(df_GraphSAGE, id_vars=['embedding_dim'], var_name='Overlap Rate', value_name='Value')
    sns.pointplot(x='embedding_dim', y='Value', hue='Overlap Rate', data=df_melt_GraphSAGE, ax=axs[2])
    axs[2].set_title('GraphSAGE')
    axs[2].set_yticks(yticks)
    axs[2].set_xlabel('Node Embedding Dimension')
    axs[2].set_ylabel(y_label)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.subplots_adjust(bottom=0.15)

    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[2].get_legend().remove()

    plt.savefig(des_file_path, format='png', dpi=150)
    plt.close()

    return

def draw_accuracy(yticks, src_file_path='./results/citation2/citation2_accuracy.csv', des_file_path='./visualization/citation2/citation2_accuracy.png', y_label='Accuracy on Test Set'):
    data = pd.read_csv(src_file_path)

    df_Node2Vec_0 = conver_data(data[data['model_name'] == 'Node2Vec_0'].iloc[:, 1:])
    df_Node2Vec_1 = conver_data(data[data['model_name'] == 'Node2Vec_1'].iloc[:, 1:])
    df_GraphSAGE = conver_data(data[data['model_name'] == 'GraphSAGE'].iloc[:, 1:])

    fig, axs = plt.subplots(ncols=3, figsize=(18, 6))

    sns.pointplot(x='embedding_dim', y='mean', hue='type', data=df_Node2Vec_0, ax=axs[0], alpha=.6)
    axs[0].set_title('Node2Vec without Node Features')
    axs[0].set_yticks(yticks)
    axs[0].set_xlabel('Node Embedding Dimension')
    axs[0].set_ylabel(y_label)

    sns.pointplot(x='embedding_dim', y='mean', hue='type', data=df_Node2Vec_1, ax=axs[1])
    axs[1].set_title('Node2Vec with Node Features')
    axs[1].set_yticks(yticks)
    axs[1].set_xlabel('Node Embedding Dimension')
    axs[1].set_ylabel(y_label)

    sns.pointplot(x='embedding_dim', y='mean', hue='type', data=df_GraphSAGE, ax=axs[2])
    axs[2].set_title('GraphSAGE')
    axs[2].set_yticks(yticks)
    axs[2].set_xlabel('Node Embedding Dimension')
    axs[2].set_ylabel(y_label)

    handles, labels = axs[0].get_legend_handles_labels()
    labels = ['Positive Predictions', 'Negative Predictions', 'All Predictions']
    fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.subplots_adjust(bottom=0.15)

    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[2].get_legend().remove()

    plt.savefig(des_file_path, format='png', dpi=150)
    plt.close()
    return


def draw_geometric_stability_for_citation2(model_name='Node2Vec', src_file_path='./results/citation2/citation2_geometric_stability.csv'):
    data = pd.read_csv(src_file_path)

    data_df = data[data['model_name'] == model_name].iloc[:, 1:]

    fig, axs = plt.subplots(ncols=3, figsize=(18, 6))

    sns.set_theme(style='whitegrid')

    # Procruste Similarity
    sns.pointplot(data=data_df, x='num_node', y='pro_sim', hue='embedding_dim', ax=axs[0], ci=None)
    yticks = list(range(1, 11, 1))
    yticks = [i/10 for i in yticks]
    # axs[0].set_yticks(yticks)
    axs[0].set_xlabel('the Number of Nodes')
    axs[0].set_ylabel('Procruste Similarity')

    # k-NN Jaccard Similarity
    sns.pointplot(data=data_df, x='num_node', y='knn_sim', hue='embedding_dim', ax=axs[1], ci=None)
    yticks = list(range(1, 11, 1))
    yticks = [i/10 for i in yticks]
    # axs[1].set_yticks(yticks)
    axs[1].set_xlabel('the Number of Nodes')
    axs[1].set_ylabel('k-NN Jaccard Similarity')

    # Second-Order Cosine Similarity
    sns.pointplot(data=data_df, x='num_node', y='sec_sim', hue='embedding_dim', ax=axs[2], ci=None)
    yticks = list(range(1, 11, 1))
    yticks = [i/10 for i in yticks]
    # axs[2].set_yticks(yticks)
    axs[2].set_xlabel('the Number of Nodes')
    axs[2].set_ylabel('Second-Order Cosine Similarity')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6)
    plt.subplots_adjust(bottom=0.15)

    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[2].get_legend().remove()

    des_file_path = f'./visualization/citation2/citation2_{model_name}_geometric_stability.png'


    plt.savefig(des_file_path, format='png', dpi=150)
    plt.close()
    return




if __name__ == '__main__':
    # # Cora
    # draw_geometric_stability()
    # ytricks = list(range(7, 11, 1))
    # ytricks = [i/10 for i in ytricks]
    # draw_link_prediction_stability(ytricks)
    # ytricks = list(range(4, 11, 1))
    # ytricks = [i/10 for i in ytricks]
    # draw_link_prediction_stability(ytricks, './results/cora/cora_link_prediction_overlap_pair_all_true.csv', './visualization/cora/cora_link_prediction_overlap_pair_all_true.png', 'Link Prediction Correct Overlap Rate for Seed Pairs')
    # ytricks = list(range(2, 11, 1))
    # ytricks = [i/10 for i in ytricks]
    # draw_link_prediction_stability(ytricks, './results/cora/cora_link_prediction_overlap_all_seeds.csv', './visualization/cora/cora_link_prediction_overlap_all_seeds.png', 'Link Prediction Overlap Rate for All Seeds')
    # ytricks = list(range(5, 11, 1))
    # ytricks = [i/10 for i in ytricks]
    # draw_accuracy(ytricks, src_file_path='./results/cora/cora_accuracy.csv', des_file_path='./visualization/cora/cora_accuracy.png')
    

    # Citation2
    ytricks = list(range(8, 11, 1))
    ytricks = [i/10 for i in ytricks]
    draw_link_prediction_stability(ytricks, './results/citation2/citation2_link_prediction_overlap_pair.csv', './visualization/citation2/citation2_link_prediction_overlap_pair.png')
    ytricks = list(range(8, 11, 1))
    ytricks = [i/10 for i in ytricks]
    draw_link_prediction_stability(ytricks, './results/citation2/citation2_link_prediction_overlap_pair_all_true.csv', './visualization/citation2/citation2_link_prediction_overlap_pair_all_true.png', 'Link Prediction Correct Overlap Rate for Seed Pairs')
    ytricks = list(range(7, 11, 1))
    ytricks = [i/10 for i in ytricks]
    draw_link_prediction_stability(ytricks, './results/citation2/citation2_link_prediction_overlap_all_seeds.csv', './visualization/citation2/citation2_link_prediction_overlap_all_seeds.png', 'Link Prediction Overlap Rate for All Seeds')
    # draw_geometric_stability_for_citation2(model_name='Node2Vec')
    # draw_geometric_stability_for_citation2(model_name='GraphSAGE')

    # ytricks = list(range(88, 101, 3))
    # ytricks = [i/100 for i in ytricks]
    # draw_accuracy(ytricks)

    

