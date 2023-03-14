import torch
import torch.nn as nn

class MLPPredictor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(embedding_dim, 1)
        self.sig = nn.Sigmoid()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, src_embedding, dst_embedding):
        x = torch.cat((src_embedding, dst_embedding), dim=1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.sig(x)
        return x