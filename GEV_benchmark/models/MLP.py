import torch.nn as nn
import torch

class MLP_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_block, self).__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128, out_dim)
        )
        self.sm = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('before', x)
        x = self.mlp(x)
        # print('after', x)
        score = self.sigmoid(x)
        return score