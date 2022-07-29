import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PositionWiseFFNN(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout=0.5):
        super(PositionWiseFFNN, self).__init__()

        self.layer1 = nn.Linear(hidden_dim, ff_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ff_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)

        return x