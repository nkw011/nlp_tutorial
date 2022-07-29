import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_dim, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, hidden_dim).to(device)
        pos = torch.arange(max_len).view(-1,1)
        index = torch.arange(0, hidden_dim, step=2)

        self.encoding[:,::2] = torch.sin(pos/10000**(index/hidden_dim)).to(device)
        self.encoding[:,1::2] = torch.cos(pos/10000**(index/hidden_dim)).to(device)

    def forward(self, x):
        seq_len = x.shape[1] # (batch_size, seq_len, hidden_dim)
        return x + self.encoding[:seq_len,:]