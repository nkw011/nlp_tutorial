import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, n_heads, hidden_dim, ff_dim, dropout=0.5):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(n_heads, hidden_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        self.ffnn = PositionWiseFFNN(hidden_dim, ff_dim, dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src_, score = self.attention(src, src, src, src_mask)
        src = self.layer_norm1(src + self.dropout(src_))

        src_ = self.ffnn(src)
        src = self.layer_norm2(src + self.dropout(src_))

        return src, score