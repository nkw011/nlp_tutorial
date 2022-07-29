import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, n_heads, hidden_dim, ff_dim, dropout=0.5):
        super(DecoderBlock, self).__init__()

        self.masked_attention = MultiHeadAttention(n_heads, hidden_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        self.attention = MultiHeadAttention(n_heads, hidden_dim, dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.ffnn = PositionWiseFFNN(hidden_dim, ff_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):

        tgt_, _ = self.masked_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.layer_norm1(tgt + self.dropout(tgt_))

        tgt_, score = self.attention(tgt, src, src, src_mask)
        tgt = self.layer_norm2(tgt + self.dropout(tgt_))

        tgt_ = self.ffnn(tgt)
        tgt = self.layer_norm3(tgt + self.dropout(tgt_))

        return tgt, score