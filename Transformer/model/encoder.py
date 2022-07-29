import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_layers, src_vocab_size, max_len, n_heads, hidden_dim, ff_dim, device,dropout=0.5):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(src_vocab_size, hidden_dim)
        self.encoding = PositionalEncoding(max_len, hidden_dim, device)

        self.layers = nn.ModuleList([EncoderBlock(n_heads, hidden_dim, ff_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src = self.embedding(src) # (batch_size, seq_len, hidden_dim)
        src = self.encoding(src) # (batch_size, seq_len, hidden_dim)

        for encoder_block in self.layers:
            src, score = encoder_block(src, src_mask)

        return src, score