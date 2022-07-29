import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

    def forward(self, src, hidden, cell):
        src = self.embedding(src)
        src, (next_hidden, next_cell) = self.lstm(src, (hidden, cell))

        return next_hidden, next_cell