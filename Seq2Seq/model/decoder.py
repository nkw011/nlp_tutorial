import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, hidden, cell):
        tgt = self.embedding(tgt)
        tgt, (next_hidden, next_cell) = self.lstm(tgt, (hidden, cell))
        out = self.output(tgt)

        return out, next_hidden, next_cell