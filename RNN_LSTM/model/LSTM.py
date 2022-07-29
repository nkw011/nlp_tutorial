import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MyLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(MyLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.model_type = 'LSTM'

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self.init_param()

    def init_param(self):
        k = torch.tensor(1 / self.hidden_dim)
        for param in self.parameters():
            nn.init.uniform_(param.data, -torch.sqrt(k), torch.sqrt(k))

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_dim)), torch.zeros(
            (self.num_layers, batch_size, self.hidden_dim))

    def forward(self, x, hidden):
        x = self.embedding(x)

        out, (next_h, next_c) = self.lstm(x, hidden)
        out = self.output(out)
        log_prob = F.log_softmax(out, dim=-1)

        return log_prob, (next_h, next_c)