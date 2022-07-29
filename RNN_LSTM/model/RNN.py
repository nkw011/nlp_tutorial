import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MyRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(MyRNN, self).__init__()
        self.model_type = 'RNN'

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, nonlinearity='relu',batch_first=True, dropout=dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def forward(self, x, h):
        '''
        x: (batch_size, seq_len)
        h: (num_layers, batch_size, hidden_dim)
        '''

        x = self.embedding(x) # (batch_size, seq_len, emb_size)
        out, h_n = self.rnn(x, h) # (batch_size, seq_len, hidden_dim), (num_layer, batch_size, hidden_dim)
        out = F.log_softmax(self.output(out),dim=-1) # (batch_size, seq_len, vocab_size)
        return out, h_n

    def init_weights(self):
        k = torch.tensor(1/self.hidden_dim)
        for param in self.parameters():
            nn.init.uniform_(param.data, -torch.sqrt(k), torch.sqrt(k))

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_dim))