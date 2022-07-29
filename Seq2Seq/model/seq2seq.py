import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim, hidden_dim, device, num_layers=2, dropout=0.5):
        super(Seq2Seq, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = Encoder(src_vocab_size, emb_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, hidden_dim, num_layers, dropout)

    def forward(self, src, tgt):
        batch_size = src.size()[0]
        init_hidden, init_cell = torch.zeros((self.num_layers, batch_size, self.hidden_dim)), torch.zeros(
            (self.num_layers, batch_size, self.hidden_dim))
        init_hidden = init_hidden.to(self.device)
        init_cell = init_cell.to(self.device)

        src_hidden, src_cell = self.encoder(src, init_hidden, init_cell)
        # src_hidden = src_hidden.to(self.device)
        # src_cell = src_cell.to(self.device)
        out, _, _ = self.decoder(tgt, src_hidden, src_cell)  # (batch_size, seq_len, vocab_size)
        out = F.log_softmax(out, dim=-1)

        return out