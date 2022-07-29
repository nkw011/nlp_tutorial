import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, n_layers, tgt_vocab_size, max_len, n_heads, hidden_dim, ff_dim, device, dropout=0.5):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(tgt_vocab_size, hidden_dim)
        self.encoding = PositionalEncoding(max_len, hidden_dim, device)

        self.layers = nn.ModuleList([ DecoderBlock(n_heads, hidden_dim, ff_dim, dropout) for _ in range(n_layers)])

        self.output = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, tgt, src, tgt_mask, src_mask):
        tgt = self.embedding(tgt) # (batch_size, seq_len, hidden_dim)
        tgt = self.encoding(tgt)

        for decoder_block in self.layers:
            tgt, score = decoder_block(src, tgt, src_mask, tgt_mask)

        output = self.output(tgt)

        return output, score