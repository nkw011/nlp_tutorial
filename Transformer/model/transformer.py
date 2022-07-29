import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, encoder_layers, src_vocab_size, src_max_len,
                 decoder_layers, tgt_vocab_size, tgt_max_len,
                 src_pad_idx, tgt_pad_idx,
                 n_heads, hidden_dim, ff_dim, device, dropout=0.5
                 ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        self.encoder = Encoder(encoder_layers, src_vocab_size, src_max_len, n_heads, hidden_dim, ff_dim, device,
                               dropout)
        self.decoder = Decoder(decoder_layers, tgt_vocab_size, tgt_max_len, n_heads, hidden_dim, ff_dim, device,
                               dropout)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_mask1 = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        tgt_len = tgt.size()[1]
        tgt_mask2 = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()

        tgt_mask = tgt_mask1 & tgt_mask2

        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_src, _ = self.encoder(src, src_mask)

        output, attention = self.decoder(tgt, enc_src, tgt_mask, src_mask)

        return output, attention