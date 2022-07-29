import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_dim, dropout=0.5):
        super(MultiHeadAttention, self).__init__()

        assert hidden_dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

        self.attention = ScaledDotProductAttention()

        self.W_O = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):

        batch_size = Q.shape[0]

        Q = self.W_Q(Q) # (batch_size, seq_len, hidden_dim)
        K = self.W_K(K) # (batch_size, seq_len, hidden_dim)
        V = self.W_V(V) # (batch_size, seq_len, hidden_dim)

        # Multi-Head
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1,2) # (batch_size, n_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1,2) # (batch_size, n_heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1,2) # (batch_size, n_heads, seq_len, head_dim)

        output, score = self.attention(Q,K,V, mask)

        output = self.dropout(output)

        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim) # (batch_size, seq_len, hidden_dim)
        output = self.W_O(output) # (batch_size, seq_len, input_dim)

        return output, score