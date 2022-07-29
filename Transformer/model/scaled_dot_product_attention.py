import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = torch.tensor(Q.shape[-1])
        QK_T = torch.matmul(Q, K.transpose(-1,-2)) # (batch_size, n_heads, seq_len, seq_len)
        scaled = QK_T / torch.sqrt(d_k)
        if mask is not None:
            scaled.masked_fill_(mask==0, -1e14) # 0인 부분 masking 처리, 아주 작은 값을 주어 attention score를 작게 만든다.
        attention_distribution = F.softmax(scaled, dim=-1) # (batch_size, n_heads, seq_len, seq_len)
        output = torch.matmul(attention_distribution, V) # (batch_size, n_heads, seq_len, hidden_dim)

        return output, attention_distribution