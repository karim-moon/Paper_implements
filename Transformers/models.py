import torch
from torch import nn


class ScaledDotProduct(nn.Module):
    """
    torch Model Snippet
    """

    def __init__(self):
        super(ScaledDotProduct, self).__init__()
        self.scale_attn_table = nn.Softmax()

    def forward(self, query, key, value, mask, d_k):
        attention_table = ((query.matmul(key.transpose(1, 2)) / torch.sqrt(d_k)) * mask)
        attention_score = self.scale_attn_table(attention_table)
        attention_out = attention_score.matmul(value)

        return attention_out


class MultiHeadAttention(nn.Module):
    """
    torch Model Snippet
    """

    def __init__(self, in_dim, out_dim, head_num):
        super(MultiHeadAttention, self).__init__()
        self.query = nn.Linear(in_dim, head_num * out_dim)
        self.key = nn.Linear(in_dim, head_num * out_dim)
        self.value = nn.Linear(in_dim, head_num * out_dim)
        self.scaled_dot_product = ScaledDotProduct()
        self.word_dim_projection = nn.Linear(head_num * out_dim, in_dim)
        self.layer_norm = nn.LayerNorm()

        self.head_num = head_num

    def forward(self, q, k, v, mask):
        residual = q
        batch, q_len, word_dim = q.size()

        query = self.query(q).view(batch, q_len, self.head_num, -1).transpose(1, 2)  # [batch, head, seq_len, word_dim]
        key = self.key(k).view(batch, q_len, self.head_num, -1).transpose(1, 2)  # [batch, head, seq_len, word_dim]
        value = self.value(v).view(batch, q_len, self.head_num, -1).transpose(1, 2)  # [batch, head, seq_len, word_dim]

        mask = mask.repeat(1, self.head_num, 1, 1)
        attention_out = self.scaled_dot_product(query, key, value, mask)
        output = self.word_dim_projection(attention_out.transpose(1, 2).view(batch, q_len, -1))

        return self.layer_norm(residual + output)

    # TDDO: implement EncoderLayer
