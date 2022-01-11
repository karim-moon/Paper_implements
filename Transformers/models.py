import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, in_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, in_dim)

    def __forward__(self, word_vector):
        return self.embedding(word_vector)


class ScaledDotProduct(nn.Module):
    """
    torch Model Snippet
    """

    def __init__(self):
        super(ScaledDotProduct, self).__init__()
        self.scale_attn_table = nn.Softmax()

    def forward(self, query, key, value, mask, d_k):
        if mask == None:
            attention_table = ((query.matmul(key.transpose(1, 2)) / torch.sqrt(d_k)))
        else:
            attention_table = ((query.matmul(key.transpose(1, 2)) / torch.sqrt(d_k)) * mask)

        attention_score = self.scale_attn_table(attention_table)
        attention_out = attention_score.matmul(value)

        return attention_out


class MultiHeadAttention(nn.Module):
    """
    torch Model Snippet
    """

    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.query = nn.Linear(in_dim, num_heads * out_dim)
        self.key = nn.Linear(in_dim, num_heads * out_dim)
        self.value = nn.Linear(in_dim, num_heads * out_dim)
        self.scaled_dot_product = ScaledDotProduct()
        self.word_dim_projection = nn.Linear(num_heads * out_dim, in_dim)
        self.layer_norm = nn.LayerNorm(in_dim)

        self.num_heads = num_heads

    def forward(self, sequence, mask=None):
        residual = sequence
        batch, seq_len, word_dim = sequence.size()

        query = self.query(sequence).view(batch,
                                          seq_len,
                                          self.num_heads, -1).transpose(1, 2)  # [batch, num_heads, seq_len, word_dim]
        key = self.key(sequence).view(batch,
                                      seq_len,
                                      self.num_heads, -1).transpose(1, 2)  # [batch, num_heads, seq_len, word_dim]
        value = self.value(sequence).view(batch,
                                          seq_len,
                                          self.num_heads, -1).transpose(1, 2)  # [batch, num_heads, seq_len, word_dim]
        if mask:
            mask = mask.repeat(1, self.head_num, 1, 1)
        attention_out = self.scaled_dot_product(query, key, value, mask)
        output = self.word_dim_projection(attention_out.transpose(1, 2).view(batch, seq_len, -1))

        return self.layer_norm(residual + output)


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        self.forward_net = nn.Linear(in_dim, out_dim)
        self.recall_net = nn.Linear(out_dim, in_dim)
        self.dropout = nn.Dropout(0.5)

    def __forward__(self, sequence):
        forward = self.forward_net(sequence)
        recall = self.dropout(self.recall_net(forward))
        return recall


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(in_dim, out_dim, num_heads)
        self.feed_forward = FeedForward(in_dim, out_dim)

    def __forward__(self, embedding_vector):
        after_attention_vector = self.attention(embedding_vector)
        feed_forward_vector = self.feed_forward(after_attention_vector)
        return feed_forward_vector


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(Decoder, self).__init__()
        self.self_attention = MultiHeadAttention(in_dim, out_dim, num_heads)
        self.self_forward = FeedForward(in_dim, out_dim)

        self.encoder_decoder_attention = MultiHeadAttention(in_dim, out_dim, num_heads)
        self.encoder_decoder_forward = FeedForward(in_dim, out_dim)

    def __forward__(self, encoder_output):
        batch_size, seq_len, in_dim = encoder_output.size()
        self_attention_output = self.self_output(encoder_output)
        feed_forward_output = self.self_forward(self_attention_output)

        mask = torch.ones(batch_size, seq_len, seq_len).tril()
        encoder_decoder_att_output = self.encoder_decoder_attention(feed_forward_output, mask)
        feed_forward_output = self.encoder_decoder_forward(encoder_decoder_att_output)



