import torch.nn as nn
import torch
from torch import Tensor
from helpers import freeze_params, MultiHeadAttention, PoswiseFeedForwardNet, get_attn_pad_mask, Config
from transformer_layers import PositionalEncoding
from words import get_sinusoid_encoding_table
from constants import config

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = MultiHeadAttention(self.config,n_head=4, d_head=64)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq + 1, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0),
                                                                                                  inputs.size(
                                                                                                      1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)

        # (bs, n_enc_seq, d_hidn)
        outputs = self.enc_emb(inputs) + self.pos_emb(positions)

        # (bs, n_enc_seq, n_enc_seq)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs
# class Encoder(nn.Module):
#     @property
#     def output_size(self):
#         return self._output_size
#
# class TransformerEncoder(Encoder):
#     def __init__(self,
#                  hidden_size : int = 512,
#                  ff_size : int = 2048,
#                  num_layers : int = 8,
#                  num_heads : int = 4,
#                  dropout : float = 0.1,
#                  emb_dropout: float = 0.1,
#                  freeze : bool = False,
#                  **kwargs):
#
#         super(TransformerEncoder, self).__init__()
#
#         self.layers = nn.ModuleList([
#             TransformerEncoder(size=hidden_size, ff_size=ff_size,
#                                num_heads=num_heads, dropout=dropout)
#             for _ in range(num_layers)
#         ])
#
#         self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
#         self.pe = PositionalEncoding(hidden_size)
#         self.emb_dropout = nn.Dropout(dropout)
#         self._output_size = hidden_size
#         if freeze:
#             freeze_params(self)
#
#     def forward(self,
#                 embed_src : Tensor,
#                 src_length : Tensor,
#                 mask : Tensor)  ->(Tensor, Tensor):
#
#         x = embed_src
#         x = self.pe(x)
#         x = self.emb_dropout(x)
#
#         for layer in self.layers:
#             x = layer(x, mask)
#
#     def __repr__(self):
#         return "%s(num_layers=%r, num_heads=%r)" % (
#             self.__class__.__name__, len(self.layers),
#             self.layers[0].src_src_att.num_heads
#         )