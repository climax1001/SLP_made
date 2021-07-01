import torch.nn as nn
import torch
from torch import Tensor
from helpers import freeze_params
from transformer_layers import PositionalEncoding


class Encoder(nn.Module):
    @property
    def output_size(self):
        return self._output_size

class TransformerEncoder(Encoder):
    def __init__(self,
                 hidden_size : int = 512,
                 ff_size : int = 2048,
                 num_layers : int = 8,
                 num_heads : int = 4,
                 dropout : float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze : bool = False,
                 **kwargs):

        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoder(size=hidden_size, ff_size=ff_size,
                               num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self._output_size = hidden_size
        if freeze:
            freeze_params(self)

    def forward(self,
                embed_src : Tensor,
                src_length : Tensor,
                mask : Tensor)  ->(Tensor, Tensor):

        x = embed_src
        x = self.pe(x)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads
        )