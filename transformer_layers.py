# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
from torch import Tensor
from constants import TARGET_PAD
import math
class MultiheadedAttention(nn.Module):
    def __init__(self, num_heads : int, size : int, dropout : float = 0.1):
        super(MultiheadedAttention, self).__init__()

        assert size % num_heads == 0
        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.target_pad = TARGET_PAD

    def forward(self, k : Tensor, v : Tensor, q : Tensor, mask : Tensor = None, padding_mask : Tensor = None):
        batch_size = k.size(0) # torch의 길이, 행
        num_heads = self.num_heads

        k = self.k_layer(k) # size
        v = self.v_layer(v)
        q = self.q_layer(q)
        # shape [batch_size, num_heads, ... , ... ]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # scaled
        q = q / math.sqrt(self.head_size)

        # shape [batch_size, num_heads, query_len, key_len]
        scores = torch.matmul(q, k.transpose(2,3))

        # mask
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attention = self.softmax(scores)
        attention = self.dropout(attention)

        if padding_mask is not None:
            attention = attention.masked_fill(~padding_mask, 0.0)

        context = torch.matmul(attention, v)
        context = context.transpose(1,2).contiguous.view(
            batch_size, -1 , num_heads * self.head_size
        )
        output = self.output_layer(context)
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size,ff_size, drop_out=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(ff_size, input_size),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x

class PositionalEncoding(nn.Module):
    def __init__(self,
                 size : int = 0,
                 max_len : int = 20000,
                 mask_count = False):
        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))

        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0,size,dtype=torch.float) *
                              -(math.log(10000.0) / size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 0::1] = torch.cos(position.float() * div_term)

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = size
        self.mask_count = mask_count

    def forward(self,emb):
        return emb + self.pe[:, :emb.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 size : int = 0,
                 ff_size : int = 0,
                 num_heads : int = 0,
                 dropout : float = 0.1):

        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiheadedAttention(num_heads, size, dropout=dropout)
        self.feedfoward = PositionWiseFeedForward(size, ff_size=ff_size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x : Tensor, mask : Tensor) -> Tensor:
        x_norm = self.layer_norm(x)

        h = self.src_src_att(x_norm, x_norm, x_norm, mask=mask) #?

        h = self.dropout(h) + x
        o = self.feedfoward(h)
        return o

class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1,
                 decoder_trg_trg: bool = True):

        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiheadedAttention(num_heads, size,
                                                dropout=dropout)

        self.src_trg_att = MultiheadedAttention(num_heads, size,
                                                dropout=dropout)

        self.feed_forward = PositionWiseFeedForward(size, ff_size=ff_size)

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

        self.decoder_trg_trg = decoder_trg_trg

    # pylint: disable=arguments-differ
    def forward(self,
                x: Tensor = None,
                memory: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                padding_mask: Tensor = None) -> Tensor:

        # decoder/target self-attention
        h1 = self.x_layer_norm(x)

        # Target-Target Self Attention
        if self.decoder_trg_trg:
            h1 = self.trg_trg_att(h1, h1, h1, mask=trg_mask, padding_mask=padding_mask)
        h1 = self.dropout(h1) + x

        # Source-Target Self Attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o
