"""Causal transformers with a three-class answer head and sinusoidal positions."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .representations import PAD, VOCAB_SIZE

SIZES = {
    "10k": (16, 3, 2), "50k": (32, 3, 4), "100k": (48, 3, 4), "500k": (96, 4, 4),
    "1m": (128, 5, 4), "3m": (192, 7, 6), "10m": (320, 8, 8),
    "5m": (256, 6, 8),
}

PRIMARY_SIZES = ("10k", "50k", "100k", "500k", "1m")
FF_WIDTH = {"50k": 176, "100k": 224, "5m": 1088}


def expected_parameter_count(size):
    width, layers, _ = SIZES[size]
    ff = FF_WIDTH.get(size,4*width)
    return layers*(4*width*width + 2*width*ff + 9*width + ff) + (VOCAB_SIZE+5)*width + 3


class Block(nn.Module):
    def __init__(self, width, heads, dropout, ff_width=None):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.out = nn.Linear(width, 3 * width), nn.Linear(width, width)
        ff_width = ff_width or 4 * width
        self.ff = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(), nn.Linear(ff_width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                   dropout_p=self.dropout.p if self.training else 0.0)
        x = x + self.dropout(self.out(attention.transpose(1, 2).reshape(batch, length, width)))
        return x + self.dropout(self.ff(self.norm2(x)))


class TinyTransformer(nn.Module):
    def __init__(self, size="100k", dropout=0.0, max_context=4096):
        super().__init__()
        width, layers, heads = SIZES[size]
        self.embedding = nn.Embedding(VOCAB_SIZE, width, padding_idx=PAD)
        self.blocks = nn.ModuleList(Block(width, heads, dropout, FF_WIDTH.get(size)) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 3)
        positions = torch.arange(max_context).unsqueeze(1).float()
        frequencies = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        encoding = torch.zeros(max_context, width)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(positions * frequencies), torch.cos(positions * frequencies)
        self.register_buffer("positions", encoding, persistent=False)

    def forward(self, tokens, lengths):
        if tokens.shape[1] > len(self.positions):
            raise ValueError("Context exceeded; truncation is forbidden")
        x = self.embedding(tokens) + self.positions[:tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        # Right padding cannot influence EOS because all attention is causal.
        final = self.norm(x)[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        return self.classifier(final)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
