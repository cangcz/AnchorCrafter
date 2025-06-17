import os
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock


class ObjAttnNet(nn.Module):
    def __init__(self, inner_dim=1024, num_heads=32, out_dim=1024, emb_size=1370):
        super().__init__()
        self.emb_size = emb_size
        self.proj_out = torch.nn.Linear(inner_dim, out_dim)
        self.norm = torch.nn.LayerNorm(out_dim)

        self.seq_reduce = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

        self.space_attn = BasicTransformerBlock(
            dim=inner_dim,
            num_attention_heads=num_heads,
            attention_head_dim=inner_dim//num_heads
        )
        self.space_attn2 = BasicTransformerBlock(
            dim=inner_dim,
            num_attention_heads=num_heads,
            attention_head_dim=inner_dim // num_heads
        )

    def forward(self, embeds):  # [b, n, c]
        embeds = self.space_attn(embeds)
        embeds = embeds[:, self.emb_size:self.emb_size*2, :]  # [b, 1370, c]

        embeds = self.space_attn2(embeds)
        embeds = self.seq_reduce(embeds.transpose(1, 2)).transpose(1, 2)
        
        embeds = self.proj_out(embeds)
        embeds = self.norm(embeds)
        return embeds

   