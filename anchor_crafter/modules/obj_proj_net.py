import os
from pathlib import Path

import torch
import torch.nn as nn


class ObjProjNet(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=3072,
                 clip_extra_context_tokens=4, inner_dim=1024):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj_in = torch.nn.Linear(clip_embeddings_dim, inner_dim)
        self.proj_out = torch.nn.Linear(inner_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, embeds):
        embeds = self.proj_in(embeds)
        clip_extra_context_tokens = self.proj_out(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

