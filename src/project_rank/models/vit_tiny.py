from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from project_rank.vit_spec import ViTSpec


class Attention(nn.Module):
    """Explicit projections so parameter transfer and MAC hooks remain transparent."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, d // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(out.transpose(1, 2).reshape(b, t, d))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class ViTTinyCIFAR(nn.Module):
    """Pre-norm ViT with independently configurable MLP widths; no pretrained weights."""

    def __init__(
        self, inner_widths: Sequence[int] | None = None, spec: ViTSpec = ViTSpec()
    ) -> None:
        super().__init__()
        self.spec = spec
        self.inner_widths = spec.widths(inner_widths)
        self.num_classes = spec.num_classes
        self.patch_embed = nn.Conv2d(3, spec.embed_dim, spec.patch_size, spec.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, spec.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, spec.patch_tokens + 1, spec.embed_dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(spec.embed_dim, spec.num_heads, width)
            for width in self.inner_widths
        )
        self.norm = nn.LayerNorm(spec.embed_dim, eps=1e-6)
        self.head = nn.Linear(spec.embed_dim, spec.num_classes)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def activation_modules(self) -> dict[str, nn.Module]:
        return {f"block{i + 1}": block.act for i, block in enumerate(self.blocks)}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.spec.image_size, self.spec.image_size):
            raise ValueError("Input must be B x 3 x image_size x image_size")
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])
