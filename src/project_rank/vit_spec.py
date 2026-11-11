"""Pure-Python specification and exact parameter/multiply-add counts for CIFAR ViT."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ViTSpec:
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    num_classes: int = 10

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("ViT dimensions must be positive integers")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

    @property
    def patch_tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def default_widths(self) -> tuple[int, ...]:
        return (4 * self.embed_dim,) * self.depth

    def widths(self, values: Sequence[int] | None = None) -> tuple[int, ...]:
        result = self.default_widths if values is None else tuple(values)
        if len(result) != self.depth or any(type(v) is not int or v < 1 for v in result):
            raise ValueError(f"Expected {self.depth} positive integer MLP widths")
        return result

    def to_dict(self) -> dict[str, int | str]:
        return {"type": "vit_tiny_cifar", **asdict(self)}


def count_vit_parameters(
    inner_widths: Sequence[int] | None = None, spec: ViTSpec = ViTSpec()
) -> int:
    widths = spec.widths(inner_widths)
    d, t = spec.embed_dim, spec.patch_tokens + 1
    # Patch embedding (with bias), CLS, learned position embedding, final LN/head.
    total = 3 * spec.patch_size**2 * d + d + d + t * d
    total += 2 * d + d * spec.num_classes + spec.num_classes
    # Two LNs, QKV/projection biases, fc2 bias; variable fc1/fc2 weights + fc1 bias.
    return total + spec.depth * (4 * d * d + 9 * d) + (2 * d + 1) * sum(widths)


def count_vit_macs(
    inner_widths: Sequence[int] | None = None, spec: ViTSpec = ViTSpec()
) -> int:
    """Per image: projections AND QK^T/AV; excludes softmax/norm/GELU/add/bias."""
    widths = spec.widths(inner_widths)
    d, t = spec.embed_dim, spec.patch_tokens + 1
    total = spec.patch_tokens * 3 * spec.patch_size**2 * d + d * spec.num_classes
    return total + spec.depth * (4 * t * d * d + 2 * t * t * d) + 2 * t * d * sum(widths)
