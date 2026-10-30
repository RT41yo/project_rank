from __future__ import annotations

from collections.abc import Sequence


DEFAULT_INNER_WIDTHS = (16, 16, 16, 32, 32, 32, 64, 64, 64)
STAGE_OUTPUT_WIDTHS = (16, 16, 16, 32, 32, 32, 64, 64, 64)
BLOCK_STRIDES = (1, 1, 1, 2, 1, 1, 2, 1, 1)


def count_resnet20_parameters(
    inner_widths: Sequence[int] = DEFAULT_INNER_WIDTHS,
    num_classes: int = 10,
) -> int:
    """Exact trainable-parameter count without constructing a model."""
    widths = tuple(int(value) for value in inner_widths)
    if len(widths) != 9:
        raise ValueError(f"Expected 9 inner widths, got {len(widths)}")

    total = 3 * 16 * 3 * 3
    total += 2 * 16
    in_channels = 16
    for out_channels, inner_channels, stride in zip(
        STAGE_OUTPUT_WIDTHS,
        widths,
        BLOCK_STRIDES,
        strict=True,
    ):
        total += in_channels * inner_channels * 3 * 3
        total += 2 * inner_channels
        total += inner_channels * out_channels * 3 * 3
        total += 2 * out_channels
        if stride != 1 or in_channels != out_channels:
            total += in_channels * out_channels
            total += 2 * out_channels
        in_channels = out_channels
    total += 64 * num_classes + num_classes
    return int(total)


def count_resnet20_macs(
    inner_widths: Sequence[int] = DEFAULT_INNER_WIDTHS,
    num_classes: int = 10,
) -> int:
    """Exact Conv2d/Linear MAC count for one 32x32 image.

    The definition intentionally matches :func:`project_rank.benchmark.count_macs`:
    convolutions and the final linear layer are counted, while batch
    normalization, activations, residual additions and pooling are not.
    """
    widths = tuple(int(value) for value in inner_widths)
    if len(widths) != 9:
        raise ValueError(f"Expected 9 inner widths, got {len(widths)}")
    if any(value < 1 for value in widths):
        raise ValueError("Every inner width must be positive")

    spatial = 32
    total = spatial * spatial * 16 * 3 * 3 * 3
    in_channels = 16
    for out_channels, inner_channels, stride in zip(
        STAGE_OUTPUT_WIDTHS,
        widths,
        BLOCK_STRIDES,
        strict=True,
    ):
        spatial //= stride
        total += spatial * spatial * inner_channels * in_channels * 3 * 3
        total += spatial * spatial * out_channels * inner_channels * 3 * 3
        if stride != 1 or in_channels != out_channels:
            total += spatial * spatial * out_channels * in_channels
        in_channels = out_channels
    total += 64 * num_classes
    return int(total)

