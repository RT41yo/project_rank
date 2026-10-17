from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from project_rank.spec import (
    BLOCK_STRIDES,
    DEFAULT_INNER_WIDTHS,
    STAGE_OUTPUT_WIDTHS,
    count_resnet20_parameters,
)


class BasicBlock(nn.Module):
    """Residual block whose internal channel count can be changed safely."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inner_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, inner_channels) < 1:
            raise ValueError("All channel counts must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channels = inner_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            inner_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            inner_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + residual)


class ResNet20(nn.Module):
    """CIFAR ResNet-20 with nine configurable internal block widths.

    The input and output widths of each residual block remain fixed at the
    conventional stage widths (16, 32, 64). Only the intermediate width in
    ``Conv1 -> BN -> ReLU -> Conv2`` changes, so residual additions stay valid.
    """

    def __init__(
        self,
        inner_widths: Sequence[int] = DEFAULT_INNER_WIDTHS,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in inner_widths)
        if len(widths) != 9:
            raise ValueError(f"Expected 9 inner widths, got {len(widths)}")
        if any(value < 1 for value in widths):
            raise ValueError("Every inner width must be positive")

        self.inner_widths = widths
        self.num_classes = int(num_classes)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
        )

        blocks: list[nn.Module] = []
        in_channels = 16
        for out_channels, inner_channels, stride in zip(
            STAGE_OUTPUT_WIDTHS,
            widths,
            BLOCK_STRIDES,
            strict=True,
        ):
            blocks.append(
                BasicBlock(in_channels, out_channels, inner_channels, stride)
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, self.num_classes)

        self._initialize_weights()

    @property
    def activation_modules(self) -> dict[str, nn.Module]:
        """Modules whose outputs are used for effective-rank analysis."""
        return {
            f"block{index + 1}": block.relu1
            for index, block in enumerate(self.blocks)
        }

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for block in self.blocks:
            out = block(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)

