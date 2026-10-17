from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from project_rank.models import ResNet20
from project_rank.utils import load_json


def _copy_batch_norm(source: nn.BatchNorm2d, target: nn.BatchNorm2d, indices=None) -> None:
    if indices is None:
        target.load_state_dict(source.state_dict())
        return
    for name in ("weight", "bias", "running_mean", "running_var"):
        source_value = getattr(source, name)
        target_value = getattr(target, name)
        target_value.copy_(source_value[indices])
    target.num_batches_tracked.copy_(source.num_batches_tracked)


@torch.no_grad()
def transfer_by_l1(source: ResNet20, target: ResNet20) -> dict[str, list[int]]:
    """Transfer a full model into a narrower model using Conv1 filter L1 norm.

    ERank chooses how many internal channels a block receives. L1 norm chooses
    which particular channels are copied. Inputs and outputs of blocks are not
    narrowed, so channel choices do not propagate between residual blocks.
    """
    if source.num_classes != target.num_classes:
        raise ValueError("Source and target must have the same number of classes")
    target.stem.load_state_dict(source.stem.state_dict())
    target.fc.load_state_dict(source.fc.state_dict())
    selected: dict[str, list[int]] = {}

    for index, (source_block, target_block) in enumerate(
        zip(source.blocks, target.blocks, strict=True)
    ):
        keep = target_block.inner_channels
        if keep > source_block.inner_channels:
            raise ValueError(
                f"Target block {index + 1} is wider than source: "
                f"{keep} > {source_block.inner_channels}"
            )
        importance = source_block.conv1.weight.abs().sum(dim=(1, 2, 3))
        indices = importance.topk(keep, largest=True).indices.sort().values
        selected[f"block{index + 1}"] = indices.cpu().tolist()

        target_block.conv1.weight.copy_(source_block.conv1.weight[indices])
        _copy_batch_norm(source_block.bn1, target_block.bn1, indices)
        target_block.conv2.weight.copy_(source_block.conv2.weight[:, indices])
        _copy_batch_norm(source_block.bn2, target_block.bn2)
        target_block.shortcut.load_state_dict(source_block.shortcut.state_dict())

    return selected


def _find_architecture(data: dict[str, Any], name: str) -> list[int]:
    for item in data.get("architectures", []):
        if item.get("name") == name:
            return [int(value) for value in item["inner_widths"]]
    raise KeyError(f"Architecture {name!r} was not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an L1-transferred checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--architecture", default="erank")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    source = ResNet20(
        source_checkpoint["inner_widths"],
        source_checkpoint.get("num_classes", 10),
    )
    source.load_state_dict(source_checkpoint["model_state"])

    widths = _find_architecture(load_json(args.architectures), args.architecture)
    target = ResNet20(widths, source.num_classes)
    selected = transfer_by_l1(source, target)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 0,
            "model_state": target.state_dict(),
            "inner_widths": widths,
            "num_classes": source.num_classes,
            "source_checkpoint": str(args.checkpoint),
            "transfer_method": "conv1_l1",
            "selected_channels": selected,
        },
        output,
    )
    print(json.dumps({"output": str(output), "inner_widths": widths}, indent=2))


if __name__ == "__main__":
    main()

