from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from project_rank.models.resnet20 import DEFAULT_INNER_WIDTHS, ResNet20
from project_rank.models.vit_tiny import ViTTinyCIFAR
from project_rank.vit_spec import ViTSpec


def build_model(config: Mapping[str, Any]) -> nn.Module:
    specification = dict(config.get("model_spec", {"type": "resnet20"}))
    kind = specification.pop("type", "resnet20")
    classes = int(config.get("num_classes", specification.get("num_classes", 10)))
    if kind == "resnet20":
        return ResNet20(config.get("inner_widths", DEFAULT_INNER_WIDTHS), classes)
    if kind == "vit_tiny_cifar":
        if "num_classes" in specification and specification["num_classes"] != classes:
            raise ValueError("Conflicting num_classes and model_spec")
        specification["num_classes"] = classes
        return ViTTinyCIFAR(config.get("inner_widths"), ViTSpec(**specification))
    raise ValueError(f"Unsupported model type: {kind}")


def model_specification(model: nn.Module) -> dict[str, Any]:
    if isinstance(model, ViTTinyCIFAR):
        return model.spec.to_dict()
    if isinstance(model, ResNet20):
        return {"type": "resnet20"}
    raise TypeError(f"Unknown model class: {type(model).__name__}")


def model_from_checkpoint(checkpoint: Mapping[str, Any]) -> nn.Module:
    # Legacy checkpoints deliberately default to ResNet; do not infer from widths.
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    return model
