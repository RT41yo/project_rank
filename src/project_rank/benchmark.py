from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from project_rank.models.factory import model_from_checkpoint
from project_rank.models.vit_tiny import Attention
from project_rank.utils import resolve_device, save_json


def count_macs(model: nn.Module, sample: torch.Tensor) -> int:
    macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs: Any, output: torch.Tensor) -> None:
        nonlocal macs
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        kernel_ops *= module.in_channels // module.groups
        macs += int(output.numel() * kernel_ops)

    def linear_hook(module: nn.Linear, _inputs: Any, output: torch.Tensor) -> None:
        nonlocal macs
        macs += int(output.numel() * module.in_features)

    def attention_hook(module: Attention, inputs: Any, _output: torch.Tensor) -> None:
        nonlocal macs
        batch, tokens, dim = inputs[0].shape
        macs += 2 * batch * tokens * tokens * dim  # QK^T and AV, projections above

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, Attention):
            handles.append(module.register_forward_hook(attention_hook))
    try:
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return macs


@torch.inference_mode()
def benchmark(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | str]:
    model.eval()
    if batch_size < 1 or warmup < 0 or iterations < 1:
        raise ValueError("batch_size/iterations must be positive and warmup non-negative")
    image_size = getattr(getattr(model, "spec", None), "image_size", 32)
    sample = torch.randn(batch_size, 3, image_size, image_size, device=device)
    macs = count_macs(model, sample)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    measurements = []
    for _ in range(iterations):
        started = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        measurements.append((time.perf_counter() - started) * 1000.0)

    measurements.sort()
    p90_index = min(len(measurements) - 1, int(0.9 * len(measurements)))
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return {
        "device": str(device),
        "batch_size": batch_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "macs_per_batch": macs,
        "flops_per_batch_approx": 2 * macs,
        "latency_median_ms": statistics.median(measurements),
        "latency_p90_ms": measurements[p90_index],
        "images_per_second": 1000.0 * batch_size / statistics.median(measurements),
        "peak_cuda_memory_bytes": peak_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a CIFAR checkpoint (FP32)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint).to(device)
    result = benchmark(
        model, device, args.batch_size, args.warmup, args.iterations
    )
    if args.output:
        save_json(result, Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
