from __future__ import annotations

import argparse
import csv
from pathlib import Path

from project_rank.models.factory import build_model
from project_rank.models.vit_tiny import ViTTinyCIFAR
from project_rank.utils import load_json, save_json
from project_rank.vit_architectures import random_vit_architecture, scaled_vit_architecture
from project_rank.vit_spec import count_vit_macs, count_vit_parameters


RANK_COLUMNS = {
    "erank_per_image": "mean_per_image_effective_rank",
    "erank_pooled": "pooled_effective_rank",
    "erank_weight": "weight_effective_rank",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ViT rank/uniform/random profiles without training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-param-ratios", type=float, nargs="+", default=[.6])
    parser.add_argument("--random-seeds", type=int, nargs="+", default=list(range(1001, 1011)))
    args = parser.parse_args()
    config = load_json(args.config)
    model = build_model(config)
    if not isinstance(model, ViTTinyCIFAR) or model.inner_widths != model.spec.default_widths:
        parser.error("Expected the full ViT configuration")
    if not args.random_seeds or len(set(args.random_seeds)) != len(args.random_seeds):
        parser.error("Random seeds must be nonempty and unique")
    spec = model.spec
    with open(args.ranks, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if [r["block"] for r in rows] != [f"block{i + 1}" for i in range(spec.depth)]:
        raise ValueError("Expected ordered ranks for every transformer block")
    if any(int(r["channels"]) != width for r, width in zip(rows, model.inner_widths, strict=True)):
        raise ValueError("Rank widths differ from source configuration")
    full = count_vit_parameters(spec=spec)
    metadata_path = Path(args.ranks).with_name("analysis_metadata.json")
    metadata = load_json(metadata_path) if metadata_path.exists() else None
    if metadata and metadata["model_spec"] != spec.to_dict():
        raise ValueError("Rank metadata specification differs from configuration")
    report = {
        "schema": "vit_mlp_architectures_v1", "protocol_id": config.get("protocol_id"),
        "model_spec": spec.to_dict(), "source_config": args.config, "source_ranks": args.ranks,
        "analysis_metadata": metadata,
        "full_parameters": full,
        "rounding": {"minimum": 4, "erank_multiple": 2, "random_multiple": 2, "uniform_multiple": 1},
        "random_sampler": "uniform_bounded_composition_conditional_on_width_sum",
        "not_yet_included": ["l1_weight", "taylor", "training_manifest"], "budgets": [],
    }
    for ratio in args.target_param_ratios:
        if not 0 < ratio <= 1:
            raise ValueError("Parameter ratios must be in (0,1]")
        target = round(full * ratio)
        architectures = [scaled_vit_architecture(
            [float(r[column]) for r in rows], target, spec, name=name,
        ) for name, column in RANK_COLUMNS.items()]
        reference = next(a.parameters for a in architectures if a.name == "erank_pooled")
        architectures.append(scaled_vit_architecture(
            spec.default_widths, reference, spec, multiple=1, name="uniform",
        ))
        profiles, entries = set(), []
        for architecture, seed in [(a, None) for a in architectures] + [
            (random_vit_architecture(reference, seed, spec), seed) for seed in args.random_seeds
        ]:
            if seed is not None:
                if architecture.inner_widths in profiles:
                    raise ValueError("Duplicate random profile; fix seeds/design before training")
                profiles.add(architecture.inner_widths)
            difference = (architecture.parameters - reference) / reference
            if abs(difference) > .005:
                raise ValueError("Parameter mismatch exceeds 0.5%; do not launch training")
            entries.append({
                **architecture.to_dict(), "architecture_seed": seed,
                "parameter_ratio_vs_full": architecture.parameters / full,
                "difference_vs_reference_percent": 100 * difference,
                "macs_per_image": count_vit_macs(architecture.inner_widths, spec),
            })
        report["budgets"].append({
            "label": f"p{round(ratio * 100):03d}", "requested_parameters": target,
            "reference_architecture": "erank_pooled", "reference_parameters": reference,
            "architectures": entries,
        })
    save_json(report, args.output)
    print(f"Saved {len(report['budgets'])} budget(s) to {args.output}; no training started")


if __name__ == "__main__":
    main()
