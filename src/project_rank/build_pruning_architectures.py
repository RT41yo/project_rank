from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project_rank.architectures import extend_with_pruning_architectures
from project_rank.utils import load_json, save_json


def read_pruning_scores(data: dict[str, Any]) -> dict[str, list[list[float]]]:
    if data.get("schema_version") != 1:
        raise ValueError("Expected pruning score schema_version=1")
    criteria = data.get("criteria", {})
    result: dict[str, list[list[float]]] = {}
    for method in ("l1_weight", "taylor"):
        blocks = criteria.get(method)
        if not isinstance(blocks, dict):
            raise ValueError(f"Missing criterion {method!r}")
        result[method] = []
        for index in range(1, 10):
            block = blocks.get(f"block{index}")
            if not isinstance(block, dict) or "raw_scores" not in block:
                raise ValueError(f"Missing raw scores for {method} block{index}")
            result[method].append([float(value) for value in block["raw_scores"]])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add L1-weight and Taylor architectures to the ERank report"
    )
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = extend_with_pruning_architectures(
        load_json(args.architectures),
        read_pruning_scores(load_json(args.scores)),
    )
    report["source_architectures"] = str(args.architectures)
    report["source_pruning_scores"] = str(args.scores)
    save_json(report, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
