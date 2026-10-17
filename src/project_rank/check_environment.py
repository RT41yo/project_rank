from __future__ import annotations

import json
import platform

import numpy
import torch
import torchvision

from project_rank.utils import environment_summary


def main() -> None:
    summary = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torchvision": torchvision.__version__,
        **environment_summary(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

