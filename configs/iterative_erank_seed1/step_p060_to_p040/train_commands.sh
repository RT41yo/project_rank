#!/usr/bin/env bash
set -euo pipefail

python -m project_rank.train --config configs/iterative_erank_seed1/step_p060_to_p040/p040_finetune_l1.json
python -m project_rank.train --config configs/iterative_erank_seed1/step_p060_to_p040/p040_scratch.json
