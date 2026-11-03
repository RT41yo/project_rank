#!/usr/bin/env bash
set -euo pipefail

python -m project_rank.train --config configs/iterative_erank_seed1/step_p080_to_p060/p060_finetune_l1.json
python -m project_rank.train --config configs/iterative_erank_seed1/step_p080_to_p060/p060_scratch.json
