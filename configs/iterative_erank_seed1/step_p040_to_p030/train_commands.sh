#!/usr/bin/env bash
set -euo pipefail

python -m project_rank.train --config configs/iterative_erank_seed1/step_p040_to_p030/p030_finetune_l1.json
python -m project_rank.train --config configs/iterative_erank_seed1/step_p040_to_p030/p030_scratch.json
