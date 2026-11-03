#!/usr/bin/env bash
set -euo pipefail

python -m project_rank.evaluate \
  --config configs/iterative_erank_seed1/step_p040_to_p030/p030_finetune_l1.json \
  --checkpoint results/iterative_erank_seed1/step_p040_to_p030/finetune_l1/best.pt \
  --split test \
  --output results/iterative_erank_seed1/step_p040_to_p030/finetune_l1/test.json

python -m project_rank.evaluate \
  --config configs/iterative_erank_seed1/step_p040_to_p030/p030_scratch.json \
  --checkpoint results/iterative_erank_seed1/step_p040_to_p030/scratch/best.pt \
  --split test \
  --output results/iterative_erank_seed1/step_p040_to_p030/scratch/test.json
