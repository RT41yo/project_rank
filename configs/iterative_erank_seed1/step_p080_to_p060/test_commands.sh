#!/usr/bin/env bash
set -euo pipefail

python -m project_rank.evaluate \
  --config configs/iterative_erank_seed1/step_p080_to_p060/p060_finetune_l1.json \
  --checkpoint results/iterative_erank_seed1/step_p080_to_p060/finetune_l1/best.pt \
  --split test \
  --output results/iterative_erank_seed1/step_p080_to_p060/finetune_l1/test.json

python -m project_rank.evaluate \
  --config configs/iterative_erank_seed1/step_p080_to_p060/p060_scratch.json \
  --checkpoint results/iterative_erank_seed1/step_p080_to_p060/scratch/best.pt \
  --split test \
  --output results/iterative_erank_seed1/step_p080_to_p060/scratch/test.json
