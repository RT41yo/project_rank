#!/usr/bin/env bash
set -euo pipefail

python -m project_rank.evaluate \
  --config configs/channel_transfer_seed1/p060_l1_finetune_seed1.json \
  --checkpoint results/channel_transfer_initializations_seed1/p060_l1_init.pt \
  --split test \
  --output results/channel_transfer_seed1/p060_l1_initial_test.json

python -m project_rank.evaluate \
  --config configs/channel_transfer_seed1/p060_random_finetune_seed1.json \
  --checkpoint results/channel_transfer_initializations_seed1/p060_random_init.pt \
  --split test \
  --output results/channel_transfer_seed1/p060_random_initial_test.json

python -m project_rank.evaluate \
  --config configs/channel_transfer_seed1/p060_erank_subspace_finetune_seed1.json \
  --checkpoint results/channel_transfer_initializations_seed1/p060_erank_subspace_init.pt \
  --split test \
  --output results/channel_transfer_seed1/p060_erank_subspace_initial_test.json

python -m project_rank.evaluate \
  --config configs/channel_transfer_seed1/p030_l1_finetune_seed1.json \
  --checkpoint results/channel_transfer_initializations_seed1/p030_l1_init.pt \
  --split test \
  --output results/channel_transfer_seed1/p030_l1_initial_test.json

python -m project_rank.evaluate \
  --config configs/channel_transfer_seed1/p030_random_finetune_seed1.json \
  --checkpoint results/channel_transfer_initializations_seed1/p030_random_init.pt \
  --split test \
  --output results/channel_transfer_seed1/p030_random_initial_test.json

python -m project_rank.evaluate \
  --config configs/channel_transfer_seed1/p030_erank_subspace_finetune_seed1.json \
  --checkpoint results/channel_transfer_initializations_seed1/p030_erank_subspace_init.pt \
  --split test \
  --output results/channel_transfer_seed1/p030_erank_subspace_initial_test.json
