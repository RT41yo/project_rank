#!/usr/bin/env bash
set -euo pipefail

if [[ ! -s results/p060_erank_per_image_finetune_seed1/best.pt ]]; then
  python -m project_rank.train --config configs/channel_transfer_seed1/p060_l1_finetune_seed1.json
fi

if [[ ! -s results/channel_transfer_seed1/p060_random_finetune_seed1/best.pt ]]; then
  python -m project_rank.train --config configs/channel_transfer_seed1/p060_random_finetune_seed1.json
fi

if [[ ! -s results/channel_transfer_seed1/p060_erank_subspace_finetune_seed1/best.pt ]]; then
  python -m project_rank.train --config configs/channel_transfer_seed1/p060_erank_subspace_finetune_seed1.json
fi

if [[ ! -s results/channel_transfer_seed1/p030_l1_finetune_seed1/best.pt ]]; then
  python -m project_rank.train --config configs/channel_transfer_seed1/p030_l1_finetune_seed1.json
fi

if [[ ! -s results/channel_transfer_seed1/p030_random_finetune_seed1/best.pt ]]; then
  python -m project_rank.train --config configs/channel_transfer_seed1/p030_random_finetune_seed1.json
fi

if [[ ! -s results/channel_transfer_seed1/p030_erank_subspace_finetune_seed1/best.pt ]]; then
  python -m project_rank.train --config configs/channel_transfer_seed1/p030_erank_subspace_finetune_seed1.json
fi
