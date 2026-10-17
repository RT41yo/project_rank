# Сводка экспериментов

## Отдельные запуски

| Запуск | Лучшая эпоха | Validation accuracy | Test accuracy | Параметры |
|---|---:|---:|---:|---:|
| baseline_seed1 | 160 | 92.36% | 92.49% | 272474 |
| baseline_seed2 | 154 | 92.70% | 92.04% | 272474 |
| baseline_seed3 | 146 | 92.10% | 91.80% | 272474 |
| p060_erank_per_image_finetune_seed1 | 28 | 91.98% | 91.70% | 162790 |
| p060_erank_per_image_scratch_seed1 | 160 | 92.32% | 91.33% | 162790 |
| p060_erank_per_image_scratch_seed2 | 154 | 91.66% | 91.85% | 162790 |
| p060_erank_per_image_scratch_seed3 | 149 | 91.26% | 91.20% | 162790 |
| p060_erank_pooled_finetune_seed1 | 28 | 91.72% | 91.17% | 162470 |
| p060_erank_pooled_scratch_seed1 | 156 | 91.72% | 90.95% | 162470 |
| p060_erank_pooled_scratch_seed2 | 154 | 91.76% | 91.03% | 162470 |
| p060_erank_pooled_scratch_seed3 | 159 | 91.06% | 90.64% | 162470 |
| p060_random_finetune_seed1 | 25 | 91.70% | 91.00% | 162470 |
| p060_random_scratch_seed1 | 157 | 91.76% | 91.04% | 162470 |
| p060_uniform_finetune_seed1 | 28 | 91.60% | 91.05% | 162470 |
| p060_uniform_scratch_seed1 | 147 | 92.12% | 91.47% | 162470 |
| p060_uniform_scratch_seed2 | 157 | 91.60% | 91.12% | 162470 |
| p060_uniform_scratch_seed3 | 159 | 91.06% | 90.95% | 162470 |
| p080_erank_per_image_finetune_seed1 | 28 | 92.56% | 92.07% | 218226 |
| p080_erank_per_image_scratch_seed1 | 154 | 92.66% | 92.22% | 218226 |
| p080_erank_pooled_finetune_seed1 | 23 | 91.76% | 91.83% | 218766 |
| p080_erank_pooled_scratch_seed1 | 144 | 92.62% | 91.72% | 218766 |
| p080_random_finetune_seed1 | 28 | 92.32% | 91.66% | 218770 |
| p080_random_scratch_seed1 | 152 | 92.76% | 91.81% | 218770 |
| p080_uniform_finetune_seed1 | 29 | 92.38% | 92.23% | 219062 |
| p080_uniform_scratch_seed1 | 158 | 92.78% | 91.95% | 219062 |

## Средние значения по seed

| Бюджет | Метод | Режим | Seed | Validation mean ± std | Test mean ± std |
|---|---|---|---|---:|---:|
| full | baseline | scratch | 1, 2, 3 | 92.39% ± 0.30 п.п. | 92.11% ± 0.35 п.п. |
| p060 | erank_per_image | finetune | 1 | 91.98% | 91.70% |
| p060 | erank_per_image | scratch | 1, 2, 3 | 91.75% ± 0.54 п.п. | 91.46% ± 0.34 п.п. |
| p060 | erank_pooled | finetune | 1 | 91.72% | 91.17% |
| p060 | erank_pooled | scratch | 1, 2, 3 | 91.51% ± 0.39 п.п. | 90.87% ± 0.21 п.п. |
| p060 | random | finetune | 1 | 91.70% | 91.00% |
| p060 | random | scratch | 1 | 91.76% | 91.04% |
| p060 | uniform | finetune | 1 | 91.60% | 91.05% |
| p060 | uniform | scratch | 1, 2, 3 | 91.59% ± 0.53 п.п. | 91.18% ± 0.27 п.п. |
| p080 | erank_per_image | finetune | 1 | 92.56% | 92.07% |
| p080 | erank_per_image | scratch | 1 | 92.66% | 92.22% |
| p080 | erank_pooled | finetune | 1 | 91.76% | 91.83% |
| p080 | erank_pooled | scratch | 1 | 92.62% | 91.72% |
| p080 | random | finetune | 1 | 92.32% | 91.66% |
| p080 | random | scratch | 1 | 92.76% | 91.81% |
| p080 | uniform | finetune | 1 | 92.38% | 92.23% |
| p080 | uniform | scratch | 1 | 92.78% | 91.95% |
