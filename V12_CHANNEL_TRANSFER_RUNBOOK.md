# V12: исследование переноса каналов

## Что проверяется

Для одной и той же архитектуры сравниваются три способа выбора каналов из
полной ResNet-20:

- `l1` — сохраняются каналы с наиболее крупными весами;
- `random` — сохраняются случайные каналы, выбор воспроизводится по seed;
- `erank_subspace` — сохраняются каналы, которые сильнее участвуют в основном
  пространстве активаций.

Ширины блоков фиксированы. Поэтому различие результатов связано со способом
переноса, а не с размером модели.

Проверяются два бюджета:

- `p060` — умеренное сжатие;
- `p030` — сильное сжатие, где `erank_per_image` выиграл одношаговое сравнение.

## 1. Подготовка

Все команды выполняются из корня проекта.

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Проверьте наличие исходного checkpoint:

```bash
test -f results/baseline_seed1/best.pt
```

## 2. Создание начальных моделей и конфигураций

```bash
python -m project_rank.prepare_channel_transfer_study \
  --base-config configs/baseline.json \
  --architectures results/rank_analysis_seed1/architectures_grid_v4.json \
  --baseline-checkpoint results/baseline_seed1/best.pt \
  --budgets p060 p030 \
  --architecture erank_per_image \
  --fine-tune-epochs 30 \
  --fine-tune-lr 0.01 \
  --selection-seed 2001 \
  --max-samples 5000
```

Команда один раз собирает статистику активаций полной модели и создаёт шесть
начальных моделей: три способа переноса для каждого из двух бюджетов.

## 3. Точность до fine-tune

```bash
set -o pipefail
bash configs/channel_transfer_seed1/initial_test_commands.sh \
  2>&1 | tee results/channel_transfer_seed1/initial_test.log
```

Эта проверка показывает качество модели сразу после переноса каналов.

## 4. Fine-tune

```bash
set -o pipefail
bash configs/channel_transfer_seed1/train_commands.sh \
  2>&1 | tee results/channel_transfer_seed1/train.log
```

Уже существующий `p060 l1` автоматически пропускается, если его `best.pt`
сохранился в проекте. Обычно выполняются пять новых обучений: два для `p060` и
три для `p030`.

## 5. Оценка после fine-tune

```bash
set -o pipefail
bash configs/channel_transfer_seed1/test_commands.sh \
  2>&1 | tee results/channel_transfer_seed1/test.log
```

## 6. Итоговая сводка

```bash
python -m project_rank.summarize_channel_transfer \
  --manifest configs/channel_transfer_seed1/manifest.json \
  --output-dir results/channel_transfer_seed1/summary
```

После завершения нужны три небольших файла:

```text
results/channel_transfer_seed1/summary/summary.json
results/channel_transfer_seed1/summary/summary.csv
results/channel_transfer_seed1/channel_selection.json
```

Их достаточно, чтобы проверить результаты и добавить сравнение в отчёт.

## Что не нужно добавлять в Git

Начальные и обученные checkpoint имеют расширение `.pt` и не включаются в Git
или ZIP-snapshot. Машиночитаемые JSON/CSV и итоговый текстовый отчёт можно
сохранить.
