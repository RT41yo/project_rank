# Запуск экспериментов v06

Эта версия добавляет честное сравнение ERank с двумя классическими критериями
прунинга: `l1_weight` и `taylor`. Исходная полная модель повторно не обучается.
Используется уже существующий файл `results/baseline_seed1/best.pt`.

## 1. Быстрая проверка после распаковки

```bash
python -m project_rank.check_environment
python -m pytest -q
python -m project_rank.train --config configs/smoke.json
```

## 2. Расчет оценок каналов

```bash
python -m project_rank.pruning_criteria \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --output results/rank_analysis_seed1/pruning_scores.json \
  --max-samples 5000
```

Этот шаг использует GPU, если в `configs/baseline.json` указано `auto` или
`cuda` и CUDA доступна. Обучения нет: веса не обновляются.

## 3. Построение новых архитектур

```bash
python -m project_rank.build_pruning_architectures \
  --architectures results/rank_analysis_seed1/architectures_v2.json \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output results/rank_analysis_seed1/architectures_v3.json

python -m project_rank.prepare_classic_experiments \
  --base-config configs/baseline.json \
  --architectures results/rank_analysis_seed1/architectures_v3.json \
  --output-dir configs/generated_v3
```

После этого в `configs/generated_v3/manifest.json` будет список ровно восьми
обучений.

## 4. Анализ чувствительности блоков

```bash
python -m project_rank.layer_sensitivity \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output-dir results/layer_sensitivity_seed1 \
  --reduction-ratio 0.25
```

Это десять коротких проверок: исходная модель и девять вариантов, в которых
сужен только один блок. Дообучения нет.

## 5. Восемь новых обучений

```bash
python -m project_rank.train --config configs/generated_v3/p080_l1_weight_scratch_seed1.json
python -m project_rank.train --config configs/generated_v3/p080_taylor_scratch_seed1.json

python -m project_rank.train --config configs/generated_v3/p060_l1_weight_scratch_seed1.json
python -m project_rank.train --config configs/generated_v3/p060_l1_weight_scratch_seed2.json
python -m project_rank.train --config configs/generated_v3/p060_l1_weight_scratch_seed3.json

python -m project_rank.train --config configs/generated_v3/p060_taylor_scratch_seed1.json
python -m project_rank.train --config configs/generated_v3/p060_taylor_scratch_seed2.json
python -m project_rank.train --config configs/generated_v3/p060_taylor_scratch_seed3.json
```

Запуски можно выполнять по одному. Если процесс остановится, уже завершенные
каталоги результатов сохранятся.

## 6. Оценка на test

Для каждого из восьми запусков выполните команду такого вида:

```bash
python -m project_rank.evaluate \
  --config results/p060_l1_weight_scratch_seed1/config.json \
  --checkpoint results/p060_l1_weight_scratch_seed1/best.pt \
  --split test \
  --output results/p060_l1_weight_scratch_seed1/test.json
```

Меняются только имя каталога и соответствующий `seed`.

## 7. Benchmark для четырех архитектур seed1

```bash
python -m project_rank.benchmark \
  --checkpoint results/p080_l1_weight_scratch_seed1/best.pt \
  --device cuda --batch-size 128 --warmup 50 --iterations 200 \
  --output results/p080_l1_weight_scratch_seed1/benchmark.json

python -m project_rank.benchmark \
  --checkpoint results/p080_taylor_scratch_seed1/best.pt \
  --device cuda --batch-size 128 --warmup 50 --iterations 200 \
  --output results/p080_taylor_scratch_seed1/benchmark.json

python -m project_rank.benchmark \
  --checkpoint results/p060_l1_weight_scratch_seed1/best.pt \
  --device cuda --batch-size 128 --warmup 50 --iterations 200 \
  --output results/p060_l1_weight_scratch_seed1/benchmark.json

python -m project_rank.benchmark \
  --checkpoint results/p060_taylor_scratch_seed1/best.pt \
  --device cuda --batch-size 128 --warmup 50 --iterations 200 \
  --output results/p060_taylor_scratch_seed1/benchmark.json
```

## 8. Итоговая сводка

```bash
python -m project_rank.summarize_results \
  --results-dir results \
  --output-dir results/summary_v06
```

## 9. Что прислать для завершения отчета

Нужны только легкие файлы, без `best.pt` и `last.pt`:

- `results/rank_analysis_seed1/pruning_scores.json`;
- `results/rank_analysis_seed1/architectures_v3.json`;
- каталог `results/layer_sensitivity_seed1/`;
- каталог `results/summary_v06/`;
- `config.json`, `history.json`, `test.json`, `benchmark.json` из восьми новых
  каталогов запусков. `benchmark.json` обязателен только для четырех `seed1`.

Можно снова прислать полный ZIP-snapshot репозитория по прежним правилам: веса,
набор данных, `.venv`, `.git` и кэши в архив включать не нужно.
