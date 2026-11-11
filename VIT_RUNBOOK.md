# ViT: текущий рабочий контур

Полный научный план: [VIT_EXPERIMENT_PLAN.md](VIT_EXPERIMENT_PLAN.md).
Команды из корня проекта. Результаты ViT изолированы в `results/vit/`.

## Окружение

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m project_rank.check_environment
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

Не копировать версии PyTorch из README коллеги без проверки CUDA.
CPU-тесты не подтверждают GPU-совместимость — нужен настоящий backward.

## Synthetic smoke (не научный результат)

```bash
.venv/bin/python -m project_rank.train --config configs/vit/smoke.json
.venv/bin/python -m project_rank.evaluate \
  --config configs/vit/smoke.json \
  --checkpoint results/vit/smoke/best.pt \
  --split validation --output results/vit/smoke/validation.json
.venv/bin/python -m project_rank.analyze \
  --config configs/vit/smoke.json \
  --checkpoint results/vit/smoke/best.pt \
  --output-dir results/vit/smoke/rank_analysis --max-samples 8 --ranks-only
.venv/bin/python -m project_rank.benchmark \
  --checkpoint results/vit/smoke/best.pt --device cpu \
  --batch-size 4 --warmup 2 --iterations 5 \
  --output results/vit/smoke/benchmark.json
```

Synthetic E2E-тест проверяет также физический перенос в меньший MLP,
fine-tune и загрузку checkpoint. Старые checkpoint без `model_spec`
интерпретируются как ResNet, а не угадываются по числу ширин.

## Baseline — следующий длительный этап

Только после GPU/overfit-проверок из плана; не запускать сетку сразу:

```bash
.venv/bin/python -m project_rank.train --config configs/vit/baseline.json
```

`device=cuda` задан явно, чтобы не начать случайно 160 эпох на CPU.
Baseline-конфиг сохраняет SGD-рецепт коллеги. Test не используется для
решения о смене рецепта. Перед повторным запуском проверить, что каталог
результатов не содержит нужного незавершённого/завершённого обучения:
нынешний train не является полноценным restart. Для ViT перезапись
best/last/history запрещена без явного `overwrite=true`. Только synthetic
smoke разрешает перезапись. Baseline сохраняет также промежуточные веса
epoch_000/001/010/040/080/120/160.pt для offline ERank.

После завершения можно собрать ранги:

```bash
.venv/bin/python -m project_rank.analyze \
  --config configs/vit/baseline.json \
  --checkpoint results/vit/baseline_seed1/best.pt \
  --output-dir results/vit/rank_analysis_seed1 --max-samples 5000 --ranks-only
.venv/bin/python -m project_rank.pruning_criteria \
  --config configs/vit/baseline.json \
  --checkpoint results/vit/baseline_seed1/best.pt \
  --output results/vit/rank_analysis_seed1/pruning_scores.json --max-samples 5000
```

## Отдельный AdamW pilot после SGD baseline

SGD baseline завершена: 160 эпох, best validation accuracy 76.10% на эпохе
158; gate 80% не пройден. Результат сохранён и не перезаписывается.
Test не оценивался. Сводка: `reports/data/vit/baseline_sgd_v1.json`.

Согласованный отдельный pilot: `configs/vit/baseline_adamw_pilot.json`.
Данные, seed=1, архитектура, аугментации, batch128 и 160 эпох неизменны.
Модель обучается с нуля, не получает веса SGD-модели.

| Настройка | Значение |
|---|---:|
| Оптимизатор | AdamW |
| Peak LR | 0.0003 |
| Betas / eps | (0.9, 0.999) / 1e-8 |
| Weight decay | 0.05 |
| Исключения decay | bias, LayerNorm affine, CLS/position embeddings |
| Warmup | 5 эпох, start factor 0.01 |
| Затем | Cosine до нуля |

LR используется с 3e-6 на эпохе 1 до 3e-4 на эпохе 6, затем снижается.
`learning_rate_used` в history — LR выполненной эпохи; прежнее поле
`learning_rate` оставлено совместимым и содержит LR ПОСЛЕ scheduler.step.

```bash
.venv/bin/python -m project_rank.train --config configs/vit/baseline_adamw_pilot.json
```

Лог фонового запуска: `results/vit/baseline_adamw_pilot_seed1/train.log`.
Новый каталог и protocol_id отделяют pilot от SGD. После 160 эпох проверить
качество на validation, finite loss, полноту истории и checkpoint.
До фиксации ViT-рецепта не запускать сетку и не оценивать test.
При принятии нового рецепта применить его ко ВСЕМ ViT-методам; прежние
SGD-результаты не смешивать с основной AdamW-кривой. Технический gate
не является доказательством превосходства оптимизатора по одному seed.

### Полученный результат pilot

160 эпох завершены. Best validation accuracy **82.76%** (эпоха 152),
на 6.66 п.п. выше SGD. Технический gate 80% пройден; test не оценивался.
При этом train accuracy ≈99.99%, validation loss=1.1878 — выше, чем у SGD.
Это ограничение/признак переобучения, а не повод автоматически подбирать
рецепт по тому же validation бесконечно. Сводка:
`reports/data/vit/baseline_adamw_pilot_v1.json`.

Критерии собираются отдельно от SGD:

```bash
.venv/bin/python -m project_rank.analyze \
  --config configs/vit/baseline_adamw_pilot.json \
  --checkpoint results/vit/baseline_adamw_pilot_seed1/best.pt \
  --output-dir results/vit/rank_analysis_adamw_pilot_seed1 \
  --max-samples 5000 --ranks-only
.venv/bin/python -m project_rank.pruning_criteria \
  --config configs/vit/baseline_adamw_pilot.json \
  --checkpoint results/vit/baseline_adamw_pilot_seed1/best.pt \
  --output results/vit/rank_analysis_adamw_pilot_seed1/pruning_scores.json \
  --max-samples 5000
```

## Первый p060-опыт (stage D)

После успешного pilot фиксируем `vit_cifar10_adamw_v1`: тот же AdamW-рецепт
для всех scratch-методов. Pilot checkpoint служит baseline; его старое имя
и protocol_id сохраняются как provenance, повторно baseline не обучается.
SGD остаётся отдельным предварительным опытом.

| Запуск | Архитектура | Инициализация | Эпохи | Peak LR |
|---|---|---|---:|---:|
| erank_per_image scratch | ERank-профиль | Случайная | 160 | 3e-4 |
| random_arch1001 scratch | Random-профиль | Случайная | 160 | 3e-4 |
| l1 fine-tune | Тот же ERank-профиль | L1 из baseline | 30 | 3e-5 |
| random fine-tune | Тот же ERank-профиль | Mask seed2001 из baseline | 30 | 3e-5 |
| erank_subspace fine-tune | Тот же ERank-профиль | Subspace из baseline | 30 | 3e-5 |

У ВСЕХ пяти моделей 3 217 542 параметра и 226 030 464 MACs/image.
В трёх fine-tune совпадают ширины каждого блока. Train/data seed=1.
Во всех ветках AdamW WD=0.05 с теми же исключениями, warmup5 от factor0.01,
затем cosine. Fine-tune получает LR в 10 раз меньше, не SGD LR=0.01.
Короткие fine-tune не сохраняют дополнительные rank-checkpoints, scratch
сохраняет 0/1/10/40/80/120/160. Веса и optimizer scratch НЕ наследуются.

Подготовка (один раз; существующие каталоги НЕ перезаписываются):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m project_rank.prepare_vit_p060
```

Команда собирает covariance по 5 000 validation (без CLS), создаёт 3 init.pt,
пять JSON-конфигураций и `configs/vit/p060_stage_d_seed1/manifest.json`.
В manifest фиксируются SHA256 baseline, init и конфигураций, параметры/MACs,
seeds и происхождение. Индексы нейронов:
`results/vit/p060_stage_d_seed1/selected_neurons.json`.

Запуск или проверка/пропуск полностью завершённых запусков:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u -m project_rank.run_vit_study
```

Порядок: initial validation трёх init, scratch ERank, scratch random, затем
fine-tune L1/random/subspace. Обучения последовательны, test НЕ загружается.
Каждый run имеет свой `train.log` в `results/vit/p060_stage_d_seed1/<run>/`.
Общий лог фонового runner: `/tmp/opencode/vit_p060_stage_d_seed1.log`.
Состояния: `results/vit/p060_stage_d_seed1/run_status.json`.
После завершения: `results/vit/p060_stage_d_seed1/validation_summary.json`.

Первый stage D завершён (160/160/30/30/30 эпох). Validation accuracy:
ERank scratch 82.66%, random architecture 82.64%, L1 FT 81.38%,
random-mask FT 81.32%, subspace FT 82.44%. Test не оценивался.
Разница scratch всего 0.02 п.п.; не объявлять её доказанным преимуществом.
Публичная JSON/CSV-сводка — `reports/data/vit/p060_stage_d_validation.*`,
интерпретация и таблица initial/final — `reports/VIT_REPORT.md`.

Пропуск обучения разрешается только после сверки всей истории 1…N,
config, best/last checkpoint и числа изображений. Неполный запуск вызывает
ошибку, а не считается завершённым; автоматического restart ещё нет.
Не удалять и не перезаписывать частичный результат без отдельного решения.
Повторная подготовка поверх существующих init запрещена. Повторный runner
защищён файловой блокировкой от двух одновременных запусков одной серии.

## Ограничения текущей реализации

- Ранги и L1/Taylor pilot собраны: 5 000 изображений, 12 блоков;
  все оценки проверены на конечность. Публичная копия рангов:
  `reports/data/vit/rank_analysis_adamw_pilot_seed1/ranks.csv`.
- ViT scaling и условный random-сэмплер реализованы и протестированы.
  Превью архитектур p060 можно построить без обучения:

  ```bash
  .venv/bin/python -m project_rank.build_vit_architectures \
    --config configs/vit/baseline_adamw_pilot.json \
    --ranks reports/data/vit/rank_analysis_adamw_pilot_seed1/ranks.csv \
    --output reports/data/vit/architectures_p060_preview.json \
    --target-param-ratios 0.60
  ```

- В превью пока только 3 ERank-варианта, uniform и 10 random-профилей;
  L1/Taylor-архитектуры, полная grid-серия, sensitivity, iterative и полная
  отчётная система ещё не портированы. Эта команда НЕ обучает сети.
  Manifest/completion для первого пятизапускового stage D уже реализованы.
- `analyze` для ViT требует `--ranks-only`; отдельный budget builder —
  `build_vit_architectures`, а не старый `build_architectures` ResNet.
- CLI переноса `project_rank.transfer` пока ResNet-only; ViT-функции
  находятся в `vit_transfer.py` и проверяются тестами.
- Offline-анализ динамики рангов ещё не автоматизирован; промежуточные
  checkpoint уже сохраняются baseline-конфигурацией.
- Не называть P0/synthetic-проверки завершёнными CIFAR-экспериментами.
