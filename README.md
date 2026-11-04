# Поиск архитектуры ResNet-20 через Effective Rank

Проект исследует выбор компактной архитектуры ResNet-20 для CIFAR-10 по
Effective Rank и сопоставляет его с классическими критериями прунинга и
контрольными способами распределения ширин.

Реализованы:

- варианты `erank_per_image`, `erank_pooled` и `erank_weight`;
- критерии `l1_weight` и `taylor`;
- контрольные методы `uniform` и `random`;
- одношаговая сетка бюджетов от `p090` до `p030`;
- повторные обучения для оценки разброса;
- анализ чувствительности отдельных блоков;
- сравнение способов переноса каналов;
- итеративный поиск `p080 → p060 → p040 → p030` с повторным расчётом
  Effective Rank.

README содержит только описание инфраструктуры и порядок воспроизведения.
Численные результаты, графики, интерпретация и ограничения экспериментов
приведены в [`reports/REPORT.md`](reports/REPORT.md).

## 1. Экспериментальная инфраструктура

### Аппаратное обеспечение основного запуска

| Компонент | Конфигурация |
|---|---|
| Процессор | Intel Core Ultra 7 265KF |
| Видеокарта | NVIDIA GeForce RTX 5070 Ti |
| Память видеокарты | 16 ГБ |
| Оперативная память | 64 ГБ |
| Накопитель | NVMe SSD |

GPU используется для обучения, сбора активаций, расчёта `taylor`, переноса
каналов и оценки моделей. Подсчёт параметров и MACs выполняется по архитектуре
и не требует checkpoint или GPU.

### Программное окружение основного запуска

| Компонент | Версия |
|---|---|
| Операционная система | Manjaro Linux |
| Ядро Linux | 6.6.156-2-MANJARO |
| Python | 3.14.7 |
| PyTorch | 2.14.1+cu130 |
| CUDA runtime PyTorch | 13.0 |
| cuDNN | 92400 |
| `numpy` | не ниже 1.26 |
| `matplotlib` | не ниже 3.8 |
| `pytest` | не ниже 8.0 |

Проект допускает Python 3.10 и новее, PyTorch 2.2 и новее и совместимый с ним
`torchvision`. Фактическое окружение каждого обучения сохраняется в
`results/<ИМЯ_ЗАПУСКА>/environment.json`.

## 2. Структура проекта

```text
project_rank/
├── configs/
│   ├── baseline.json                 полная ResNet-20
│   ├── smoke.json                    короткая синтетическая проверка
│   ├── generated_v3/                 дополнительные повторы p060
│   ├── generated_v4/                 одношаговая сетка p090–p030
│   ├── repeats/                      повторы baseline и p060
│   ├── channel_transfer_seed1/       создаётся для переноса каналов
│   └── iterative_erank_seed1/        три шага итеративного поиска
├── reports/
│   ├── REPORT.md                     результаты и выводы
│   ├── data/                         проверенные данные для отчёта
│   └── figures/                      воспроизводимые графики
├── results/
│   ├── rank_analysis_seed1/          ERank, критерии и архитектуры
│   ├── layer_sensitivity_seed1/      чувствительность при сужении на 25%
│   ├── layer_sensitivity_r050_seed1/ чувствительность при сужении на 50%
│   ├── channel_transfer_seed1/       JSON/CSV опыта переноса каналов
│   ├── iterative_erank_seed1/        JSON/CSV итеративной цепочки
│   └── summary_v09/                  сводка одношаговой сетки
├── src/project_rank/                 исходный код
├── tests/                            автоматические тесты
├── EXPERIMENT_PROTOCOL.md            зафиксированный протокол
├── V12_CHANNEL_TRANSFER_RUNBOOK.md   подробности переноса каналов
└── V14_ITERATIVE_ERANK_RUNBOOK.md    подробности итеративного поиска
```

Каталоги `.git/`, `.venv/`, `data/`, локальные checkpoint `*.pt` и кэши не
должны попадать в Git или ZIP-snapshot. В репозитории сохраняются код,
конфигурации, отчёт, графики и небольшие итоговые JSON/CSV.

## 3. Подготовка окружения

Все последующие команды выполняются из корня `project_rank`.

```bash
git clone <URL_РЕПОЗИТОРИЯ> project_rank
cd project_rank
git switch roman_feature

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Если PyTorch устанавливается отдельным wheel для конкретной CUDA, сначала
установите совместимые `torch` и `torchvision`, затем зависимости проекта:

```bash
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev]"
```

Для другой версии CUDA следует заменить индекс на вариант из официального
PyTorch selector. Проверка окружения и кода:

```bash
python -m project_rank.check_environment
python -m pytest -q
python -m project_rank.train --config configs/smoke.json
```

Smoke-проверка использует синтетические данные и не загружает CIFAR-10.

## 4. Данные и общий протокол обучения

CIFAR-10 автоматически загружается в каталог `data/`. Официальные 50 000
обучающих изображений делятся на 45 000 для обучения и 5 000 для проверки.
Официальные 10 000 тестовых изображений используются только после обучения и
выбора лучшей эпохи.

| Параметр | Значение |
|---|---:|
| Эпохи `scratch` | 160 |
| Batch size | 128 |
| Оптимизатор | SGD с momentum и Nesterov |
| Начальный `learning rate` | 0,1 |
| Momentum | 0,9 |
| Weight decay | 0,0005 |
| План `learning rate` | CosineAnnealing |
| AMP | включён на CUDA |
| Выбор checkpoint | максимум `validation accuracy` |

Основной протокол зафиксирован в `configs/baseline.json` и
`EXPERIMENT_PROTOCOL.md`. Не изменяйте гиперпараметры между сравниваемыми
методами.

## 5. Воспроизведение одношаговой сетки

### Шаг 1. Обучение полной модели

```bash
python -m project_rank.train --config configs/baseline.json
```

Checkpoint создаётся в `results/baseline_seed1/best.pt`.

### Шаг 2. Расчёт Effective Rank

```bash
python -m project_rank.analyze \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --output-dir results/rank_analysis_seed1 \
  --max-samples 5000 \
  --target-param-ratio 0.60 \
  --random-seed 1001
```

Команда сохраняет ранги, оценки и базовые варианты архитектур в
`results/rank_analysis_seed1/`.

### Шаг 3. Расчёт `l1_weight` и `taylor`

```bash
python -m project_rank.pruning_criteria \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --output results/rank_analysis_seed1/pruning_scores.json \
  --max-samples 5000
```

Для `taylor` выполняются прямой и обратный проходы, но веса не обновляются.

### Шаг 4. Построение архитектур семи бюджетов

```bash
python -m project_rank.build_architectures \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --output results/rank_analysis_seed1/architectures_grid_v4_base.json \
  --target-param-ratios 0.90 0.80 0.70 0.60 0.50 0.40 0.30 \
  --random-seed 1001

python -m project_rank.build_pruning_architectures \
  --architectures \
    results/rank_analysis_seed1/architectures_grid_v4_base.json \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output results/rank_analysis_seed1/architectures_grid_v4.json
```

### Шаг 5. Подготовка конфигураций

```bash
python -m project_rank.prepare_grid_experiments \
  --base-config configs/baseline.json \
  --architectures results/rank_analysis_seed1/architectures_grid_v4.json \
  --output-dir configs/generated_v4 \
  --results-dir results \
  --budgets p090 p080 p070 p060 p050 p040 p030 \
  --methods \
    erank_per_image erank_pooled erank_weight \
    l1_weight taylor uniform \
  --train-seed 1 \
  --random-architecture-seeds \
    1001 1002 1003 1004 1005 1006 1007 1008 1009 1010
```

Команда создаёт конфигурации, `manifest.json`, `train_commands.txt` и
`test_commands.txt`. Совпадающие архитектуры отмечаются как `aliases` и не
обучаются повторно.

### Шаг 6. Длительное обучение

```bash
set -o pipefail

bash configs/generated_v4/train_commands.txt \
  2>&1 | tee results/grid_v4_train.log
```

Команды выполняются последовательно и останавливаются при первой ошибке.
Повторный запуск шага 5 обновляет manifest и пропускает уже завершённые
обучения.

### Шаг 7. Оценка на тестовой выборке

```bash
set -o pipefail

bash configs/generated_v4/test_commands.txt \
  2>&1 | tee results/grid_v4_test.log
```

Для каждого физического запуска создаётся `results/<ИМЯ_ЗАПУСКА>/test.json`
с оценкой на 10 000 тестовых изображениях.

## 6. Сравнение scratch и fine-tune на `p080` и `p060`

Этот эксперимент использует одинаковые архитектуры в двух режимах:

- `scratch` — обучение со случайной инициализации;
- `finetune` — 30 эпох после L1-переноса каналов из полной модели.

Подготовка конфигураций и начальных checkpoint:

```bash
python -m project_rank.prepare_experiments \
  --base-config configs/baseline.json \
  --architectures results/rank_analysis_seed1/architectures_v2.json \
  --baseline-checkpoint results/baseline_seed1/best.pt \
  --output-dir configs/generated_v2 \
  --initializations-dir results/compact_initializations \
  --fine-tune-epochs 30 \
  --fine-tune-lr 0.01
```

Одношаговые `scratch`-модели входят в основную сетку. Ветки fine-tune
запускаются отдельно:

```bash
for config in configs/generated_v2/*_finetune.json; do
  python -m project_rank.train --config "$config" || exit 1
done
```

Каждый полученный checkpoint оценивается по тому же шаблону, что и модели
основной сетки:

```bash
python -m project_rank.evaluate \
  --config results/<ИМЯ_FINETUNE_ЗАПУСКА>/config.json \
  --checkpoint results/<ИМЯ_FINETUNE_ЗАПУСКА>/best.pt \
  --split test \
  --output results/<ИМЯ_FINETUNE_ЗАПУСКА>/test.json
```

## 7. Повторные обучения на `p060`

Одношаговая сетка использует `train_seed=1`. Дополнительные `seed2` и `seed3`
нужны для оценки разброса обучения на выбранном бюджете.

Для `baseline`, `erank_per_image`, `erank_pooled` и `uniform`:

```bash
for config in configs/repeats/*.json; do
  if [[ "$(basename "$config")" == "repeat_manifest.json" ]]; then
    continue
  fi
  python -m project_rank.train --config "$config" || exit 1
done
```

Для `l1_weight` и `taylor`:

```bash
for method in l1_weight taylor; do
  for seed in 2 3; do
    python -m project_rank.train \
      --config "configs/generated_v3/p060_${method}_scratch_seed${seed}.json" \
      || exit 1
  done
done
```

Оценка каждого лучшего checkpoint:

```bash
python -m project_rank.evaluate \
  --config results/<ИМЯ_ЗАПУСКА>/config.json \
  --checkpoint results/<ИМЯ_ЗАПУСКА>/best.pt \
  --split test \
  --output results/<ИМЯ_ЗАПУСКА>/test.json
```

## 8. Проверка чувствительности блоков

Каждый остаточный блок по очереди сужается при неизменных остальных блоках.
Дообучение после переноса весов не выполняется.

```bash
python -m project_rank.layer_sensitivity \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output-dir results/layer_sensitivity_seed1 \
  --reduction-ratio 0.25

python -m project_rank.layer_sensitivity \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output-dir results/layer_sensitivity_r050_seed1 \
  --reduction-ratio 0.50
```

## 9. Исследование переноса каналов

Для фиксированных архитектур `p060` и `p030` сравниваются `l1`, `random` и
`erank_subspace`. Эти способы выбирают конкретные переносимые каналы, но не
определяют ширины блоков.

### Шаг 1. Подготовка инициализаций и конфигураций

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

Создаются `configs/channel_transfer_seed1/`, начальные checkpoint и manifest.

### Шаг 2. Оценка до fine-tune

```bash
set -o pipefail
bash configs/channel_transfer_seed1/initial_test_commands.sh \
  2>&1 | tee results/channel_transfer_seed1/initial_test.log
```

### Шаг 3. Fine-tune и итоговая оценка

```bash
set -o pipefail
bash configs/channel_transfer_seed1/train_commands.sh \
  2>&1 | tee results/channel_transfer_seed1/train.log

bash configs/channel_transfer_seed1/test_commands.sh \
  2>&1 | tee results/channel_transfer_seed1/test.log
```

### Шаг 4. Сводка

```bash
python -m project_rank.summarize_channel_transfer \
  --manifest configs/channel_transfer_seed1/manifest.json \
  --output-dir results/channel_transfer_seed1/summary
```

Подробное описание выходных файлов приведено в
[`V12_CHANNEL_TRANSFER_RUNBOOK.md`](V12_CHANNEL_TRANSFER_RUNBOOK.md).

## 10. Итеративный поиск по Effective Rank

На каждом шаге Effective Rank заново рассчитывается на текущей модели,
обученной с нуля. ERank определяет ширины следующей архитектуры. L1 используется
только для выбора конкретных каналов в параллельной ветке fine-tune.

### Переход `p080 → p060`

Если исходный checkpoint отсутствует, сначала обучите одношаговую `p080`:

```bash
python -m project_rank.train \
  --config configs/generated_v4/p080_erank_per_image_scratch_seed1.json
```

Подготовка, обучение и тестирование первого шага:

```bash
python -m project_rank.prepare_iterative_erank \
  --base-config configs/baseline.json \
  --source-checkpoint \
    results/p080_erank_per_image_scratch_seed1/best.pt \
  --target-param-ratio 0.60 \
  --config-dir configs/iterative_erank_seed1/step_p080_to_p060 \
  --output-dir results/iterative_erank_seed1/step_p080_to_p060 \
  --max-samples 5000 \
  --fine-tune-epochs 30 \
  --fine-tune-lr 0.01

bash configs/iterative_erank_seed1/step_p080_to_p060/train_commands.sh
bash configs/iterative_erank_seed1/step_p080_to_p060/test_commands.sh
```

### Переход `p060 → p040`

Источником служит итеративная `p060`, обученная с нуля:

```bash
python -m project_rank.prepare_iterative_erank \
  --base-config configs/baseline.json \
  --source-checkpoint \
    results/iterative_erank_seed1/step_p080_to_p060/scratch/best.pt \
  --target-param-ratio 0.40 \
  --config-dir configs/iterative_erank_seed1/step_p060_to_p040 \
  --output-dir results/iterative_erank_seed1/step_p060_to_p040 \
  --max-samples 5000 \
  --fine-tune-epochs 30 \
  --fine-tune-lr 0.01

bash configs/iterative_erank_seed1/step_p060_to_p040/train_commands.sh
bash configs/iterative_erank_seed1/step_p060_to_p040/test_commands.sh
```

### Переход `p040 → p030`

Источником служит итеративная `p040`, обученная с нуля:

```bash
python -m project_rank.prepare_iterative_erank \
  --base-config configs/baseline.json \
  --source-checkpoint \
    results/iterative_erank_seed1/step_p060_to_p040/scratch/best.pt \
  --target-param-ratio 0.30 \
  --config-dir configs/iterative_erank_seed1/step_p040_to_p030 \
  --output-dir results/iterative_erank_seed1/step_p040_to_p030 \
  --max-samples 5000 \
  --fine-tune-epochs 30 \
  --fine-tune-lr 0.01

bash configs/iterative_erank_seed1/step_p040_to_p030/train_commands.sh
bash configs/iterative_erank_seed1/step_p040_to_p030/test_commands.sh
```

Для каждого перехода сохраняются:

- `manifest.json` — исходные и рассчитанные ширины и бюджет параметров;
- `ranks.csv` — повторно рассчитанный Effective Rank;
- `finetune_l1/test.json` — итог ветки переноса и fine-tune;
- `scratch/test.json` — итог обучения той же архитектуры с нуля.

Подробности приведены в
[`V14_ITERATIVE_ERANK_RUNBOOK.md`](V14_ITERATIVE_ERANK_RUNBOOK.md).

## 11. Измерение MACs, задержки и памяти

Benchmark выполняется на том же GPU и с одинаковыми параметрами для всех
сравниваемых checkpoint. Перед измерением модель прогревается; затем
сохраняются MACs, медианная и p90-задержка, пропускная способность и пиковая
CUDA-память.

Пример для одной модели:

```bash
python -m project_rank.benchmark \
  --checkpoint results/baseline_seed1/best.pt \
  --device cuda \
  --batch-size 128 \
  --warmup 50 \
  --iterations 200 \
  --output results/baseline_seed1/benchmark.json
```

Для воспроизведения набора измерений `p080` и `p060`:

```bash
runs=(
  baseline_seed1
  p080_erank_per_image_scratch_seed1
  p080_erank_pooled_scratch_seed1
  p080_l1_weight_scratch_seed1
  p080_taylor_scratch_seed1
  p080_uniform_scratch_seed1
  p080_random_scratch_seed1
  p060_erank_per_image_scratch_seed1
  p060_erank_pooled_scratch_seed1
  p060_l1_weight_scratch_seed1
  p060_taylor_scratch_seed1
  p060_uniform_scratch_seed1
  p060_random_scratch_seed1
)

for run in "${runs[@]}"; do
  python -m project_rank.benchmark \
    --checkpoint "results/${run}/best.pt" \
    --device cuda \
    --batch-size 128 \
    --warmup 50 \
    --iterations 200 \
    --output "results/${run}/benchmark.json" \
    || exit 1
done
```

Результаты benchmark зависят от устройства, драйвера и текущей нагрузки,
поэтому сравнивать модели следует в рамках одного окружения и одной сессии.

## 12. Формирование сводок и графиков

### Одношаговая сетка

```bash
python -m project_rank.grid_analysis \
  --manifest configs/generated_v4/manifest.json \
  --project-root . \
  --baseline-dir results/baseline_seed1 \
  --output-dir results/summary_v09 \
  --report-data reports/data/grid_results_v09.json
```

Команда создаёт `summary.json`, `method_points.csv`, `summary.md` и набор
данных для графиков одношаговой сетки.

Дополнительная сводка ранних scratch/fine-tune и benchmark-запусков:

```bash
python -m project_rank.summarize_results \
  --results-dir results \
  --output-dir results/summary_v06
```

### Графики

```bash
python -m project_rank.plot_results \
  --data reports/data/final_results.json \
  --output-dir reports/figures

python -m project_rank.plot_grid_results \
  --data reports/data/grid_results_v09.json \
  --output-dir reports/figures
```

`reports/data/final_results.json` — проверенная объединённая сводка
дополнительных экспериментов. Если эксперименты повторяются с другими seed или
гиперпараметрами, этот файл следует обновить соответствующими итоговыми
JSON/CSV перед перестроением графиков отчёта.

## 13. Финальная проверка воспроизводимости

```bash
python -m pytest -q
python -m compileall -q src tests

python -m project_rank.grid_analysis
python -m project_rank.plot_results
python -m project_rank.plot_grid_results
```

Проверка обязательных артефактов:

```bash
test -s reports/REPORT.md
test -s reports/data/final_results.json
test -s reports/data/grid_results_v09.json
test -s results/summary_v09/summary.json

test -s results/iterative_erank_seed1/step_p080_to_p060/ranks.csv
test -s results/iterative_erank_seed1/step_p060_to_p040/ranks.csv
test -s results/iterative_erank_seed1/step_p040_to_p030/ranks.csv

test -s reports/figures/quality_parameters_grid.png
test -s reports/figures/quality_macs_grid.png
test -s reports/figures/channel_transfer.png
test -s reports/figures/iterative_search.png
```

Проверка, что checkpoint не подготовлены к коммиту:

```bash
git diff --cached --name-only \
  | grep -E '\.(pt|pth|ckpt|onnx)$' \
  || true
```

Команда не должна ничего вывести. Численные результаты и их интерпретация не
дублируются в README и должны обновляться только в
[`reports/REPORT.md`](reports/REPORT.md).
