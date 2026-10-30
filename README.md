# Поиск архитектуры ResNet-20 через Effective Rank

Проект проверяет, помогает ли Effective Rank выбирать компактную архитектуру
ResNet-20 лучше классических критериев прунинга и простых контрольных методов.
Эксперименты выполнены на CIFAR-10.

Сравниваются:

- `erank_per_image`, `erank_pooled`, `erank_weight`;
- `l1_weight`, `taylor`;
- `uniform`, `random`.

Основная одношаговая сетка содержит семь бюджетов параметров — от `p090` до
`p030`. Для `random` на каждом бюджете используются десять разных профилей
ширин. Итоговый отчёт: [`reports/REPORT.md`](reports/REPORT.md).

## 1. Экспериментальная инфраструктура

### Аппаратное обеспечение

| Компонент | Конфигурация |
|---|---|
| Процессор | Intel Core Ultra 7 265KF |
| Видеокарта | NVIDIA GeForce RTX 5070 Ti |
| Память видеокарты | 16 ГБ |
| Оперативная память | 64 ГБ |
| Накопитель | NVMe SSD |

GPU использовался для обучения, расчёта активаций и `taylor`, а также для
оценки моделей. MACs в итоговой сводке рассчитываются точно по архитектуре и
не требуют GPU или файлов весов.

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

Фактическое окружение каждого запуска сохраняется в `environment.json`.

## 2. Структура проекта

```text
project_rank/
├── configs/
│   ├── baseline.json              полная ResNet-20
│   ├── smoke.json                 быстрая синтетическая проверка
│   ├── generated_v2/              ранние scratch/finetune-эксперименты
│   ├── generated_v3/              повторения l1_weight и taylor
│   ├── generated_v4/              расширенная одношаговая сетка
│   └── repeats/                   повторения p060 и baseline
├── reports/
│   ├── REPORT.md                  итоговый отчёт
│   ├── data/                      проверенные данные отчёта
│   └── figures/                   воспроизводимые графики
├── results/
│   ├── rank_analysis_seed1/       ранги, оценки и архитектуры
│   └── summary_v09/               сводка расширенной сетки
├── src/project_rank/              исходный код
└── tests/                         автоматические тесты
```

Каталоги `.git/`, `.venv/`, `data/`, локальные веса `*.pt` и кэши не должны
попадать в ZIP-snapshot или Git.

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

Если PyTorch для нужной версии CUDA устанавливается отдельно:

```bash
python -m pip install -e ".[dev]" --no-deps
python -m pip install numpy matplotlib pytest
```

Проверка окружения и кода:

```bash
python -m project_rank.check_environment
python -m pytest -q
python -m project_rank.train --config configs/smoke.json
```

Smoke-проверка использует синтетические данные и не загружает CIFAR-10.

## 4. Данные и протокол обучения

CIFAR-10 автоматически загружается в корневой каталог `data/`. Официальные
50 000 обучающих изображений делятся на 45 000 для обучения и 5 000 для
проверки. Официальные 10 000 тестовых изображений используются только после
обучения и выбора лучшей эпохи.

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

## 5. Полное воспроизведение одношаговой сетки

### Шаг 1. Обучение полной модели

```bash
python -m project_rank.train --config configs/baseline.json
```

Основной checkpoint: `results/baseline_seed1/best.pt`.

### Шаг 2. Расчёт вариантов Effective Rank

```bash
python -m project_rank.analyze \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --output-dir results/rank_analysis_seed1 \
  --max-samples 5000 \
  --target-param-ratio 0.60 \
  --random-seed 1001
```

Основной результат этого шага — `results/rank_analysis_seed1/ranks.csv`.

### Шаг 3. Расчёт `l1_weight` и `taylor`

```bash
python -m project_rank.pruning_criteria \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --output results/rank_analysis_seed1/pruning_scores.json \
  --max-samples 5000
```

Для `taylor` выполняется прямой и обратный проход, но веса не обновляются.

### Шаг 4. Построение архитектур семи бюджетов

```bash
python -m project_rank.build_architectures \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --output results/rank_analysis_seed1/architectures_grid_v4_base.json \
  --target-param-ratios 0.90 0.80 0.70 0.60 0.50 0.40 0.30 \
  --random-seed 1001
```

```bash
python -m project_rank.build_pruning_architectures \
  --architectures results/rank_analysis_seed1/architectures_grid_v4_base.json \
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

Создаются 106 физических запусков. Логических точек сравнения 112, потому что
совпадающая архитектура может относиться к двум критериям через `aliases`, но
повторно не обучается.

### Шаг 6. Длительное обучение

```bash
set -o pipefail

bash configs/generated_v4/train_commands.txt \
  2>&1 | tee results/grid_v4_train.log
```

Команды выполняются последовательно и останавливаются при первой ошибке.

После обучения повторно выполните шаг 5. Ожидаемое состояние манифеста:

```json
{
  "total": 106,
  "already_complete": 106,
  "needs_training": 0
}
```

### Шаг 7. Оценка на тестовой выборке

```bash
set -o pipefail

bash configs/generated_v4/test_commands.txt \
  2>&1 | tee results/grid_v4_test.log
```

Каждый `test.json` должен содержать результат по 10 000 изображениям.

## 6. Повторные обучения на p060

Расширенная сетка строит кривые при `train_seed=1`. Для оценки разброса
обучения на `p060` используются дополнительные `seed2` и `seed3`.

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

Каждый лучший checkpoint оценивается командой:

```bash
python -m project_rank.evaluate \
  --config results/<ИМЯ_ЗАПУСКА>/config.json \
  --checkpoint results/<ИМЯ_ЗАПУСКА>/best.pt \
  --split test \
  --output results/<ИМЯ_ЗАПУСКА>/test.json
```

## 7. Проверка чувствительности блоков

```bash
python -m project_rank.layer_sensitivity \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output-dir results/layer_sensitivity_seed1 \
  --reduction-ratio 0.25
```

Каждый блок по очереди уменьшается на 25%. Дообучение не выполняется.

## 8. Формирование сводки и графиков

Повторное обучение не требуется:

```bash
python -m project_rank.grid_analysis \
  --manifest configs/generated_v4/manifest.json \
  --project-root . \
  --baseline-dir results/baseline_seed1 \
  --output-dir results/summary_v09 \
  --report-data reports/data/grid_results_v09.json
```

Создаются:

- `results/summary_v09/summary.json`;
- `results/summary_v09/method_points.csv`;
- `results/summary_v09/summary.md`;
- `reports/data/grid_results_v09.json`.

Графики:

```bash
python -m project_rank.plot_results \
  --data reports/data/final_results.json \
  --output-dir reports/figures

python -m project_rank.plot_grid_results \
  --data reports/data/grid_results_v09.json \
  --output-dir reports/figures
```

Команда строит:

- кривую точность–параметры;
- кривую точность–MACs;
- распределение десяти `random`-архитектур;
- тепловую карту преимущества над средним `random`;
- тепловые карты девяти ширин;
- график победителей по бюджетам.

## 9. Финальная проверка

```bash
python -m pytest -q

python -m project_rank.grid_analysis
python -m project_rank.plot_results \
  --data reports/data/final_results.json \
  --output-dir reports/figures
python -m project_rank.plot_grid_results

test -f reports/REPORT.md
test -f reports/data/grid_results_v09.json
test -f results/summary_v09/summary.json
test -f reports/figures/quality_parameters_grid.png
test -f reports/figures/quality_macs_grid.png
```

Ожидаемая сводка:

```json
{
  "physical_runs": 106,
  "logical_method_points": 112,
  "budgets": 7
}
```

## 10. Основные результаты

- универсального лучшего критерия для всех бюджетов нет;
- `l1_weight` выигрывает на `p090` и `p080`;
- `erank_pooled` выигрывает на `p070`;
- `taylor` выигрывает на `p060` и `p050`;
- `uniform` выигрывает на `p040`;
- `erank_per_image` выигрывает на `p030` и превосходит средний `random` на
  всех семи бюджетах;
- при одинаковом числе параметров MACs могут отличаться более чем в два раза.

Подробная интерпретация, таблицы и ограничения приведены в
[`reports/REPORT.md`](reports/REPORT.md).
