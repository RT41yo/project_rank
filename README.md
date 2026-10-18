# Effective Rank Architecture Search для ResNet-20

Проект исследует, насколько Effective Rank полезен для выбора компактной
архитектуры нейронной сети по сравнению с классическими критериями прунинга.
Эксперименты выполнены на ResNet-20 и наборе данных CIFAR-10.

Сравниваются `erank_per_image`, `erank_pooled`, `erank_weight`, `l1_weight`,
`taylor`, `uniform` и `random`. Используются два бюджета: `p080` сохраняет
около 80% параметров полной модели, а `p060` - около 60%.

Полное описание методики, таблицы, графики и выводы находятся в
[`reports/REPORT.md`](reports/REPORT.md).

## 1. Экспериментальная инфраструктура

### 1.1. Аппаратное обеспечение

| Компонент | Конфигурация |
|---|---|
| Процессор | Intel Core Ultra 7 265KF |
| Видеокарта | NVIDIA GeForce RTX 5070 Ti |
| Память видеокарты | 16 ГБ |
| Оперативная память | 64 ГБ |
| Накопитель | NVMe SSD |

GPU использовался для обучения, вычисления `taylor`, анализа активаций,
оценки моделей и измерения производительности. Синтетическая быстрая проверка
выполняется на CPU.

### 1.2. Программное окружение

| Компонент | Версия или требование |
|---|---|
| Операционная система | Manjaro Linux |
| Ядро Linux | 6.6.156-2-MANJARO |
| Python | 3.14.7 |
| PyTorch | 2.14.1+cu130 |
| CUDA runtime PyTorch | 13.0 |
| cuDNN | 92400, значение `torch.backends.cudnn.version()` |
| `torchvision` | не ниже 0.17 |
| `numpy` | не ниже 1.26 |
| `matplotlib` | не ниже 3.8 |
| `pytest` | не ниже 8.0 |

Точная информация о PyTorch, CUDA и GPU сохраняется в `environment.json`
каждого запуска. Проверить текущее окружение можно командой:

```bash
python -m project_rank.check_environment
```

Ожидаемые ключевые признаки исходного GPU-окружения:

```text
"cuda_available": true
"python_torch": "2.14.1+cu130"
"cuda_runtime": "13.0"
"gpu": "NVIDIA GeForce RTX 5070 Ti"
```

Использование другой современной NVIDIA GPU допускается, но абсолютные
значения задержки, пропускной способности и памяти будут отличаться.

## 2. Структура проекта

```text
project_rank/
├── configs/                 конфигурации обучения
│   ├── baseline.json
│   ├── smoke.json
│   ├── generated_v2/        ERank, uniform, random, scratch и finetune
│   ├── generated_v3/        l1_weight и taylor
│   └── repeats/             повторные запуски seed2 и seed3
├── reports/
│   ├── REPORT.md            итоговый отчет
│   ├── data/                проверенные данные для графиков
│   └── figures/             итоговые графики
├── results/                 результаты запусков
├── src/project_rank/        исходный код
└── tests/                   автоматические тесты
```

Каталоги `data/`, `.venv/`, файлы весов `*.pt` и другие тяжелые локальные
артефакты не хранятся в Git.

## 3. Подготовка окружения

Все команды ниже выполняются из корня `project_rank`.

### 3.1. Получение проекта

```bash
git clone <URL_РЕПОЗИТОРИЯ> project_rank
cd project_rank
git switch roman_feature
```

### 3.2. Создание виртуального окружения

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Если PyTorch для нужной версии CUDA устанавливается отдельно, сначала нужно
установить совместимые `torch` и `torchvision`, а затем выполнить:

```bash
python -m pip install -e ".[dev]" --no-deps
python -m pip install numpy matplotlib pytest
```

### 3.3. Первичная проверка

```bash
python -m project_rank.check_environment
pytest -q
python -m project_rank.train --config configs/smoke.json
```

Быстрая проверка использует синтетические данные, длится одну эпоху и не
загружает CIFAR-10.

## 4. Данные и общий протокол обучения

CIFAR-10 загружается автоматически при первом запуске в каталог `data/`.
Официальные 50 000 обучающих изображений разделяются следующим образом:

- 45 000 изображений - обучение;
- 5 000 изображений - проверка и выбор лучшей эпохи;
- 10 000 официальных тестовых изображений - финальная оценка.

| Параметр | Значение |
|---|---:|
| Эпохи `scratch` | 160 |
| Эпохи `finetune` | 30 |
| Batch size | 128 |
| Оптимизатор | SGD с momentum и Nesterov |
| Начальный `learning rate` для `scratch` | 0,1 |
| Начальный `learning rate` для `finetune` | 0,01 |
| Momentum | 0,9 |
| Weight decay | 0,0005 |
| План изменения `learning rate` | CosineAnnealing |
| AMP | включен на CUDA |
| Основные `seed` | 1, 2, 3 |

Для каждого запуска сохраняются:

- `config.json` - фактическая конфигурация;
- `environment.json` - сведения об окружении;
- `history.json` - история обучения;
- `best.pt` - модель с лучшей `validation accuracy`;
- `last.pt` - состояние после последней эпохи;
- `test.json` - результат на официальной тестовой выборке;
- `benchmark.json` - измерения производительности, если они выполнялись.

## 5. Полное воспроизведение эксперимента

Полный цикл содержит много независимых обучений. Команды рекомендуется
выполнять последовательно и после каждого этапа проверять созданные файлы.

### Шаг 1. Обучение полной ResNet-20

```bash
python -m project_rank.train --config configs/baseline.json
```

Основной результат: `results/baseline_seed1/best.pt`.

### Шаг 2. Расчет Effective Rank

```bash
python -m project_rank.analyze \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --output-dir results/rank_analysis_seed1 \
  --max-samples 5000 \
  --target-param-ratio 0.60 \
  --random-seed 1001
```

Создаются `ranks.csv` и первоначальный набор архитектур.

### Шаг 3. Построение архитектур `p080` и `p060`

```bash
python -m project_rank.build_architectures \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --output results/rank_analysis_seed1/architectures_v2.json \
  --target-param-ratios 0.80 0.60 \
  --random-seed 1001
```

Строятся `erank_per_image`, `erank_pooled`, `erank_weight`, `uniform` и
`random`. Если разные критерии дают одинаковые ширины, одна и та же сеть
повторно не обучается. В данном эксперименте `erank_weight` совпал с
`erank_pooled`.

### Шаг 4. Подготовка `scratch` и `finetune`

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

Конкретные каналы для `finetune` переносятся из `baseline` по L1-норме
фильтров.

### Шаг 5. Обучение основных компактных моделей

```bash
for config in configs/generated_v2/*.json; do
  if [[ "$(basename "$config")" == "manifest.json" ]]; then
    continue
  fi
  python -m project_rank.train --config "$config" || exit 1
done
```

Запускаются модели обоих бюджетов в режимах `scratch` и `finetune`.

### Шаг 6. Повторные запуски `seed2` и `seed3`

```bash
python -m project_rank.prepare_repeats \
  --base-config configs/baseline.json \
  --manifest configs/generated_v2/manifest.json \
  --project-root . \
  --output-dir configs/repeats \
  --budget p060 \
  --methods erank_pooled erank_per_image uniform \
  --seeds 2 3
```

```bash
for config in configs/repeats/*.json; do
  if [[ "$(basename "$config")" == "repeat_manifest.json" ]]; then
    continue
  fi
  python -m project_rank.train --config "$config" || exit 1
done
```

Повторно обучаются `baseline`, `erank_per_image`, `erank_pooled` и `uniform`.
Ширины компактных моделей остаются фиксированными.

### Шаг 7. Расчет `l1_weight` и `taylor`

```bash
python -m project_rank.pruning_criteria \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --output results/rank_analysis_seed1/pruning_scores.json \
  --max-samples 5000
```

`l1_weight` рассчитывается по весам. Для `taylor` выполняется прямой и обратный
проход по проверочной выборке, но веса модели не обновляются.

### Шаг 8. Архитектуры классических критериев

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

### Шаг 9. Обучение `l1_weight` и `taylor`

```bash
for config in configs/generated_v3/*.json; do
  if [[ "$(basename "$config")" == "manifest.json" ]]; then
    continue
  fi
  python -m project_rank.train --config "$config" || exit 1
done
```

Для каждого критерия выполняется один запуск `p080` и три запуска `p060`.
Все эти модели обучаются с нуля 160 эпох.

### Шаг 10. Анализ чувствительности блоков

```bash
python -m project_rank.layer_sensitivity \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output-dir results/layer_sensitivity_seed1 \
  --reduction-ratio 0.25
```

Каждый из девяти блоков по очереди сужается на 25%. Дообучение на этом шаге не
выполняется.

### Шаг 11. Оценка всех моделей на тестовой выборке

```bash
for checkpoint in results/*/best.pt; do
  run_dir="$(dirname "$checkpoint")"
  python -m project_rank.evaluate \
    --config "$run_dir/config.json" \
    --checkpoint "$checkpoint" \
    --split test \
    --output "$run_dir/test.json" || exit 1
done
```

Тестовая выборка используется только после завершения обучения и выбора
`best.pt` по проверочной выборке.

### Шаг 12. Измерение производительности

Benchmark выполняется для `baseline` и всех уникальных `scratch`-архитектур
`seed1`:

```bash
benchmark_runs=(
  baseline_seed1
  p080_erank_per_image_scratch_seed1
  p080_erank_pooled_scratch_seed1
  p080_uniform_scratch_seed1
  p080_random_scratch_seed1
  p080_l1_weight_scratch_seed1
  p080_taylor_scratch_seed1
  p060_erank_per_image_scratch_seed1
  p060_erank_pooled_scratch_seed1
  p060_uniform_scratch_seed1
  p060_random_scratch_seed1
  p060_l1_weight_scratch_seed1
  p060_taylor_scratch_seed1
)

for run in "${benchmark_runs[@]}"; do
  python -m project_rank.benchmark \
    --checkpoint "results/$run/best.pt" \
    --device cuda \
    --batch-size 128 \
    --warmup 50 \
    --iterations 200 \
    --output "results/$run/benchmark.json" || exit 1
done
```

Для корректного сравнения все измерения необходимо выполнять на одном GPU без
параллельной нагрузки.

### Шаг 13. Формирование сводки

```bash
python -m project_rank.summarize_results \
  --results-dir results \
  --output-dir results/summary_v06
```

Создаются `runs.csv`, `summary.json` и `summary.md`.

### Шаг 14. Построение итоговых графиков

```bash
python -m project_rank.plot_results \
  --data reports/data/final_results.json \
  --output-dir reports/figures
```

Команда создает:

- `accuracy_p060.png`;
- `accuracy_by_seed.png`;
- `resources_p060.png`;
- `width_profiles_p060.png`;
- `quality_size_seed1.png`;
- `layer_sensitivity.png`.

`reports/data/final_results.json` содержит проверенные числа, перенесенные из
файлов отдельных запусков. После нового полного эксперимента сначала следует
проверить `summary_v06`, `test.json` и `benchmark.json`, а затем обновлять этот
файл.

## 6. Быстрое воспроизведение только отчета

Если обучение уже выполнено и `reports/data/final_results.json` присутствует,
повторно обучать модели не требуется:

```bash
source .venv/bin/activate

python -m project_rank.plot_results \
  --data reports/data/final_results.json \
  --output-dir reports/figures

pytest -q tests/test_plot_results.py
```

Итоговый текст находится в `reports/REPORT.md`.

## 7. Проверка результата

```bash
pytest -q

test -f results/baseline_seed1/best.pt
test -f results/rank_analysis_seed1/ranks.csv
test -f results/rank_analysis_seed1/pruning_scores.json
test -f results/rank_analysis_seed1/architectures_v3.json
test -f results/layer_sensitivity_seed1/sensitivity.json
test -f results/summary_v06/summary.json
test -f reports/data/final_results.json
test -f reports/figures/quality_size_seed1.png
test -f reports/figures/layer_sensitivity.png

echo "Основные результаты и итоговые графики присутствуют"
```

## 8. Воспроизводимость и ограничения

- Для инициализации моделей, разбиения обучающей части CIFAR-10 и порядка
  пакетов используются фиксированные `seed`.
- Повторные запуски используют фиксированные архитектуры, найденные по
  `baseline_seed1`.
- GPU-операции могут иметь небольшую недетерминированность, поэтому последние
  знаки точности могут отличаться.
- Значения задержки и пропускной способности зависят от GPU, версии драйвера,
  CUDA и фоновой нагрузки.
- Результаты относятся к ResNet-20 на CIFAR-10 и не переносятся автоматически
  на другие модели и наборы данных.
- Файлы `best.pt` и `last.pt` необходимы для полного локального
  воспроизведения, но не публикуются в Git из-за размера.

## 9. Основной результат

При сохранении около 60% параметров лучшую среднюю тестовую точность показал
`taylor`: 91,65 ± 0,23%. `erank_per_image` достиг 91,46 ± 0,34%, а
`l1_weight` - 91,43 ± 0,18%. Различия между этими методами невелики.

Effective Rank оказался полезен как способ построения конкурентоспособной
компактной архитектуры. При почти одинаковом числе параметров
`erank_per_image` требует меньше MACs, чем `l1_weight` и `taylor`, однако
`taylor` лучше согласуется с измеренной чувствительностью отдельных блоков.
