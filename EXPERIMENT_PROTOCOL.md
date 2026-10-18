# Протокол эксперимента

`README.md` намеренно оставлен пустым до завершения проекта. Этот файл — рабочая
инструкция для текущего этапа.

## Что проверяем

Основная гипотеза: эффективный ранг активаций помогает распределить ограниченное
число внутренних каналов между блоками ResNet-20 лучше, чем случайное или
равномерное сужение.

Изменяется только внутренняя ширина остаточного блока:

```text
C_in -> Conv1 -> M -> Conv2 -> C_out
```

`C_in` и `C_out` остаются стандартными, поэтому residual-связи не ломаются.
В полной модели девять значений `M` равны:

```text
[16, 16, 16, 32, 32, 32, 64, 64, 64]
```

## Данные

CIFAR-10 разделяется детерминированно:

- 45 000 изображений — обучение;
- 5 000 изображений — валидация;
- 10 000 официальных test-изображений — финальная независимая оценка.

Архитектура и лучшая эпоха выбираются только по validation. Официальный test не
влияет ни на обучение, ни на выбор checkpoint. Он запускается после выбора
кандидатов и показывает, переносится ли результат на ранее не использованные
изображения.

## Что записывает анализ

Для выхода `Conv1 -> BN -> ReLU` каждого из девяти блоков считаются:

1. эффективный ранг отдельно для каждого изображения, затем среднее;
2. эффективный ранг объединённых и центрированных активаций validation;
3. эффективный ранг матрицы весов `Conv1`.

Для объединённого варианта не сохраняются все активации. По батчам накапливаются
только число наблюдений, сумма векторов и матрица `X^T X`. Поэтому расход памяти
не растёт пропорционально размеру validation.

Для выбора архитектуры сравниваются все три критерия. Если разные критерии после
округления дают одинаковые ширины, повторное обучение такой сети не выполняется:
эквивалентность явно фиксируется в JSON и манифесте экспериментов.

## Как выбираются ширины

ERank задаёт относительное распределение ширины между блоками. Общий коэффициент
масштаба подбирается так, чтобы приблизиться к заданному бюджету параметров.
Ширины ERank- и random-моделей округляются до чётных значений, имеют минимум 4
и не превышают исходную ширину. Для uniform автоматически выбирается наиболее
точный вариант из целочисленной и чётной сеток. Это позволяет одновременно
сохранить почти равномерное сужение и согласовать число параметров. Реальная
скорость в любом случае измеряется отдельно.

После построения ERank-архитектуры создаются два baseline:

- `uniform`: одинаковая относительная степень сужения слоёв;
- `random`: случайное распределение ширин.

Они подбираются к фактическому числу параметров ERank-модели, а не к простой
сумме каналов. JSON содержит точную разницу в процентах.

В версии 0.6.0 дополнительно используются два классических критерия:

- `l1_weight` — сумма модулей весов каждого фильтра `Conv1`;
- `taylor` — модуль произведения активации канала на градиент функции ошибки.

Чтобы оценки из разных слоев можно было сравнивать, в каждом слое они
нормируются по L2-норме. Затем по два канала удаляются там, где отношение
важности к экономии параметров минимально. Для обоих критериев используется
одно и то же правило, поэтому различается только способ оценки важности.

## Перенос весов

Эффективный ранг определяет, сколько внутренних каналов оставить, но не выбирает
их номера. Для первого fine-tune каналы выбираются по L1-норме фильтров `Conv1`.
Те же индексы используются для:

- выходов `Conv1`;
- параметров и статистик `BN1`;
- входных каналов `Conv2`.

Это отдельное техническое правило переноса и не должно интерпретироваться как
часть критерия ERank.

## Этапы запуска

### 1. Проверка окружения и кода

```bash
python -m project_rank.check_environment
pytest -q
python -m project_rank.train --config configs/smoke.json
```

### 2. Обучение полной модели

```bash
python -m project_rank.train --config configs/baseline.json
```

Лучший checkpoint появится в `results/baseline_seed1/best.pt`.

### 3. Анализ эффективного ранга

```bash
python -m project_rank.analyze \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --output-dir results/rank_analysis_seed1 \
  --max-samples 5000 \
  --target-param-ratio 0.60 \
  --random-seed 1001
```

Результаты:

- `ranks.csv` — ранги девяти блоков;
- `architectures.json` — full, ERank, uniform и random.

Если `ranks.csv` уже рассчитан, повторять этот шаг не требуется.

### 4. Построение архитектур для двух бюджетов

```bash
python -m project_rank.build_architectures \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --output results/rank_analysis_seed1/architectures_v2.json \
  --target-param-ratios 0.80 0.60 \
  --random-seed 1001
```

Для каждого бюджета создаются `erank_pooled`, `erank_per_image`,
`erank_weight`, `uniform` и `random`. Основным размером считается фактический
размер `erank_pooled`; отклонения остальных вариантов записываются в процентах.

### 5. Подготовка конфигураций и checkpoint для fine-tune

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

Создаётся `configs/generated_v2/manifest.json`. Для совпавших архитектур в нём
будет одна физическая конфигурация и список эквивалентных названий.

### 6. Обучение моделей бюджета 80%

```bash
python -m project_rank.train --config configs/generated_v2/p080_erank_pooled_scratch.json
python -m project_rank.train --config configs/generated_v2/p080_erank_pooled_finetune.json
python -m project_rank.train --config configs/generated_v2/p080_erank_per_image_scratch.json
python -m project_rank.train --config configs/generated_v2/p080_erank_per_image_finetune.json
python -m project_rank.train --config configs/generated_v2/p080_uniform_scratch.json
python -m project_rank.train --config configs/generated_v2/p080_uniform_finetune.json
python -m project_rank.train --config configs/generated_v2/p080_random_scratch.json
python -m project_rank.train --config configs/generated_v2/p080_random_finetune.json
```

### 7. Обучение моделей бюджета 60%

```bash
python -m project_rank.train --config configs/generated_v2/p060_erank_pooled_scratch.json
python -m project_rank.train --config configs/generated_v2/p060_erank_pooled_finetune.json
python -m project_rank.train --config configs/generated_v2/p060_erank_per_image_scratch.json
python -m project_rank.train --config configs/generated_v2/p060_erank_per_image_finetune.json
python -m project_rank.train --config configs/generated_v2/p060_uniform_scratch.json
python -m project_rank.train --config configs/generated_v2/p060_uniform_finetune.json
python -m project_rank.train --config configs/generated_v2/p060_random_scratch.json
python -m project_rank.train --config configs/generated_v2/p060_random_finetune.json
```

При текущих рангах `erank_weight` совпадает с `erank_pooled`, поэтому отдельные
запуски для него не создаются.

Для всех четырёх уникальных архитектур выполняются и scratch, и fine-tune.
Во всех fine-tune-вариантах конкретные переносимые каналы выбираются одинаково
по L1-норме. Поэтому сравнение изолирует влияние распределения ширины, а не
разницу в способе выбора отдельных каналов.

### 8. Измерение скорости

Измерения следует выполнять отдельно для каждого checkpoint на одном устройстве,
с одинаковым batch size, warmup и числом повторений.

```bash
python -m project_rank.benchmark \
  --checkpoint results/baseline_seed1/best.pt \
  --device cuda \
  --batch-size 128 \
  --warmup 50 \
  --iterations 200
```

### 9. Финальная оценка на официальном test-наборе

Для каждого уже обученного запуска используется сохранённый `best.pt`:

```bash
python -m project_rank.evaluate \
  --config results/baseline_seed1/config.json \
  --checkpoint results/baseline_seed1/best.pt \
  --split test \
  --output results/baseline_seed1/test.json
```

`test.json` содержит loss, accuracy, число изображений и время оценки. Команда
не изменяет веса и ничего не дообучает.

### 10. Повторные запуски для seed 2 и 3

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

Создаются восемь scratch-конфигураций: две полные модели и по две модели для
каждого из трёх методов. Ширины компактных моделей остаются такими же, как в
эксперименте seed 1. Это проверяет устойчивость уже найденных архитектур к
случайной инициализации и порядку обучающих данных, но не повторяет весь поиск
архитектуры с нуля.

### 11. Автоматическая сводка

После обучения и test-оценки:

```bash
python -m project_rank.summarize_results \
  --results-dir results \
  --output-dir results/summary_v04
```

Создаются:

- `summary.json` — полная машиночитаемая сводка;
- `runs.csv` — одна строка на запуск;
- `summary.md` — таблицы отдельных запусков и средних значений по seed.

Если доступны результаты нескольких seed, дополнительно рассчитывается
стандартное отклонение accuracy.

### 12. Расчет `l1_weight` и `taylor`

```bash
python -m project_rank.pruning_criteria \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --output results/rank_analysis_seed1/pruning_scores.json \
  --max-samples 5000
```

`l1_weight` читается непосредственно из весов. Для `taylor` выполняется один
проход по validation с расчетом градиентов. Оптимизатор не запускается, веса
не изменяются, то есть на этом шаге модель не обучается.

### 13. Построение архитектур `l1_weight` и `taylor`

```bash
python -m project_rank.build_pruning_architectures \
  --architectures results/rank_analysis_seed1/architectures_v2.json \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output results/rank_analysis_seed1/architectures_v3.json
```

Файл `architectures_v3.json` сохраняет все прежние архитектуры и добавляет
`l1_weight` и `taylor` для бюджетов `p080` и `p060`.

### 14. Подготовка обучений для классических критериев

```bash
python -m project_rank.prepare_classic_experiments \
  --base-config configs/baseline.json \
  --architectures results/rank_analysis_seed1/architectures_v3.json \
  --output-dir configs/generated_v3
```

Создаются два запуска `p080` для `seed1` и шесть запусков `p060` для `seed1`,
`seed2`, `seed3`. Все восемь моделей обучаются с нуля 160 эпох. Полный список
команд приведен в `V06_RUNBOOK.md`.

### 15. Проверка чувствительности отдельных блоков

```bash
python -m project_rank.layer_sensitivity \
  --config configs/baseline.json \
  --checkpoint results/baseline_seed1/best.pt \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output-dir results/layer_sensitivity_seed1 \
  --reduction-ratio 0.25
```

Каждый из девяти блоков по очереди сужается на 25%, остальные блоки не
меняются. Веса переносятся по одному правилу L1, после чего измеряется падение
validation accuracy без дообучения. Результат показывает, какие блоки
действительно чувствительнее к потере каналов. Дополнительно рассчитывается
корреляция этого падения с ERank, `l1_weight` и `taylor`.

## Ограничения эксперимента

- Выполняется одна итерация поиска архитектуры.
- Основной анализ относится только к ResNet-20 и CIFAR-10.
- Network Slimming и HRank пока не реализованы.
- Эффективный ранг — эвристика. Значение ранга не равно гарантированно
  допустимому числу каналов.
- Повторные seed используют фиксированные архитектуры, найденные по baseline
  seed 1; устойчивость полного процесса поиска пока не проверяется.
- Random представлен одной случайной архитектурой на бюджет. Для статистически
  корректного случайного baseline нужны несколько независимо сгенерированных
  архитектур.

## Итоговые графики и отчёт

Проверенные итоговые числа собраны в `reports/data/final_results.json`. Графики
перестраиваются одной командой:

```bash
python -m project_rank.plot_results \
  --data reports/data/final_results.json \
  --output-dir reports/figures
```

Команда строит шесть графиков, включая кривые качество–размер для всех методов
и анализ чувствительности блоков. Кривая `quality_size_seed1.png` объединяет
результаты `baseline`, `p080` и `p060` для `erank_per_image`, `erank_pooled`,
`uniform`, `random`, `l1_weight` и `taylor`.

Итоговый отчёт находится в `reports/REPORT.md`. Все рисунки, на которые он
ссылается, находятся в каталоге `reports/figures/`.

