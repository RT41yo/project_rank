# Протокол первого эксперимента

`README.md` намеренно оставлен пустым до завершения проекта. Этот файл — рабочая
инструкция для первого этапа.

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
- официальный test сейчас не используется.

Архитектура выбирается только по validation. Test следует подключить один раз
после окончательного выбора метода и его настроек.

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

## Ограничения текущей версии

- Выполняется одна итерация поиска архитектуры.
- Основной анализ относится только к ResNet-20 и CIFAR-10.
- Пока нет L1-, Network Slimming-, Taylor- и HRank-baseline как отдельных
  способов выбора ширины.
- Эффективный ранг — эвристика. Значение ранга не равно гарантированно
  допустимому числу каналов.
- Для итоговых выводов каждое основное обучение нужно повторить с 3–4 seed.

