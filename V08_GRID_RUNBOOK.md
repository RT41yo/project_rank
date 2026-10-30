# Расширенная одношаговая сетка экспериментов

Эта версия подготавливает первую полную сетку `scratch`-экспериментов для
бюджетов от `p090` до `p030`. Все новые модели сначала обучаются только с
`train_seed=1`. Уже завершённые запуски `p080` и `p060` повторно не запускаются,
если в соответствующем каталоге найден непустой `history.json`.

## 1. Что сравнивается

Для каждого бюджета строятся архитектуры:

- `erank_per_image`;
- `erank_pooled`;
- `erank_weight`;
- `l1_weight`;
- `taylor`;
- `uniform`;
- десять независимых `random`-архитектур с `architecture_seed=1001...1010`.

Если два детерминированных метода дают одинаковые девять ширин, создаётся один
физический запуск, а второй метод записывается в `aliases` манифеста.

## 2. Проверка после распаковки

Из корня `project_rank`:

```bash
source .venv/bin/activate
python -m project_rank.check_environment
python -m pytest -q
python -m project_rank.train --config configs/smoke.json
```

## 3. Построение ERank-, uniform- и служебных random-архитектур

Повторно вычислять Effective Rank не требуется. Используется готовый файл
`results/rank_analysis_seed1/ranks.csv`.

```bash
python -m project_rank.build_architectures \
  --ranks results/rank_analysis_seed1/ranks.csv \
  --output results/rank_analysis_seed1/architectures_grid_v4_base.json \
  --target-param-ratios 0.90 0.80 0.70 0.60 0.50 0.40 0.30 \
  --random-seed 1001
```

Случайная архитектура, которая попадает в этот промежуточный JSON, не является
полным random-baseline. Десять воспроизводимых random-профилей на каждый бюджет
создаются на шаге 5.

## 4. Добавление `l1_weight` и `taylor`

Повторно вычислять оценки каналов также не требуется. Используется готовый файл
`results/rank_analysis_seed1/pruning_scores.json`.

```bash
python -m project_rank.build_pruning_architectures \
  --architectures results/rank_analysis_seed1/architectures_grid_v4_base.json \
  --scores results/rank_analysis_seed1/pruning_scores.json \
  --output results/rank_analysis_seed1/architectures_grid_v4.json
```

## 5. Подготовка конфигураций полной сетки

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

Создаются:

- `configs/generated_v4/manifest.json` — полная машиночитаемая сетка;
- отдельный JSON для каждого уникального запуска;
- `configs/generated_v4/train_commands.txt` — только ещё не выполненные
  обучения;
- `configs/generated_v4/test_commands.txt` — test-оценка всех точек сетки.

В манифесте три разных вида случайности не смешиваются:

- `train_seed` — обучение модели;
- `architecture_seed` — случайный профиль девяти ширин;
- `mask_seed` появится позднее в отдельном эксперименте переноса каналов.

## 6. Проверка подготовленной сетки без обучения

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("configs/generated_v4/manifest.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(data["counts"], ensure_ascii=False, indent=2))

for item in data["experiments"]:
    if item["needs_training"]:
        print(item["run"])
PY
```

Дополнительно проверьте первые команды:

```bash
sed -n '1,30p' configs/generated_v4/train_commands.txt
```

На этом месте рекомендуется остановиться, сохранить вывод `counts` и проверить
сетку перед длительным обучением.

Для snapshot, на котором подготовлена версия 0.8.0, ожидается:

```json
{
  "total": 106,
  "already_complete": 12,
  "needs_training": 94
}
```

Число меньше прямого произведения «7 бюджетов × 16 вариантов», потому что
совпадающие профили `erank_pooled` и `erank_weight` обучаются один раз. В 12
готовых точек входят существующие детерминированные запуски `p080`/`p060` и по
одной совпавшей random-архитектуре этих бюджетов.

## 7. Запуск длительного обучения

Только после проверки манифеста:

```bash
bash configs/generated_v4/train_commands.txt \
  2>&1 | tee results/grid_v4_train.log
```

Команды выполняются последовательно и останавливаются при первой ошибке. Уже
завершённые каталоги результатов сохраняются. При повторной подготовке манифеста
непустые `history.json` будут отмечены как завершённые.

## 8. Test-оценка

После завершения всех обучений снова выполните шаг 5, чтобы обновить флаги
`needs_training`, а затем запустите:

```bash
bash configs/generated_v4/test_commands.txt \
  2>&1 | tee results/grid_v4_test.log
```

Официальный test-набор используется только после фиксации всей сетки и не
участвует в выборе ширин или лучшей эпохи.

## 9. Что прислать после запусков

Новый ZIP-snapshot должен содержать:

- `results/rank_analysis_seed1/architectures_grid_v4_base.json`;
- `results/rank_analysis_seed1/architectures_grid_v4.json`;
- `configs/generated_v4/`;
- для каждого нового запуска: `config.json`, `environment.json`,
  `history.json`, `test.json`;
- лог `results/grid_v4_train.log`, если он не слишком большой;
- лог `results/grid_v4_test.log`, если он не слишком большой.

Файлы `best.pt`, `last.pt`, CIFAR-10, `.venv`, `.git` и кэши в архив не
включаются.

После получения результатов будут построены:

1. кривые `accuracy — параметры`;
2. кривые `accuracy — MACs`;
3. полоса разброса десяти random-архитектур;
4. тепловая карта профилей ширин;
5. таблица смены лидирующего метода по бюджетам.
