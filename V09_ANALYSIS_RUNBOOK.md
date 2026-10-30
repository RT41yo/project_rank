# Сбор сводки и графиков расширенной сетки

Версия 0.9.0 обрабатывает уже завершённые 106 запусков. Повторное обучение и
файлы весов для этого этапа не требуются.

## 1. Проверка окружения

Из корня `project_rank`:

```bash
source .venv/bin/activate

python -m project_rank.check_environment
python -m pytest -q
```

## 2. Проверка манифеста

```bash
python - <<'PY'
import json
from pathlib import Path

data = json.loads(
    Path("configs/generated_v4/manifest.json").read_text(encoding="utf-8")
)
print(json.dumps(data["counts"], ensure_ascii=False, indent=2))
PY
```

Ожидается:

```json
{
  "total": 106,
  "already_complete": 106,
  "needs_training": 0
}
```

## 3. Сбор сводки

```bash
python -m project_rank.grid_analysis \
  --manifest configs/generated_v4/manifest.json \
  --project-root . \
  --baseline-dir results/baseline_seed1 \
  --output-dir results/summary_v09 \
  --report-data reports/data/grid_results_v09.json
```

Ожидается:

```json
{
  "physical_runs": 106,
  "logical_method_points": 112,
  "budgets": 7
}
```

Физических запусков меньше логических точек, потому что совпадающие
архитектуры `erank_pooled` и `erank_weight` не обучаются повторно.

## 4. Построение графиков

```bash
python -m project_rank.plot_results \
  --data reports/data/final_results.json \
  --output-dir reports/figures

python -m project_rank.plot_grid_results \
  --data reports/data/grid_results_v09.json \
  --output-dir reports/figures
```

Должны появиться:

```text
reports/figures/quality_parameters_grid.png
reports/figures/quality_macs_grid.png
reports/figures/random_architectures_grid.png
reports/figures/advantage_vs_random_grid.png
reports/figures/width_profiles_grid.png
reports/figures/winners_grid.png
```

## 5. Финальная проверка

```bash
python -m pytest -q

test -s results/summary_v09/summary.json
test -s results/summary_v09/method_points.csv
test -s results/summary_v09/summary.md
test -s reports/data/grid_results_v09.json
test -s reports/REPORT.md
```

Никакие длительные GPU-команды в версии 0.9.0 запускать не нужно.
