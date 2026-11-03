# Первый итеративный поиск по Effective Rank

## Что проверяем

В одношаговом опыте архитектура `p060` строилась сразу по полной ResNet-20.
Теперь проверяется другой путь:

1. полная модель была уменьшена до `p080` по `erank_per_image`;
2. модель `p080` была обучена с нуля;
3. Effective Rank измеряется заново уже на обученной `p080`;
4. по новым значениям ERank строится архитектура `p060`;
5. новая `p060` дообучается после переноса весов и отдельно обучается с нуля.

Итоговую итеративную `p060` нужно сравнить с уже готовой одношаговой `p060`.
Если итеративный вариант точнее при примерно том же числе параметров, повторный
расчёт Effective Rank оказался полезен.

L1 не определяет ширины блоков. Ширины выбирает `erank_per_image`. L1 нужен
только для выбора конкретных каналов, которые копируются в ветку fine-tune.

## 1. Проверка исходного checkpoint

Нужен результат уже выполненного обучения `p080 erank_per_image` с нуля:

```bash
test -s results/p080_erank_per_image_scratch_seed1/best.pt
```

Если команда ничего не вывела и завершилась без ошибки, файл найден. Если файла
нет, сначала выполните:

```bash
python -m project_rank.train \
  --config configs/generated_v4/p080_erank_per_image_scratch_seed1.json
```

## 2. Повторный расчёт ERank и подготовка `p060`

```bash
python -m project_rank.prepare_iterative_erank \
  --base-config configs/baseline.json \
  --source-checkpoint results/p080_erank_per_image_scratch_seed1/best.pt \
  --target-param-ratio 0.60 \
  --config-dir configs/iterative_erank_seed1/step_p080_to_p060 \
  --output-dir results/iterative_erank_seed1/step_p080_to_p060 \
  --max-samples 5000 \
  --fine-tune-epochs 30 \
  --fine-tune-lr 0.01
```

Эта команда ещё не обучает модели. Она:

- измеряет ERank на модели `p080`;
- рассчитывает новые ширины `p060`;
- создаёт L1-инициализацию только для fine-tune;
- создаёт конфигурации для fine-tune и обучения с нуля;
- сохраняет `manifest.json` с исходными и новыми ширинами.

Проверьте рассчитанные ширины:

```bash
cat results/iterative_erank_seed1/step_p080_to_p060/manifest.json
```

## 3. Обучение двух одинаковых архитектур

Сначала короткая ветка fine-tune, затем полное обучение с нуля:

```bash
bash configs/iterative_erank_seed1/step_p080_to_p060/train_commands.sh
```

При необходимости команды можно запускать отдельно:

```bash
python -m project_rank.train \
  --config configs/iterative_erank_seed1/step_p080_to_p060/p060_finetune_l1.json

python -m project_rank.train \
  --config configs/iterative_erank_seed1/step_p080_to_p060/p060_scratch.json
```

## 4. Итоговая оценка на test

```bash
bash configs/iterative_erank_seed1/step_p080_to_p060/test_commands.sh
```

Нужные результаты появятся в файлах:

```text
results/iterative_erank_seed1/step_p080_to_p060/finetune_l1/test.json
results/iterative_erank_seed1/step_p080_to_p060/scratch/test.json
```

После получения этих двух файлов их нужно добавить в отчёт и сравнить с
одношаговым `p060_erank_per_image_scratch_seed1`.

## 5. Полученный результат `p080 → p060`

Итеративная архитектура:

```text
[10, 14, 14, 28, 30, 28, 36, 36, 26]
```

Результаты на test:

| Вариант | Scratch | Fine-tune |
|---|---:|---:|
| Одношаговый `p060` | 91,33% | 91,70% |
| Итеративный `p060` | 91,53% | 91,71% |

## 6. Следующий переход `p060 → p040`

Источником служит итеративная `p060`, обученная с нуля. Effective Rank снова
измеряется на 5 000 проверочных изображениях:

```bash
python -m project_rank.prepare_iterative_erank \
  --base-config configs/baseline.json \
  --source-checkpoint results/iterative_erank_seed1/step_p080_to_p060/scratch/best.pt \
  --target-param-ratio 0.40 \
  --config-dir configs/iterative_erank_seed1/step_p060_to_p040 \
  --output-dir results/iterative_erank_seed1/step_p060_to_p040 \
  --max-samples 5000 \
  --fine-tune-epochs 30 \
  --fine-tune-lr 0.01
```

Перед обучением нужно проверить новый manifest:

```bash
cat results/iterative_erank_seed1/step_p060_to_p040/manifest.json
```

После проверки запускаются две ветки:

```bash
bash configs/iterative_erank_seed1/step_p060_to_p040/train_commands.sh
bash configs/iterative_erank_seed1/step_p060_to_p040/test_commands.sh
```

## 7. Полученный результат `p060 → p040`

Итеративная архитектура:

```text
[8, 10, 10, 20, 22, 20, 24, 22, 14]
```

Результаты на test:

| Вариант | Scratch | Fine-tune |
|---|---:|---:|
| Одношаговый `p040` | 90,33% | — |
| Итеративный `p040` | 91,00% | 90,73% |

## 8. Последний переход `p040 → p030`

Источником служит итеративная `p040`, обученная с нуля:

```bash
python -m project_rank.prepare_iterative_erank \
  --base-config configs/baseline.json \
  --source-checkpoint results/iterative_erank_seed1/step_p060_to_p040/scratch/best.pt \
  --target-param-ratio 0.30 \
  --config-dir configs/iterative_erank_seed1/step_p040_to_p030 \
  --output-dir results/iterative_erank_seed1/step_p040_to_p030 \
  --max-samples 5000 \
  --fine-tune-epochs 30 \
  --fine-tune-lr 0.01
```

Перед обучением нужно проверить новый manifest:

```bash
cat results/iterative_erank_seed1/step_p040_to_p030/manifest.json
```

После проверки:

```bash
bash configs/iterative_erank_seed1/step_p040_to_p030/train_commands.sh
bash configs/iterative_erank_seed1/step_p040_to_p030/test_commands.sh
```

## 9. Полученный результат `p040 → p030`

Итеративная архитектура:

```text
[6, 8, 8, 14, 18, 16, 18, 16, 10]
```

Результаты на test:

| Вариант | Scratch | Fine-tune L1 |
|---|---:|---:|
| Одношаговый `p030` | 89,97% | 89,73% |
| Итеративный `p030` | 90,14% | 89,83% |

Основная итеративная цепочка `p080 → p060 → p040 → p030` завершена.
