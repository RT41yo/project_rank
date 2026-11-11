# План экспериментов: Effective Rank для ViT-Tiny/CIFAR

## 1. Цель и границы

Перенести методологию ResNet-20 коллеги на ViT-Tiny/CIFAR-10:
проверить выбор внутренних ширин по ERank, отдельно выбор конкретных
нейронов при переносе весов и отдельно scratch против fine-tune.
Основные источники: `EXPERIMENT_PROTOCOL.md`, `reports/REPORT.md`,
`V08_GRID_RUNBOOK.md`, `V12_CHANNEL_TRANSFER_RUNBOOK.md`,
`V14_ITERATIVE_ERANK_RUNBOOK.md` и соответствующий код.

Ближайшая цель — корректная ViT baseline и законченный опыт p060.
NAS, CIFAR-100, HRank, Network Slimming, SVD и LM не входят в ближайший этап.
Их отсутствие не мешает воспроизвести фактически выполненную ResNet-серию.

## 2. Неизменяемая методология

| Компонент | Правило |
|---|---|
| Данные | CIFAR-10: 45 000 train / 5 000 validation / 10 000 test |
| Критерии | По 5 000 validation-изображений полной baseline seed1 |
| Основная сетка | Только scratch, единый рецепт, train_seed=1 |
| ERank | erank_per_image, erank_pooled, erank_weight |
| Сравнения | l1_weight, taylor, uniform, random |
| Random | 10 профилей на бюджет, architecture_seed=1001…1010 |
| Повторения | Фиксированные архитектуры p060, train_seed=2,3 |
| Checkpoint | Первый максимум validation accuracy, как в train.py |
| Fine-tune | Отдельный опыт; не смешивать с основной scratch-сеткой |
| Перенос | L1 / random / erank_subspace при одинаковых ширинах |
| Итерации | Пересчёт ERank на scratch-ветке; ширины не увеличиваются |
| Test | Только оценка заранее зафиксированного опыта; не настройка |
| Совпадения | Одна архитектура обучается один раз, методы — aliases |
| Отчёт | Воспроизводимые JSON/CSV, таблицы и графики без GPU |

Один train_seed не доказывает превосходство. Разброс архитектур, обучения
и случайных масок — разные величины.

## 3. Исходная модель и пространство поиска

ViT-Tiny/CIFAR: изображение 32×32, patch=4, 64 patch-токена + CLS,
embed_dim D=192, depth=12, heads=3 (head_dim=64), Pre-LayerNorm,
GELU, классификация по CLS, MLP=768 во всех блоках. Bias включены.
Нет внешних pretrained-весов, dropout, attention dropout и stochastic depth.
LayerNorm eps=1e-6. Linear/patch/position/CLS веса инициализируются
truncated normal std=0.02, bias нулём, LayerNorm weight=1/bias=0.

Изменяем только M_i в `D → Linear1 → M_i → GELU → Linear2 → D`.
Это аналог внутренней ширины `Conv1 → M → Conv2` у коллеги.
Residual-размерность, attention, embeddings, глубина и токены фиксированы.
Сжатие физическое: меньшие Linear, а не только маски нулей.

Для этой спецификации:

```
P(M) = 1 814 602 + 385 * sum(M_i)
P_full = 5 362 762
P_min(M_i=4) = 1 833 082 = 34.18% full
```

Подтвердить формулу фактическими параметрами до обучения.
Бюджет p030 недостижим при MLP-only. Сетка:
`p090 p080 p070 p060 p050 p040 p035`.
p090…p040 общие с ResNet; p035 — отдельная сильная компрессия ViT,
не эквивалент p030. Бюджет — доля ВСЕХ параметров исходной модели.
Ширины: чётные, минимум 4, максимум исходная ширина блока.
Integer-grid uniform допускается, как у коллеги.

## 4. Определения критериев

Активация — выход GELU между Linear1 и Linear2. Форма B×65×M_i.
Для ERank/ковариаций исключаем CLS: остаются 64 patch-токена.
ERank: p_j = sigma_j / sum(sigma), exp(-sum(p_j * log(p_j))).
Нулевой спектр даёт 0 по соглашению существующего кода.

- **per_image:** центрировать [64,M_i] по токенам для каждого изображения,
  вычислить ERank, затем среднее ГОТОВЫХ рангов по изображениям.
- **pooled:** все изображения/patch-токены, общее центрирование.
  Потоково накапливать s=sum(x), S=sum(xx^T), n; G=S-ss^T/n.
  Получить sigma через sqrt(eigvalsh(G)), затем энтропию.
  Усреднение отдельно центрированных per-image ковариаций НЕ эквивалентно.
- **weight:** ERank матрицы Linear1 [M_i,D], без центрирования.

Статистики float64; inference в eval без autocast, как у коллеги.
Не хранить активации всей выборки. Оптимизация через малую Gram-матрицу
допустима после сравнения с прямым SVD на тестах.

Пределы: per_image ≤ min(63,M_i), weight ≤ min(192,M_i).
Это не готовые значения ширин! Ширины определяет общий бюджет:
`M_i = clip(round_to_2(alpha * rank_i), 4, source_width_i)`.
Alpha выбирается по близости полного числа параметров к бюджету.
Диапазон alpha расширять автоматически: старых [0.05,16] может не хватить.
Нефинитные/нулевые оценки не исправлять произвольно, останавливать поиск
для диагностики. На графиках показывать ранги и их отношения к ширинам.

**L1:** сумма модулей строки Linear1.weight, без bias и Linear2.
**Taylor:** mean_b(abs(sum_t(A_btj * dL_b/dA_btj))), ВСЕ 65 токенов,
включая CLS (нейрон удаляется всюду). Loss reduction=sum, optimizer не шагает.
Оба критерия L2-нормируются внутри блока. Группы по 2 удаляются по
минимальной важности на сэкономленный параметр, как в ResNet.

### Отличие статьи

https://arxiv.org/html/2607.19315v2 использует собственные значения
ковариации, то есть спектр sigma², и исследует отбор изображений.
Код коллеги использует sigma. В основной серии сохраняем sigma;
sigma² — только отдельная подписанная абляция, не подмена определения.

## 5. Контрольные сравнения и требование руководителя

Не смешиваем два random:

1. **Random architecture:** случайный профиль 12 ширин при общем бюджете;
   сравнение полезности найденной архитектуры в scratch-режиме.
2. **Random mask:** те же ширины в КАЖДОМ блоке, случайный выбор нейронов;
   сравнение с erank_subspace после переноса и fine-tune.

Второй опыт буквально обеспечивает одинаковое число каналов в вариантах
random и ERank. Сам скаляр ERank не определяет маску.
При одинаковой архитектуре scratch не зависит от маски прежних весов —
не обучать её повторно под видом другого способа переноса.

Random-профили: исходный best-of-20 000 sampler плохо попадает в крайние
бюджеты ViT. Документируемая адаптация — равномерный выбор допустимого
профиля при фиксированной сумме ширин (условное распределение на бюджет).
Это изменение реализации сэмплера, не выдавать за буквальный старый sampler.
Контроли привязываются к фактическому бюджету erank_pooled, как у коллеги.
Проверять отклонения до обучения, рабочий предел 0.5% от опорного бюджета;
при превышении остановиться, а не скрывать разницу размеров.

## 6. Рецепт обучения и случайность

Основной режим сохраняет настройки коллеги:

| Параметр | Scratch | Fine-tune |
|---|---:|---:|
| Epochs | 160 | 30 |
| Optimizer | SGD + Nesterov | SGD + Nesterov |
| LR | 0.1 | 0.01 |
| Momentum | 0.9 | 0.9 |
| Weight decay (все параметры) | 0.0005 | 0.0005 |
| Scheduler | CosineAnnealing | CosineAnnealing |
| Batch size | 128 | 128 |
| AMP | На CUDA | На CUDA |

Train: RandomCrop(32,padding=4), RandomHorizontalFlip, ToTensor,
существующая CIFAR-10 normalization. Validation/test без аугментаций.
Не добавлять Mixup, CutMix, label smoothing, distillation, EMA, RandAugment.

SGD-рецепт ResNet не гарантирует качественный ViT. Baseline — обязательный
gate перед сеткой: конечные loss/градиенты, снижение train loss,
содержательная val accuracy. Предварительный инженерный порог — 80%
validation accuracy к 160 эпохам, не целевое качество исследования.
Если обучение неудачно: остановка, отдельный pilot AdamW+warmup,
согласование новой версии протокола и применение ко ВСЕМ ViT-методам.
Не смешивать рецепты на одной кривой и не подбирать их по test.

Различаем train_seed, architecture_seed, mask_seed. Совместимый режим
повторяет существующее поведение data.py: train_seed меняет и разбиение.
Это не чистая оценка разброса инициализации. Фиксированное разбиение
для разных train_seed — отдельное усиление, не смешиваемое с этой серией.

## 7. Реализация и совместимость (P0)

- `models/vit_tiny.py`: полная модель и неравномерные MLP-ширины.
- `models/factory.py`: создание из config/checkpoint; старые checkpoint
  без типа модели остаются ResNet20.
- `vit_spec.py`: чистый аналитический подсчёт параметров и MACs.
- `effective_rank.py`: поддержка B×T×M, политика CLS, общий ERank-core.
- `pruning_criteria.py`, `transfer.py`: отдельные ViT-функции/dispatch,
  без изменения смысла старых ResNet-функций.
- `train.py`, `evaluate.py`, `analyze.py`, `benchmark.py`: model factory.
- Генераторы архитектур/сетки, sensitivity, iterative, сводки и графики:
  поддержка спецификации модели и произвольного числа блоков вместо 9.
- Переиспользовать общие алгоритмы, не копировать весь проект независимо.
- ResNet-данные/конфигурации/ожидаемые результаты не перезаписывать.

Каталоги: `configs/vit/`, `results/vit/`, `reports/data/vit/`,
`reports/figures/vit/`, отчёт `reports/VIT_REPORT.md`.

Тесты: forward/backward full/narrow; точный param count; MACs включая
attention; ERank известных спектров; streamed vs direct pooled;
исключение CLS; воспроизводимые random; перенос full сохраняет logits;
compact равен masked source; критерии не меняют веса; итерации не расширяют
блоки; synthetic E2E; вся существующая ResNet test-suite.

На каждом запуске сохранять config/environment/history/best/last,
полную модельную спецификацию, commit, protocol_id, seeds, число параметров,
время; после итоговой оценки test.json. Перенос дополнительно сохраняет
source checkpoint, способ и индексы нейронов.
resume_from сейчас загружает только веса: это не полноценный restart.
Непустой history.json не означает завершённость: сверять число эпох,
checkpoint и конфигурацию, не пропускать прерванные обучения.

## 8. Порядок запусков и условия перехода

### A. Окружение и smoke

1. Записать commit/окружение, проверить PyTorch/CUDA для RTX 5070 Ti.
2. GPU forward/backward и AMP; проверить batch=128, VRAM, finite loss.
3. Подобрать только инфраструктурные workers/pin_memory.
4. Unit/regression tests и synthetic E2E.
5. Overfit небольшого фиксированного набора без аугментаций.
Gate: корректные формы, перенос и критерии; нет NaN/скрытого CPU fallback.

### B. Baseline (P1)

Full [768]*12, seed1, 160 эпох. Проверить кривые обучения, время эпохи,
VRAM и gate качества из раздела 6. Сохранить best.pt как единственный
источник первой одношаговой серии. Не настраивать рецепт по test.

### C. Критерии

На baseline best.pt собрать 3 ERank + L1/Taylor по 5 000 validation.
Сохранить ranks.csv, pruning_scores.json, analysis_metadata.json
(checkpoint, изображения, токены, CLS, центрирование, sigma, dtype).
Gate: все 12 блоков, конечные оценки, соблюдены теоретические границы.

### D. Первый законченный опыт p060 (P1)

1. Scratch 160: erank_per_image и random_arch1001. Сравнить с baseline.
2. Фиксировать профиль erank_per_image. Перенести L1, random(mask2001),
   erank_subspace; каждый fine-tune 30 эпох. Параметры/ширины идентичны.
3. Сохранить качество до/после переноса и scratch той же архитектуры.
4. Проверка/отладка — validation; test только для фиксированного опыта.
Итого ближайший набор: 3 scratch (включая baseline) + 3 fine-tune.

### E. Полная сетка (P2)

7 бюджетов × (6 deterministic + 10 random) = 112 логических точек.
Физических запусков меньше при aliases, но НЕ обязательно 106, как у ResNet.
Все scratch seed1, единый рецепт. Сначала проверить manifest:
спецификация, ширины, параметры/MACs, отклонения, уникальные random,
aliases, пути вне ResNet. Обучать последовательно, пропускать только
полностью завершённые совпадающие конфигурации. Не перезапускать опыт D.

### F. Повторения (P2)

seed2,3 для baseline и p060 erank_per_image, erank_pooled, L1, Taylor,
uniform. Архитектуры фиксированы по seed1, ранги не пересчитывать.
Рекомендуемое усиление: также random_arch1001. Сводить mean/sample std,
не смешивать разброс профилей, обучения и масок.

### G. Scratch против fine-tune (P3)

p080/p060: erank_per_image, erank_pooled, uniform, random_arch1001.
Всем один L1-перенос + 30 эпох. Scratch уже есть. Переиспользовать
p060 erank_per_image/L1 из D. L1 — техническое правило переноса,
не критерий ERank-поиска; не объявлять его заранее лучшим для ViT.

Перенос: Linear1 строки+bias по индексам, Linear2 столбцы по тем же
индексам, Linear2.bias целиком. Остальные параметры целиком.
Fine-tune наследует стоимость baseline: не объявлять равенство compute
между 30-эпоховым fine-tune и 160-эпоховым scratch.

### H. Перенос при сильном сжатии (P3)

p035 erank_per_image: L1/random/subspace + scratch. Subspace — точный
аналог transfer.py: ведущие k eigenvectors covariance, score_j = сумма
квадратов координат j в этих vectors, top-k нейронов. Не добавлять веса
eigenvalues без отдельной абляции. Измерять initial, fine-tuned, scratch.
Все начальные/конечные test-оценки только после фиксации опыта.

### I. Чувствительность (P3)

По очереди сузить один MLP: 768→576 (-25%) и 768→384 (-50%).
24 варианта + baseline. Остальные блоки полные, L1 transfer, без обучения.
Validation: accuracy drop/loss/параметры. Spearman с относительным ERank
и долей удалённых L1/Taylor scores, как в layer_sensitivity.py.
Локальная устойчивость не равна качеству всей scratch-архитектуры.

### J. Итерации (P4)

Цепочка p080→p060→p040→p035. Источник каждого шага — best.pt предыдущей
SCRATCH-ветки; не выбирать между scratch/finetune по качеству.
Новый ERank по тем же 5 000 validation, criterion=erank_per_image,
бюджет относительно исходной полной ViT, target_width_i≤source_width_i.
Две одинаковые архитектуры: scratch160, L1-finetune30. Сравнение с
одношаговой архитектурой того же бюджета, с реальными параметрами.
Сохранять lineage/ranks/ширины/обе ветки. Устойчивость относительного
профиля не доказывает фиксированную точку при постоянном бюджете;
строгая проверка сходимости — отдельный будущий эксперимент.

## 9. Динамика рангов

Дополнение к основной методологии: checkpoints baseline на эпохах
0,1,10,40,80,120,160 и best. После обучения, offline, ранги на фиксированных
1 000 validation-изображениях. Так диагностика не меняет training RNG.
Основной поиск по-прежнему использует best.pt и все 5 000 изображений.
График: эпоха × блок, все варианты ERank, ограничения по числу токенов.

## 10. Метрики и benchmark

Accuracy/loss, точные параметры, MACs, latency median/p90, throughput,
peak allocated CUDA memory, training/analysis time.
MACs включают patch embedding, QKV, proj, QK^T, AV, обе MLP и head.
Для блока: 4*T*D² + 2*T²*D + 2*T*D*M_i.
Нормализации/softmax/GELU/residual-add не включены, FLOPs≈2*MACs.
Полная модель: 365 107 584 MACs/image (подтвердить независимым счётчиком).
При фиксированных D/T параметры и MACs почти определяются sum(M_i),
в отличие от разноразрешённых блоков ResNet. Latency всё равно зависит
от форм матриц и измеряется отдельно.

Обязательные benchmark: baseline, методы p080/p060, p035.
Основной режим: FP32, eval/inference_mode, batch128, warmup50,
iterations200, CUDA synchronize, без параллельного обучения.
Дополнительно batch1/FP16 отдельно. Не смешивать precision/backend.
Все ViT benchmark на одной машине; одинаковая RTX не гарантирует равную
задержку разных ПК. Сравнивать speedup относительно локальной baseline.

## 11. Отчёт и готовность

Таблицы: baseline; методы×бюджеты; random mean/std/min/max; p060 repeats;
scratch vs FT; masks fixed-width; sensitivity; iterative; latency/memory.
Графики: accuracy–params/MACs/latency, распределение random,
advantage vs random, MLP profiles, ERank по блокам/эпохам, transfer,
one-shot vs iterative. Данные JSON/CSV, графики воспроизводимы без весов.

Gate завершения: manifest согласован с результатами, все эпохи завершены,
test содержит 10 000 изображений, параметры/MACs подтверждены,
aliases не считаются независимыми, виды случайности разделены,
описаны p035, MLP-only, спектр sigma и ограничения рецепта/seed.
Git: код, конфиги, небольшие JSON/CSV/отчёт; НЕ веса, CIFAR, .venv, кэши.
Для results/vit продумать исключения .gitignore, не раскрывать все веса.

## 12. Приоритеты и управление затратами

P0: реализация/тесты/E2E → P1: baseline/критерии/p060 → P2: сетка/repeats
→ P3: остальной transfer/sensitivity → P4: iterative/benchmark/report.
Не запускать >100 обучений сразу. После первых запусков оценить затраты:
sum(epochs * measured_epoch_time) + analysis/evaluation time.
Проверять свободный диск, сохранять логи, не терять исходные checkpoint.
LM отложить до завершения ViT; затем возможен тот же MLP-only принцип
в маленьком GPT с фиксированными attention/embeddings и отдельным протоколом
текстовых split, токенизации и perplexity.

## 13. Журнал начала реализации

- [x] План сохранён, команды/ограничения — в `VIT_RUNBOOK.md`.
- [x] ViT, factory и совместимость старых ResNet-checkpoint.
- [x] Подтверждены 5 362 762 параметра и 365 107 584 MACs полной модели.
- [x] Токеновые ERank, ковариации, L1/Taylor, физический перенос нейронов.
- [x] Synthetic train/evaluate/ranks/benchmark и transfer/fine-tune в тестах.
- [x] 49 unit/regression tests; `git diff --check` без ошибок.
- [x] GPU AMP full ViT batch128: finite gradients, peak allocated ≈748 MiB.
- [x] Диагностический overfit full ViT на 16 random-примерах: 100%,
  loss=0.0442 за 157 шагов (не CIFAR-результат).
- [x] Baseline config, защита от перезаписи ViT, rank-checkpoints 0…160.
- [x] Полная baseline CIFAR: 160 эпох завершены, best validation 76.10%
  на эпохе 158. Checkpoint и история сохранены, test не оценивался.
- [ ] Gate качества: НЕ пройден (порог 80%). Массовая сетка остановлена
  до согласования отдельного pilot-рецепта ViT (например, AdamW+warmup).
  Исходный SGD-опыт не перезаписывать и не смешивать с новым рецептом.
- [x] ViT scaling по всему alpha-пути и равномерный budget-conditioned random.
  Подготовлено превью p060: 3 ERank, uniform, 10 уникальных random-профилей.
- [x] Training manifest и полный synthetic E2E пятизапускового p060 stage D,
  проверка завершённости/хешей, запрет перезаписи partial runs.
- [ ] L1/Taylor-архитектуры и полная одношаговая grid-серия.
- [ ] Остальные научные серии D–J и отчёт.

Проверенное локальное окружение: Python 3.12.3, torch 2.15.0+cu132,
torchvision 0.29.2+cu132, numpy 2.5.4, CUDA runtime 13.2, cuDNN 92600,
RTX 5070 Ti. Это фактические версии, не требование заменить окружение коллеги.

### Согласованный следующий pilot

После неудачного gate SGD пользователь согласовал отдельную проверку
AdamW+warmup. Конфиг `configs/vit/baseline_adamw_pilot.json`, protocol_id
`vit_cifar10_adamw_pilot_v1`, новый каталог результатов, обучение с нуля.
Peak LR=3e-4, betas=(0.9,0.999), eps=1e-8, WD=0.05 (без bias/LN/CLS/pos),
warmup=5 эпох от factor=0.01, далее cosine; прочие условия неизменны.
Подробные команды и семантика LR — в `VIT_RUNBOOK.md`.
Рецепт пока pilot, а не автоматически принятый основной протокол.

Pilot завершён: 160 эпох, best validation accuracy 82.76% на эпохе 152
(SGD: 76.10%, разница +6.66 п.п.). Технический gate 80% пройден.
Train accuracy ≈99.99%, best validation loss=1.1878 против 0.6790 у SGD:
есть признаки переобучения; улучшение accuracy не означает улучшения loss
или доказанной калибровки. Test не оценивался. Сводка:
`reports/data/vit/baseline_adamw_pilot_v1.json`.
Следующий шаг — критерии по validation на pilot best.pt; отдельные пути
`results/vit/rank_analysis_adamw_pilot_seed1/`, без смешивания с SGD.

Критерии завершены: 5 000 изображений, 12 блоков, finite ranks и L1/Taylor.
Ранги опубликованы в `reports/data/vit/rank_analysis_adamw_pilot_seed1/`.
Превью p060 — `reports/data/vit/architectures_p060_preview.json`.
Все ERank и random-профили: 3 217 542 параметра; uniform отличается на
0.0479%. На этапе превью сжатые модели ещё не обучались. Промежуточный отчёт:
`reports/VIT_REPORT.md`. После добавления builder/sampler проходят 65 тестов.

### Переход к stage D

По запросу пользователя начат следующий этап. Для него зафиксирован
`vit_cifar10_adamw_v1` (без изменения успешного pilot-рецепта scratch).
Baseline — тот же pilot best.pt с сохранённым историческим protocol_id.
Пять запусков: ERank/random_arch1001 scratch160 и три fixed-width переноса
L1/random2001/subspace fine-tune30. Fine-tune LR=3e-5, warmup5, WD и
исключения те же; scratch LR=3e-4. Три маски получены из ОДНОЙ baseline,
а не из результатов новых scratch-моделей. Все параметры/MACs совпадают.
Полный synthetic E2E и regression suite: 69 тестов проходят.
Конфиги и неизменяемый manifest: `configs/vit/p060_stage_d_seed1/`.
Test до итоговой фиксации серии не оценивается.

Stage D завершён: обе scratch-ветки прошли 160 эпох, три fine-tune — 30.
Validation: ERank scratch 82.66%, random architecture 82.64%, L1 FT 81.38%,
random mask FT 81.32%, subspace FT 82.44%. Параметры/MACs идентичны.
Разница scratch 0.02 п.п. — одно изображение, не доказательство превосходства.
Test не использовался. Проверяемые публичные данные:
`reports/data/vit/p060_stage_d_validation.json` и `.csv`.
