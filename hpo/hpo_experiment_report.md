# FLOPs-Budgeted HPO Report For Architecture 0

## 1. Цель эксперимента

Цель эксперимента - подобрать гиперпараметры обучения для одной фиксированной NATS-Bench архитектуры при ограниченном FLOPs-бюджете.

В этом запуске рассматривается только первая архитектура из `sampled_architectures_10.jsonl`:

```python
ARCH_ROWS = [0]
```

Формулировка задачи:

> Для архитектуры `arch_row = 0` найти такие `learning rate` и `weight decay`, которые дают наилучшую validation accuracy, не превышая заданный FLOPs-бюджет.

Эксперимент реализован в ноутбуке:

```text
hpo/hpo_flops_budget_first_arch.ipynb
```

Фокус эксперимента - HPO. Knowledge distillation здесь не рассматривается.

## 2. Архитектура

Используется первая архитектура из файла:

```text
hpo/sampled_architectures_10.jsonl
```

Параметры архитектуры:

```text
arch_row:    0
arch_index:  13197
dataset:     cifar10-valid
search_space: tss
```

Описание архитектуры:

```text
|nor_conv_3x3~0|+|avg_pool_3x3~0|avg_pool_3x3~1|+|avg_pool_3x3~0|nor_conv_1x1~1|none~2|
```

Модель создается через `xautodl`:

```python
config = dict2config(
    {
        "name": "infer.tiny",
        "C": 16,
        "N": 5,
        "arch_str": arch_str,
        "num_classes": 10,
    },
    None,
)

model = get_cell_based_tiny_net(config)
```

## 3. Dataset

Используется CIFAR-10.

Train split CIFAR-10 содержит:

```text
50 000 images
```

Он делится на train и validation:

```python
VALIDATION_FRACTION = 0.1
```

Итого:

```text
train:      45 000 images
validation: 5 000 images
```

Test split используется только после завершения HPO:

```text
test: 10 000 images
```

Validation accuracy используется для выбора лучшего trial. Test accuracy используется только как финальная оценка выбранной модели.

## 4. FLOPs-Бюджет Для Архитектуры 0

FLOPs-бюджет на архитектуру:

```python
BUDGET_FLOPS_PER_ARCH = 1.0e15
```

Стоимость архитектуры берется из:

```text
hpo/sampled_architecture_costs.csv
```

Для `arch_row = 0`:

```text
forward_flops_per_sample = 47 104 650
raw_flops_m              = 47.10465
params                   = 0.344346 M
latency                  = 0.01593890629316631
```

Стоимость одной train epoch:

```python
epoch_flops = 3 * forward_flops_per_sample * N_TRAIN_EXAMPLES
```

Для первой архитектуры:

```text
epoch_flops = 3 * 47 104 650 * 45 000
            = 6 359 127 750 000
            ~= 6.36e12 FLOPs
```

Стоимость одной validation evaluation:

```python
validation_flops = forward_flops_per_sample * N_VALIDATION_EXAMPLES
```

Для первой архитектуры:

```text
validation_flops = 47 104 650 * 5 000
                 = 235 523 250 000
                 ~= 2.36e11 FLOPs
```

Стоимость одного минимального стартового stage при `TARGET_MIN_EPOCHS = 3`:

```text
min_stage_flops = 3 * epoch_flops + validation_flops
                = 19 312 906 500 000
                ~= 1.93e13 FLOPs
```

FLOPs-бюджет включает:

```text
train FLOPs
validation FLOPs
```

Финальная test evaluation считается отдельно и сохраняется как `test_flops`.

## 5. HPO Search Space

В эксперименте подбираются:

```text
learning rate
weight decay
```

Оба параметра сэмплируются динамически:

```python
LR_RANGE = (1e-3, 3e-1)
WEIGHT_DECAY_RANGE = (1e-6, 1e-3)
```

Для каждого trial:

```text
lr ~ log_uniform(1e-3, 3e-1)
weight_decay ~ log_uniform(1e-6, 1e-3)
```

`log_uniform` используется потому, что для `lr` и `weight_decay` важен порядок величины.

Optimizer:

```python
torch.optim.SGD(
    model.parameters(),
    lr=config.lr,
    momentum=0.9,
    weight_decay=config.weight_decay,
)
```

Scheduler:

```text
cosine
```

## 6. ASHA-План

Для распределения FLOPs-бюджета используется ASHA-style Successive Halving.

Основные параметры:

```python
TARGET_MIN_EPOCHS = 3
TARGET_REDUCTION_FACTOR = 3
```

Rungs строятся динамически:

```text
3 -> 9 -> 27 -> 81 -> ...
```

Следующий rung добавляется только если он помещается в FLOPs-бюджет.

Число стартовых configs считается по стоимости минимального stage:

```python
num_initial_configs = budget_flops // min_stage_flops
```

Для первой архитектуры без дополнительного ограничения сверху:

```text
num_initial_configs ~= floor(1.0e15 / 1.93e13)
                    = 51
```

Если в ноутбуке задано:

```python
MAX_INITIAL_CONFIGS = 20
```

то стартовых configs будет не больше 20.

Пример ASHA-плана для `MAX_INITIAL_CONFIGS = 20`:

```text
configs=20
rungs=[3, 9, 27]
planned_train_epochs=156
```

Расчет train epochs:

```text
20 configs до 3 эпох:        20 * 3       = 60
7 configs до 9 эпох:          7 * (9-3)   = 42
3 configs до 27 эпох:         3 * (27-9)  = 54

Итого: 60 + 42 + 54 = 156 train epochs
```

Если budget не позволяет добавить следующий rung `81`, ASHA останавливается на `27`.

## 7. Процедура Запуска

Для первой архитектуры выполняется следующий процесс:

1. Загружается `arch_row = 0`.
2. Из `arch_str` создается PyTorch-модель.
3. Из CSV берется `forward_flops_per_sample`.
4. Считаются `epoch_flops` и `validation_flops`.
5. По FLOPs-бюджету строится ASHA-план.
6. Сэмплируются HPO configs: `lr`, `weight_decay`.
7. Каждый config обучается до target epoch текущего rung.
8. После каждого stage считается validation accuracy.
9. Стоимость stage списывается из FLOPs-бюджета.
10. ASHA оставляет лучшие configs.
11. Лучшие configs продолжают обучение на следующих rungs.
12. После завершения выбирается stage с максимальной `val_acc1`.
13. Checkpoint выбранного stage оценивается на test set.

## 8. Результаты

Полные результаты сохраняются в:

```text
hpo_output/hpo_real_training_results.csv
```

Summary сохраняется в:

```text
hpo_output/hpo_real_training_summary.csv
```

Для запуска одной архитектуры summary должен содержать одну строку:

```text
arch_row = 0
```

Основные поля summary:

```text
arch_row
arch_index
trial_id
target_epochs
lr
weight_decay
val_acc1
test_acc1
spent_flops
spent_train_flops
spent_validation_flops
test_flops
spent_budget_ratio
```

Ключевые метрики:

```text
val_acc1  - accuracy, по которой выбран лучший HPO trial
test_acc1 - финальная test accuracy выбранной модели
spent_flops - FLOPs, потраченные на train + validation во время HPO
```

## 9. Интерпретация Для Первой Архитектуры

Эксперимент показывает, насколько хорошо можно обучить именно архитектуру:

```text
arch_row = 0
arch_index = 13197
```

при фиксированном FLOPs-бюджете:

```text
1.0e15 FLOPs
```

Если стартовых configs много, HPO шире исследует пространство `lr` и `weight_decay`, но максимальное число эпох для лучших configs может быть меньше.

Если стартовых configs меньше, поиск уже, но лучшие configs могут дойти до более глубоких rungs.

Этот trade-off напрямую контролируется:

```python
MAX_INITIAL_CONFIGS
MAX_EPOCHS_CAP
```

## 10. Итоговая Формулировка

В эксперименте проводится FLOPs-budgeted HPO для первой NATS-Bench архитектуры из `sampled_architectures_10.jsonl`. Архитектура имеет `arch_row = 0`, `arch_index = 13197` и forward cost `47.10465M FLOPs` на одно изображение. HPO подбирает `learning rate` и `weight decay`, используя log-uniform sampling. Для распределения бюджета используется ASHA: сначала запускаются несколько configs на малое число эпох, затем слабые configs отсекаются, а лучшие продолжают обучение. FLOPs-бюджет включает train и validation вычисления. Лучшая модель выбирается по validation accuracy и затем один раз оценивается на CIFAR-10 test set.

