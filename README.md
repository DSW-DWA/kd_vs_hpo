# KD and HPO recipes

Experiments on NATS-Bench architectures with shared CIFAR-10 infrastructure
and detailed FLOPs accounting.

```text
src/
├── common/  # datasets, FLOPs, NATS models, optimizers and utilities
├── kd/      # fixed-hyperparameter/KD experiment pipeline
└── hpo/     # Optuna hyperparameter optimization
    ├── domain/          # configuration and stopping rules
    ├── application/     # experiment and architecture use cases
    ├── infrastructure/  # training, checkpoints and GPU workers
    └── reporting/       # result tables and plots
```

- `src/common/` is shared by both experiments.
- `src/kd/` and `src/hpo/` are independent experiment packages.
- `notebooks/kd1.ipynb` runs the fixed-hyperparameter baseline.
- `notebooks/hpo_optuna.py` compares Optuna samplers with Successive Halving
  and Hyperband pruners.
- `experiments/` contains architecture definitions and their measured costs.

## Installing

Python 3.11 is required.

```shell
make install
```

Run notebooks from the repository root so their relative paths resolve
correctly.

The HPO pipeline writes CSV results to `hpo_output/` and automatically creates
interactive, self-contained HTML charts in `hpo_output/plots/`. Set
`HPOExperimentConfig(generate_plots=False)` to disable chart generation.

Run the terminal version from the repository root:

```shell
python notebooks/hpo_optuna.py
```

Choose architecture rows explicitly or run all rows:

```shell
python notebooks/hpo_optuna.py --arch-rows 0 1 2
python notebooks/hpo_optuna.py --arch-rows
```

Configure trial count and the accuracy-growth stopping threshold:

```shell
python notebooks/hpo_optuna.py --n-trials 20 --lambda-growth 0.05
```

Run four study processes on CPU:

```shell
python notebooks/hpo_optuna.py --device cpu --processes 4
```

Distribute four study processes across two GPUs:

```shell
python notebooks/hpo_optuna.py --device cuda --processes 4 --gpu-ids 0 1
```

Parallelism is study-level: every process trains one independent
`architecture × sampler × pruner` study at a time. GPU-specific process pools
keep the configured number of workers assigned to each GPU. Each study process
also creates the number of DataLoader workers configured by `TrainConfig`.

For each architecture the experiment compares TPE, Grid, CMA-ES and GP samplers
with only two pruners: Successive Halving and Hyperband. A training trial stops
after warmup when the best validation accuracy growth over the patience window
is below `lambda-growth`, or when accuracy does not grow at all. Pruners may
discard unpromising trials earlier.

Every run writes merged structured events to `hpo_output/logs/events.jsonl`.
Raw per-process logs are retained under `hpo_output/logs/runs/<run_id>/`.
Each record contains a UTC timestamp and run ID. Logged events cover experiment,
dataset, architecture, study, trial, epoch, checkpoint, pruning, early stopping,
test evaluation, table export, plots, timing and failures. Epoch records include
loss, validation accuracy, best accuracy, accuracy growth, learning rates,
pruner decision, train/validation duration and peak GPU memory. The file is
flushed after every event so completed records survive an interrupted run.

The same events are printed to the console with process names while the run is
active. FLOPs are estimated per epoch as
`train_step_multiplier × forward_flops × train_samples` for training and
`forward_flops × validation_samples` for validation. Epoch, trial, study and
whole-experiment cumulative FLOPs are included in JSONL and CSV tables.

