# KD and HPO recipes

Experiments on NATS-Bench architectures with shared CIFAR-10 infrastructure
and a FLOPs budget.

```text
src/
├── common/  # datasets, FLOPs, NATS models, optimizers and utilities
├── kd/      # fixed-hyperparameter/KD experiment pipeline
└── hpo/     # layered ASHA-based hyperparameter optimization
    ├── domain/          # configuration and ASHA rules
    ├── application/     # experiment and architecture use cases
    ├── infrastructure/  # training, checkpoints and GPU workers
    └── reporting/       # result tables and plots
```

- `src/common/` is shared by both experiments.
- `src/kd/` and `src/hpo/` are independent experiment packages.
- `notebooks/kd1.ipynb` runs the fixed-hyperparameter baseline.
- `notebooks/hpo_flops_budget.ipynb` runs HPO over learning rate and weight
  decay.
- `notebooks/hpo_flops_budget.py` runs the same experiment from a terminal and
  reports its progress to the console.
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
python notebooks/hpo_flops_budget.py
```

Choose architecture rows explicitly or run all rows:

```shell
python notebooks/hpo_flops_budget.py --arch-rows 0 1 2
python notebooks/hpo_flops_budget.py --arch-rows
```

Use both GPUs and train four models concurrently on each GPU:

```shell
python notebooks/hpo_flops_budget.py --gpu-ids 0 1 --workers-per-gpu 4
```

