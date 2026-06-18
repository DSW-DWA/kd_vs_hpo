# KD and HPO recipes

Experiments on NATS-Bench architectures with shared CIFAR-10 infrastructure
and a FLOPs budget.

```text
kd_vs_hpo/
├── common/  # datasets, FLOPs, NATS models, optimizers and utilities
├── kd/      # fixed-hyperparameter/KD experiment pipeline
└── hpo/     # ASHA-based hyperparameter optimization
```

- `kd_vs_hpo/common/` is shared by both experiments.
- `kd_vs_hpo/kd/` and `kd_vs_hpo/hpo/` are independent experiment packages.
- `notebooks/kd1.ipynb` runs the fixed-hyperparameter baseline.
- `notebooks/hpo_flops_budget.ipynb` runs HPO over learning rate and weight
  decay.
- `experiments/` contains architecture definitions and their measured costs.

## Installing

```
make install
```

Run notebooks from the repository root so their relative paths resolve
correctly.
