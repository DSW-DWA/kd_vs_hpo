# HPO experiments

Optuna HPO experiments for NATS-Bench architectures on CIFAR-10.

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```shell
make install
source .venv/bin/activate
```

Run all commands from the project root.

## Running HPO

```shell
python notebooks/hpo_optuna.py [arguments]
```

Each architecture, sampler, and pruner combination runs as a separate Optuna
study.

- `--arch-rows ROW [ROW ...]` — architecture row numbers. The default is `0`;
  passing the flag without values selects all architectures.
- `--n-trials N` — maximum number of trials per study. The default is `20`.
  The default GridSampler search space requires at least `9`.
- `--max-epochs N` — maximum number of training epochs per trial. The default
  is `200`. Successive Halving and Hyperband require at least `10`.
- `--output-dir PATH` — output directory. The default is `hpo_output`.
- `--samplers NAME [NAME ...]` — samplers to run. Accepted values are `tpe`,
  `grid`, `cmaes`, and `gp`; all four are enabled by default.
- `--pruners NAME [NAME ...]` — pruners to run. Accepted values are `none`,
  `successive_halving`, and `hyperband`; `successive_halving` and `hyperband`
  are enabled by default.
- `--device DEVICE` — compute device. Accepted values are `auto`, `cpu`,
  `cuda`, and `mps`; the default is `auto`. Automatic selection prefers CUDA,
  then MPS, then CPU.
- `--processes N` — number of studies to run in parallel. The default is `1`.
- `--gpu-ids ID [ID ...]` — CUDA device IDs used by worker processes. When
  omitted, all available CUDA devices are used.
- `-h`, `--help` — show the CLI help message and exit.

Show all available arguments:

```shell
python notebooks/hpo_optuna.py --help
```

## Results

CSV tables are written to `hpo_output/tables`, checkpoints to
`hpo_output/checkpoints`, and the resolved run configuration to
`hpo_output/run_config.json`. Interrupted trial data remains in
`hpo_output/recovery` until the next run.
