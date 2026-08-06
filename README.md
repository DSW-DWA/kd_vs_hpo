# Setup

```shell
make install
source .venv/bin/activate
```

# HPO

```shell
python -u -m notebooks.hpo_optuna [OPTIONS]
```

| Option | Description | Default |
| --- | --- | --- |
| `--arch-rows ROW...` | Architecture rows | `0` |
| `--n-trials N` | Trials per study | `20` |
| `--max-epochs N` | Epochs per trial | `200` |
| `--samplers NAME...` | `tpe`, `grid`, `cmaes`, `gp` | all |
| `--pruners NAME...` | `none`, `successive_halving`, `hyperband` | `successive_halving hyperband` |
| `--device DEVICE` | `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--processes N` | Parallel studies | `1` |
| `--gpu-ids ID...` | CUDA device IDs | all |
| `--output-dir PATH` | Output directory | `hpo_output` |

# Plain training

```shell
python -u -m notebooks.train_plain [OPTIONS]
```

| Option | Description | Default |
| --- | --- | --- |
| `--arch-rows ROW...` | Architecture rows | `0` |
| `--lr VALUE` | Initial learning rate | `0.05` |
| `--weight-decay VALUE` | Weight decay | `0.0005` |
| `--device DEVICE` | `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--seed N` | Random seed | `42` |
| `--batch-size N` | Batch size | `256` |
| `--num-workers N` | Data workers | `6` |
| `--[no-]deterministic` | Deterministic mode | disabled |
| `--[no-]amp` | Mixed precision | enabled |
| `--[no-]verbose` | Epoch progress | disabled |
| `--data-root PATH` | Dataset directory | `data` |
| `--architectures-path PATH` | Architecture file | `experiments/nats_architectures_10.json` |
| `--output-dir PATH` | Output directory | `plain_training_output` |
| `--dry-run` | Print configuration | disabled |
