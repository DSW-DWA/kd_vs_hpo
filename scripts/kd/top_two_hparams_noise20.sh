#!/usr/bin/env zsh

seed=42
archs=(11570 8712 1342)
params_path='experiments/top_two_teachers_params.json'

n_configs=$(python -c "import json; print(len(json.load(open('$params_path'))))")

for arch in $archs; do
  for ((i = 0; i < n_configs; i++)); do
    read -r lr weight_decay batch_size momentum < <(python -c "
import json
p = json.load(open('$params_path'))[$i]
print(p['lr'], p['weight_decay'], p['batch_size'], p['momentum'])
")

    echo "========================================"
    echo "arch=$arch lr=$lr weight_decay=$weight_decay batch_size=$batch_size momentum=$momentum"
    echo "========================================"

    uv run python -m kd_vs_hpo.common.train_pipeline \
      kd=kld \
      "kd.student=$arch" \
      "kd.optimizer_params.lr=$lr" \
      "kd.optimizer_params.weight_decay=$weight_decay" \
      "general.batch_size=$batch_size" \
      "general.momentum=$momentum" \
      general.seed=$seed \
      kd.checkpoint_dir=experiments/checkpoints/top_two_hparams_noise20 \
      kd.log_dir=logs/top_two_hparams_noise20 \
      general.dataset=cifar10 \
      general.label_noise=0.2
  done
done