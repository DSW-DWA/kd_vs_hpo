#!/usr/bin/env zsh

student=8712
seed=42

trials=(
  "0.1 0.0005"
  "0.11 0.00045"
  "0.09 0.00055"
  "0.11 0.00055"
  "0.09 0.00045"
)

for trial in $trials; do
  lr=${trial[(w)1]}
  weight_decay=${trial[(w)2]}

  echo "========================================"
  echo "student=$student lr=$lr weight_decay=$weight_decay"
  echo "========================================"

  uv run python -m kd_vs_hpo.common.train_pipeline \
    kd=kld \
    "kd.student=$student" \
    "kd.optimizer_params.lr=$lr" \
    "kd.optimizer_params.weight_decay=$weight_decay" \
    general.seed=$seed
done