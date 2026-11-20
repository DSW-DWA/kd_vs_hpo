#!/usr/bin/env zsh

uv run python -m kd_vs_hpo.common.train_pipeline \
  --multirun \
  kd=kld \
  kd.student=11570,8712,1342 \
  general.seed=42,17,84 \
  kd.checkpoint_dir=experiments/checkpoints/cifar10_noise_fix/base_noise_10 \
  kd.log_dir=logs/base_noise_10_new \
  general.dataset=cifar10_n10 \

