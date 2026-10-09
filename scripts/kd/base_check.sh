#!/usr/bin/env zsh

uv run python -m kd_vs_hpo.common.train_pipeline \
  --multirun \
  kd=kld \
  kd.student=11570,8712,1342 \
  general.seed=42,17,84 \
  kd.checkpoint_dir=checkpoints/base_check_3_arch \
  kd.log_dir=logs/base_check_3_arch 

