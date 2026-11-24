#!/bin/sh
set -e

python -m kd_vs_hpo.common.train_pipeline \
  --multirun \
  kd=kld \
  kd.student=11570,8712,1342 \
  general.seed=42,17,84 \
  kd.checkpoint_dir=experiments/checkpoints/cifar100/base_noise_0 \
  kd.log_dir=logs/cifar100/base_noise_0 \
  general.dataset=cifar100 \
  general.architectures_path=experiments/nats_architectures_7_cifar100.json \
  general.label_noise=0.0

echo "Training finished"