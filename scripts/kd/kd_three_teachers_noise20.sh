#!/usr/bin/env zsh

teacher_dir='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/base_noise_20'

teacher_11570_17="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_17_KullbackLeiblerKDLoss/arch_11570_teachers__seed_17_KullbackLeiblerKDLossepoch=epoch=128-val_acc=val_acc=0.6808.ckpt'}"
teacher_11570_42="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_42_KullbackLeiblerKDLoss/arch_11570_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=136-val_acc=val_acc=0.6870.ckpt'}"
teacher_11570_84="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_84_KullbackLeiblerKDLoss/arch_11570_teachers__seed_84_KullbackLeiblerKDLossepoch=epoch=129-val_acc=val_acc=0.6892.ckpt'}"

teachers_mapping="[${teacher_11570_17},${teacher_11570_42},${teacher_11570_84}]"

uv run python -m kd_vs_hpo.common.train_pipeline \
  --multirun \
  kd=kd_one_teacher_different_seeds \
  "kd.teachers_mapping=$teachers_mapping" \
  "kd.student=8712,11570,1342" \
  general.seed=42,17,84 \
  kd.checkpoint_dir=experiments/checkpoints/kd_three_teachers_noise20 \
  kd.log_dir=logs/kd_three_teachers_noise20 \
  general.dataset=cifar10 \
  general.label_noise=0.2 
