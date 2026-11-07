#!/usr/bin/env zsh

teacher_dir='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/base_noise_20'

teacher_11570_84="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_84_KullbackLeiblerKDLoss/arch_11570_teachers__seed_84_KullbackLeiblerKDLossepoch=epoch=129-val_acc=val_acc=0.6892.ckpt'}"
teacher_11570_17="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_17_KullbackLeiblerKDLoss/arch_11570_teachers__seed_17_KullbackLeiblerKDLossepoch=epoch=128-val_acc=val_acc=0.6808.ckpt'}"

teacher_8712_17="{idx: 8712, path: '${teacher_dir}/arch_8712_teachers__seed_17_KullbackLeiblerKDLoss/arch_8712_teachers__seed_17_KullbackLeiblerKDLossepoch=epoch=172-val_acc=val_acc=0.6088.ckpt'}"
teacher_8712_84="{idx: 8712, path: '${teacher_dir}/arch_8712_teachers__seed_84_KullbackLeiblerKDLoss/arch_8712_teachers__seed_84_KullbackLeiblerKDLossepoch=epoch=188-val_acc=val_acc=0.6054.ckpt'}"

teacher_1342_84="{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_84_KullbackLeiblerKDLoss/arch_1342_teachers__seed_84_KullbackLeiblerKDLossepoch=epoch=188-val_acc=val_acc=0.5790.ckpt'}"
teacher_1342_17="{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_17_KullbackLeiblerKDLoss/arch_1342_teachers__seed_17_KullbackLeiblerKDLossepoch=epoch=197-val_acc=val_acc=0.5766.ckpt'}"

run() {
  local student="$1"
  local teachers_mapping="$2"

  uv run python -m kd_vs_hpo.common.train_pipeline \
    --multirun \
    kd=kd_one_teacher_different_seeds \
    "kd.teachers_mapping=${teachers_mapping}" \
    "kd.student=${student}" \
    general.seed=42,17,84 \
    kd.checkpoint_dir=experiments/checkpoints/kd_two_teachers_noise20 \
    kd.log_dir=logs/kd_two_teachers_noise20 \
    general.dataset=cifar10 \
    general.label_noise=0.2
}

run 11570 "[${teacher_11570_84},${teacher_11570_17}]"
run 8712  "[${teacher_8712_17},${teacher_8712_84}]"
run 1342  "[${teacher_1342_84},${teacher_1342_17}]"