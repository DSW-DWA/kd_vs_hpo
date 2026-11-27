#!/bin/sh
set -e

teacher_dir='experiments/checkpoints/cifar100/base_noise_10'

teacher_11570_84="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_84_KullbackLeiblerKDLoss/arch_11570_teachers__seed_84_KullbackLeiblerKDLossepoch=epoch=198-val_acc=val_acc=0.5442.ckpt'}"
teacher_11570_17="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_17_KullbackLeiblerKDLoss/arch_11570_teachers__seed_17_KullbackLeiblerKDLossepoch=epoch=188-val_acc=val_acc=0.5414.ckpt'}"

teacher_8712_17="{idx: 8712, path: '${teacher_dir}/arch_8712_teachers__seed_17_KullbackLeiblerKDLoss/arch_8712_teachers__seed_17_KullbackLeiblerKDLossepoch=epoch=195-val_acc=val_acc=0.4232.ckpt'}"
teacher_8712_84="{idx: 8712, path: '${teacher_dir}/arch_8712_teachers__seed_84_KullbackLeiblerKDLoss/arch_8712_teachers__seed_84_KullbackLeiblerKDLossepoch=epoch=198-val_acc=val_acc=0.4336.ckpt'}"

teacher_1342_84="{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_84_KullbackLeiblerKDLoss/arch_1342_teachers__seed_84_KullbackLeiblerKDLossepoch=epoch=186-val_acc=val_acc=0.3848.ckpt'}"
teacher_1342_17="{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_17_KullbackLeiblerKDLoss/arch_1342_teachers__seed_17_KullbackLeiblerKDLossepoch=epoch=198-val_acc=val_acc=0.3942.ckpt'}"

run() {
  local student="$1"
  local teachers_mapping="$2"

  python -m kd_vs_hpo.common.train_pipeline \
    --multirun \
    kd=kd_one_teacher_different_seeds \
    "kd.teachers_mapping=${teachers_mapping}" \
    "kd.student=${student}" \
    general.seed=42,17,84 \
    kd.checkpoint_dir=experiments/checkpoints/cifar100/kd_two_teachers_noise10 \
    kd.log_dir=logs/cifar100/kd_two_teachers_noise10 \
    general.dataset=cifar100_n10 \
    general.architectures_path=experiments/nats_architectures_7_cifar100.json \
    general.label_noise=0.0
}

run 11570 "[${teacher_11570_84},${teacher_11570_17}]"
run 8712  "[${teacher_8712_17},${teacher_8712_84}]"
run 1342  "[${teacher_1342_84},${teacher_1342_17}]"

echo "Training finished"
