#!/usr/bin/env zsh

teacher_dir='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/base_check_3_arch'

teacher_11570_42="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_42/arch_11570_teachers__seed_42_epoch=epoch=199-val_acc=val_acc=0.9222.ckpt'}"
teacher_11570_17="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_17/arch_11570_teachers__seed_17_epoch=epoch=193-val_acc=val_acc=0.9160.ckpt'}"
teacher_11570_84="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_84/arch_11570_teachers__seed_84_epoch=epoch=171-val_acc=val_acc=0.9220.ckpt'}"
teacher_1342_42="{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_42/arch_1342_teachers__seed_42_epoch=epoch=196-val_acc=val_acc=0.7584.ckpt'}"
teacher_1342_17="{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_17/arch_1342_teachers__seed_17_epoch=epoch=187-val_acc=val_acc=0.7346.ckpt'}"
teacher_1342_84="{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_84/arch_1342_teachers__seed_84_epoch=epoch=194-val_acc=val_acc=0.7476.ckpt'}"

teachers=(
  "$teacher_11570_42"
  "$teacher_11570_17"
  "$teacher_11570_84"
  "$teacher_1342_42"
  "$teacher_1342_17"
  "$teacher_1342_84"
)

pairs=()

for ((i = 1; i <= ${#teachers}; i++)); do
  for ((j = i + 1; j <= ${#teachers}; j++)); do
    pairs+=("[${teachers[$i]},${teachers[$j]}]")
  done
done


teachers_mapping="${(j:,:)pairs}"

uv run python -m kd_vs_hpo.common.train_pipeline \
  -m \
  kd=kd_two_teachers_different_seeds \
  "kd.student=8712" \
  "kd.teachers_mapping=$teachers_mapping" \
  general.seed=42