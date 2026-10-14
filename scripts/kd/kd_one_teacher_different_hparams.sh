#!/usr/bin/env zsh

teacher_dir='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/five_teachers/chosen_3'

teacher_11570_strong="[{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_42_strong/arch_11570_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=176-val_acc=val_acc=0.9220.ckpt'}]"
teacher_11570_mid="[{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_42_mid/arch_11570_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=146-val_acc=val_acc=0.8924.ckpt'}]"
teacher_11570_weak="[{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_42_weak/arch_11570_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=192-val_acc=val_acc=0.6022.ckpt'}]"
teacher_1342_strong="[{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_42_strong/arch_1342_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=198-val_acc=val_acc=0.7672.ckpt'}]"
teacher_1342_mid="[{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_42_mid/arch_1342_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=191-val_acc=val_acc=0.7250.ckpt'}]"
teacher_1342_weak="[{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_42_weak/arch_1342_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=157-val_acc=val_acc=0.4842.ckpt'}]"

uv run python -m kd_vs_hpo.common.train_pipeline \
  -m \
  kd=kd_one_teacher_different_seeds \
  "kd.student=8712,11570,1342" \
  "kd.teachers_mapping=${teacher_11570_strong},${teacher_11570_mid},${teacher_11570_weak},${teacher_1342_strong},${teacher_1342_mid},${teacher_1342_weak}" \
  kd.checkpoint_dir=checkpoints/kd_one_teacher_different_hparams \
  kd.log_dir=logs/kd_one_teacher_different_hparams \
  general.seed=42