#!/usr/bin/env zsh

teacher_dir='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/top_two_hparams_noise20'

teacher_11570_1="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_42_KullbackLeiblerKDLoss/arch_11570_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=81-val_acc=val_acc=0.6858.ckpt'}"
teacher_11570_2="{idx: 11570, path: '${teacher_dir}/arch_11570_teachers__seed_42_KullbackLeiblerKDLoss_2026-10-31-20-28-41/arch_11570_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=47-val_acc=val_acc=0.6726.ckpt'}"

teacher_8712_1="{idx: 8712, path: '${teacher_dir}/arch_8712_teachers__seed_42_KullbackLeiblerKDLoss/arch_8712_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=197-val_acc=val_acc=0.6276.ckpt'}"
teacher_8712_2="{idx: 8712, path: '${teacher_dir}/arch_8712_teachers__seed_42_KullbackLeiblerKDLoss_2026-11-01-00-22-35/arch_8712_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=109-val_acc=val_acc=0.5806.ckpt'}"

teacher_1342_1="{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_42_KullbackLeiblerKDLoss/arch_1342_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=191-val_acc=val_acc=0.5810.ckpt'}"
teacher_1342_2="{idx: 1342, path: '${teacher_dir}/arch_1342_teachers__seed_42_KullbackLeiblerKDLoss_2026-11-01-04-12-21/arch_1342_teachers__seed_42_KullbackLeiblerKDLossepoch=epoch=184-val_acc=val_acc=0.5442.ckpt'}"

# params_path='experiments/top_two_teachers_params.json'
params_path='experiments/top_one_teachers_params.json'

n_configs=$(python -c "import json; print(len(json.load(open('$params_path'))))")

run() {
  local student="$1"
  local teachers_mapping="$2"
  local lr="$3"
  local weight_decay="$4"
  local batch_size="$5"
  local momentum="$6"

  uv run python -m kd_vs_hpo.common.train_pipeline \
    --multirun \
    kd=kd_one_teacher_different_seeds \
    "kd.teachers_mapping=${teachers_mapping}" \
    "kd.student=${student}" \
    "kd.optimizer_params.lr=${lr}" \
    "kd.optimizer_params.weight_decay=${weight_decay}" \
    "general.batch_size=${batch_size}" \
    "general.momentum=${momentum}" \
    general.seed=17,84 \
    kd.checkpoint_dir=experiments/checkpoints/kd_two_teachers_noise20_top_two_diff_hparams \
    kd.log_dir=logs/kd_two_teachers_noise20_top_two_diff_hparams \
    general.dataset=cifar10 \
    general.label_noise=0.2
}

for ((i = 0; i < n_configs; i++)); do
  read -r lr weight_decay batch_size momentum < <(python -c "
import json
p = json.load(open('$params_path'))[$i]
print(p['lr'], p['weight_decay'], p['batch_size'], p['momentum'])
")

  echo "========================================"
  echo "config=$i lr=$lr weight_decay=$weight_decay batch_size=$batch_size momentum=$momentum"
  echo "========================================"

  run 11570 "[${teacher_11570_1},${teacher_11570_2}]" "$lr" "$weight_decay" "$batch_size" "$momentum"
  run 8712  "[${teacher_8712_1},${teacher_8712_2}]" "$lr" "$weight_decay" "$batch_size" "$momentum"
  run 1342  "[${teacher_1342_1},${teacher_1342_2}]" "$lr" "$weight_decay" "$batch_size" "$momentum"
done