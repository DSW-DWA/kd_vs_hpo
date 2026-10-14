#!/usr/bin/env zsh

typeset -A teachers
typeset -a losses

teacher_1_path='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/base200_3_archs/arch_1342_teachers__seed_84/arch_1342_teachers__seed_84epoch=epoch=180-val_acc=val_acc=0.6856.ckpt'
teacher_2_path='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/base200_3_archs/arch_8712_teachers__seed_84/arch_8712_teachers__seed_84epoch=epoch=185-val_acc=val_acc=0.7586.ckpt'
teacher_3_path='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/base200_3_archs/arch_11570_teachers__seed_84/arch_11570_teachers__seed_84epoch=epoch=176-val_acc=val_acc=0.8968.ckpt'

teachers[t1]="[{idx: 1342, path: '$teacher_1_path'}]"
teachers[t3]="[{idx: 11570, path: '$teacher_3_path'}]"

teachers[t1_t2_t3]="[{idx: 1342, path: '$teacher_1_path'},{idx: 8712, path: '$teacher_2_path'},{idx: 11570, path: '$teacher_3_path'}]"

losses=(
    'kd_vs_hpo.kd.losses.KullbackLeiblerKDLoss'
    'kd_vs_hpo.kd.losses.MSELogitKDLoss'
    'kd_vs_hpo.kd.losses.MSEProbKDLoss')

for student in 8712; do
  for t_combination in ${(k)teachers}; do
    for loss in $losses; do
        echo "========================================"
        echo "student=$student teachers=$t_combination loss=$loss"
        echo "========================================"

        uv run python -m kd_vs_hpo.common.train_pipeline \
        kd=kld \
        "kd.student=$student" \
        "kd.teachers_mapping=${teachers[$t_combination]}" \
        "kd.kd_loss._target_=$loss" \
        general.seed=84
        
    done
  done
done