#!/usr/bin/env zsh

typeset -A teachers

teacher_1_path='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/base200_3_archs/arch_11570_teachers__seed_17/arch_11570_teachers__seed_17epoch=epoch=195-val_acc=val_acc=0.8970.ckpt'
teacher_2_path='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/base200_3_archs/arch_11570_teachers__seed_42/arch_11570_teachers__seed_42epoch=epoch=185-val_acc=val_acc=0.8994.ckpt'
teacher_3_path='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/base200_3_archs/arch_11570_teachers__seed_84/arch_11570_teachers__seed_84epoch=epoch=176-val_acc=val_acc=0.8968.ckpt'

teachers[t2]="[{idx: 11570, path: '$teacher_2_path'}]"
teachers[t1_t2]="[{idx: 11570, path: '$teacher_1_path'},{idx: 11570, path: '$teacher_2_path'}]"
teachers[t1_t2_t3]="[{idx: 11570, path: '$teacher_1_path'},{idx: 11570, path: '$teacher_2_path'},{idx: 11570, path: '$teacher_3_path'}]"


for student in 1342 11570; do
  for t_combination in ${(k)teachers}; do
    for temp in 1 2 4; do
        echo "========================================"
        echo "student=$student teachers=$t_combination temp=$temp"
        echo "========================================"
    
        uv run python -m kd_vs_hpo.common.train_pipeline \
          kd=kld \
          "kd.student=$student" \
          kd.kd_loss._target_='kd_vs_hpo.kd.losses.KullbackLeiblerKDLoss' \
          "kd.kd_loss.temperature=$temp" \
          "kd.teachers_mapping=${teachers[$t_combination]}" \
          general.seed=42
    done
  done
done

