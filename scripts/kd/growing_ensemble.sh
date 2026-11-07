#!/usr/bin/env zsh

base_ckpt_path='/home/ivanov.timofey51/kd_vs_hpo/experiments/checkpoints/hpo_5_checkpoints'

student=1342
arch=11570

# Teacher checkpoints for the teacher architecture (11570).
teachers=(${base_ckpt_path}/arch_${arch}_*.pt(N))
n_teachers=${#teachers}

# Recursively generate every subset of teachers of a given size.
# `selected` holds the 1-based indices of the currently chosen teachers.
combos=()
selected=()

gen_combos() {
  local need=$1
  local start=$2
  local i

  if (( need == 0 )); then
    local mapping=""
    for i in "${selected[@]}"; do
      if [[ -n $mapping ]]; then
        mapping+=","
      fi
      mapping+="{idx: $arch, path: '${teachers[$i]}'}"
    done
    combos+=("[$mapping]")
    return
  fi

  for (( i = start; i <= n_teachers - need + 1; i++ )); do
    selected+=($i)
    gen_combos $((need - 1)) $((i + 1))
    selected=(${selected[1,$(( ${#selected} - 1 ))]})
  done
}

for (( size = 1; size <= n_teachers; size++ )); do
  combos=()
  gen_combos $size 1

  for teachers_mapping in "${combos[@]}"; do
    echo "========================================"
    echo "student=$student"
    echo "teacher_count=$size/$n_teachers"
    echo "teachers=$teachers_mapping"
    echo "========================================"

    uv run python -m kd_vs_hpo.common.train_pipeline \
      kd=kld \
      "kd.student=$student" \
      "kd.kd_loss.temperature=1" \
      "kd.teachers_mapping=$teachers_mapping" \
      general.seed=42
  done
done