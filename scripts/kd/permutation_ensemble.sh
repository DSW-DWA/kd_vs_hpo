#!/usr/bin/env zsh

set -euo pipefail

base_ckpt_path='/home/ivanov.timofey51/kd_vs_hpo/checkpoints/hpo_5_checkpoints'

# seeds=(17 84 42)
seeds=(42)

# Number of simultaneous Lightning training processes on the A100.
max_jobs=6

# Currently running child PIDs.
pids=()

# Record failed (non-zero) exit codes per PID.
declare -A failed

# Reap one finished child when the pool is full.
wait_for_slot() {
  while (( ${#pids} >= max_jobs )); do
    local remaining=()

    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        remaining+=("$pid")
      else
        wait "$pid" || failed[$pid]=$?
      fi
    done

    pids=("${remaining[@]}")

    (( ${#pids} >= max_jobs )) && sleep 1
  done
}

archs=(${base_ckpt_path}/arch_*_initial.pt(N:t:r))
archs=(${archs:s/arch_//:s/_initial//})


for arch in $archs; do
  ckpts=(${base_ckpt_path}/arch_${arch}_*.pt(N))
  n=${#ckpts}

  for ((a=1; a<=n; a++)); do
    for ((b=a+1; b<=n; b++)); do
      for ((c=b+1; c<=n; c++)); do

        teachers_mapping="[{idx: $arch, path: '${ckpts[$a]}'},{idx: $arch, path: '${ckpts[$b]}'},{idx: $arch, path: '${ckpts[$c]}'}]"

        for seed in $seeds; do
          wait_for_slot

          echo "========================================"
          echo "Launching:"
          echo "student=$arch"
          echo "teachers=${ckpts[$a]##*/},${ckpts[$b]##*/},${ckpts[$c]##*/}"
          echo "seed=$seed"
          echo "active_jobs=${#pids}/${max_jobs}"
          echo "========================================"

          uv run python -m kd_vs_hpo.common.train_pipeline \
            kd=kld \
            "kd.student=$arch" \
            "kd.kd_loss.temperature=1" \
            "kd.teachers_mapping=$teachers_mapping" \
            "general.seed=$seed" &

          pids+=("$!")
        done
      done
    done
  done
done

echo "All jobs launched. Waiting for completion..."

for pid in $pids; do
  wait "$pid" || failed[$pid]=$?
done

if (( ${#failed[@]} > 0 )); then
  echo "ERROR: ${#failed[@]} job(s) failed with non-zero exit code:"
  for pid in ${(k)failed}; do
    echo "  pid=$pid exit_code=${failed[$pid]}"
  done
  exit 1
fi

echo "All experiments completed."
