#!/usr/bin/env bash
# Restart-safe centralized two-ship trainer for the systemd VM service.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-models/team-factorized-v2}"
STEPS="${STEPS:-10000000}"
PARALLEL_ENVS="${PARALLEL_ENVS:-4}"
EVAL_EVERY="${EVAL_EVERY:-65536}"
EVAL_EPISODES="${EVAL_EPISODES:-128}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-3}"

newest_complete() {
  local candidate
  for candidate in $(ls -d "$RUN"/checkpoint_* 2>/dev/null | sort -r); do
    if [ -f "$candidate/model.zip" ] && [ -f "$candidate/metadata.json" ]; then
      echo "$candidate"
      return
    fi
  done
}

args=(--output "$RUN" --steps "$STEPS" --parallel-envs "$PARALLEL_ENVS"
      --eval-every "$EVAL_EVERY" --eval-episodes "$EVAL_EPISODES"
      --keep-checkpoints "$KEEP_CHECKPOINTS"
      --seed "${SEED:-0}" --device "${PPO_DEVICE:-cpu}" --stop-when-mastered)
checkpoint="$(newest_complete)"
if [ -n "$checkpoint" ]; then
  echo "== resuming centralized team trainer from $checkpoint =="
  args+=(--resume "$checkpoint")
elif [ -n "${INITIAL_CHECKPOINT:-}" ]; then
  echo "== initializing centralized team trainer from $INITIAL_CHECKPOINT =="
  args+=(--resume "$INITIAL_CHECKPOINT")
else
  echo "== starting centralized team trainer from scratch in $RUN =="
fi

exec .venv/bin/python -m asteroid_survival train-team "${args[@]}"
