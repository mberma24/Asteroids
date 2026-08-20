#!/usr/bin/env bash
# Resume-or-start, designed to be run by systemd with Restart=always.
#
# Every exit path is safe to repeat: if a complete checkpoint exists it resumes from the
# newest one, otherwise it forks from a source policy, otherwise it starts the ladder at
# round 1. So a reboot, an OOM kill, or an exhausted episode budget all just continue.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-models/oracle-survival-v2}"
EPISODES="${EPISODES:-100000}"
CURRICULUM="${CURRICULUM:-configs/rl-survival-v2.toml}"
SOURCE="${INITIALIZE_FROM:-models/source/champion}"

# LEARNING_RATE must be passed on every resume: the checkpoint stores target_kl but records
# the *base* learning_rate, so a bare resume silently reverts to 3e-4 -- the setting that
# was destroying the policy before the KL cap went in.
export PPO_DEVICE="${PPO_DEVICE:-cpu}"
export PPO_TARGET_KL="${PPO_TARGET_KL:-0.02}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export CURRICULUM

# A checkpoint is only usable once model.zip is on disk; the newest directory may be a
# partial write from whatever killed the previous process.
newest_complete() {
  local candidate
  for candidate in $(ls -d "$RUN"/checkpoint_* 2>/dev/null | sort -r); do
    if [ -f "$candidate/model.zip" ] && [ -f "$candidate/metadata.json" ]; then
      echo "$candidate"; return
    fi
  done
}

checkpoint="$(newest_complete)"
if [ -n "$checkpoint" ]; then
  echo "== resuming $checkpoint =="
  exec env OUTPUT="$RUN" RESUME="$checkpoint" ./run.sh train-ppo "$EPISODES"
fi

if [ -d "$SOURCE" ]; then
  echo "== forking from $SOURCE (rung is measured) =="
  exec env OUTPUT="$RUN" INITIALIZE_FROM="$SOURCE" ./run.sh train-ppo-survival-v2 "$EPISODES"
fi

echo "== no checkpoint and no source policy: starting the ladder at round 1 =="
exec env OUTPUT="$RUN" ./run.sh train-ppo "$EPISODES"
