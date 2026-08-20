#!/usr/bin/env bash
# Fresh Ubuntu ARM box -> a trainer running under tmux, in one command.
#
#   git clone https://github.com/mberma24/Asteroids.git && cd Asteroids
#   ./cloud/setup.sh
#
# Tested against Oracle Cloud "Always Free" Ampere A1 (aarch64, Ubuntu 22.04+). Nothing here
# is Oracle-specific -- any Debian-family box with Python 3.12 works the same way.
#
# The training path never imports pygame, so no display, no X, no SDL packages. Only numpy,
# torch, gymnasium and sb3-contrib, all of which ship linux-aarch64 wheels.
set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES="${EPISODES:-40000}"
SESSION="${SESSION:-asteroids}"
CURRICULUM="${CURRICULUM:-configs/rl-survival-v2.toml}"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "== installing python 3.12 and tmux =="
  sudo apt-get update -qq
  sudo apt-get install -y -qq software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12 python3.12-venv tmux git
fi
command -v tmux >/dev/null 2>&1 || sudo apt-get install -y -qq tmux

if [ ! -x .venv/bin/python ]; then
  echo "== creating .venv =="
  python3.12 -m venv .venv
fi
echo "== installing dependencies (CPU torch: the GPU build is a pointless 2GB here) =="
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -q -e '.[ppo,dev]'

echo "== self-check =="
.venv/bin/python -m pytest tests -q -x --timeout=600 2>/dev/null || .venv/bin/python -m pytest tests -q -x

# A source policy is optional. With one, the fork rung is measured; without one, the ladder
# starts at round 1 and simply takes longer.
SOURCE="${INITIALIZE_FROM:-}"
if [ -z "$SOURCE" ] && [ -d models ]; then
  SOURCE="$(ls -dt models/*/champion 2>/dev/null | head -1 || true)"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "A tmux session named '$SESSION' already exists. Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

# tmux is what makes this survive an SSH disconnect -- the whole point of a cloud box.
if [ -n "$SOURCE" ]; then
  echo "== starting training, forking from $SOURCE =="
  tmux new-session -d -s "$SESSION" \
    "INITIALIZE_FROM='$SOURCE' PPO_DEVICE=cpu ./run.sh train-ppo-survival-v2 $EPISODES 2>&1 | tee cloud-train.log"
else
  echo "== starting training from scratch on $CURRICULUM =="
  tmux new-session -d -s "$SESSION" \
    "PPO_DEVICE=cpu CURRICULUM='$CURRICULUM' ./run.sh train-ppo $EPISODES 2>&1 | tee cloud-train.log"
fi

cat <<DONE

Training is running in tmux session '$SESSION'.

  tmux attach -t $SESSION     watch it
  Ctrl-B then D               detach, leaving it running
  ./run.sh follow             stream evaluations as they land
  ./run.sh status             one-shot progress

Checkpoints land in models/<run>/ every 250 episodes. Copy one home with:
  scp -r ubuntu@THIS_HOST:~/Asteroids/models/<run>/champion ./models/cloud-champion
DONE
