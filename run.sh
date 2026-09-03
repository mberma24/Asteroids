#!/usr/bin/env bash
# Convenience launcher for the common Asteroid Survival tasks.
# Run with no arguments for a menu, or pass a command directly: ./run.sh play
set -euo pipefail
cd "$(dirname "$0")"

# Overridable so the dispatch table can be exercised without launching anything.
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || [ "${PY}" != ".venv/bin/python" ] || PY="python3"

# Best checkpoint by held-out survival, across every model directory, restricted to those
# the given config can load. Falls back to most-recently-written for runs with no
# evaluation recorded.
latest_checkpoint() {
  local config="${1:-configs/rl.toml}"
  local algorithm="${2:-}"
  $PY - "$config" "$algorithm" <<'PYEOF' 2>/dev/null || true
import sys
from asteroid_survival.cli import _latest_checkpoint
from asteroid_survival.config import load_config
try:
    algorithms = set(filter(None, sys.argv[2].split(","))) if len(sys.argv) > 2 and sys.argv[2] else None
    print(_latest_checkpoint(load_config(sys.argv[1]), algorithms=algorithms))
except SystemExit:
    pass
PYEOF
}

# Best checkpoint a given mode can actually load, resolved through the shared mode table.
best_checkpoint() {
  $PY -m asteroid_survival best-checkpoint "$1" ${2:+"$2"} --algorithms "${3:-ppo}" 2>/dev/null || true
}

# Recover how many history frames a checkpoint was trained with, from its observation size,
# so scoring a model never fails just because the flag was not repeated by hand.
infer_history_frames() {
  local checkpoint="$1" config="$2"
  $PY - "$checkpoint" "$config" <<'PYEOF' 2>/dev/null || echo 0
import json, sys
from pathlib import Path
from asteroid_survival.config import load_config
from asteroid_survival.rl.environment import ASTEROID_FEATURES, SHIP_FEATURES

metadata = json.loads((Path(sys.argv[1]) / "metadata.json").read_text())
config = load_config(sys.argv[2])
layout = metadata.get("observation_layout") or {}
if "history_frames" in layout:
    print(int(layout["history_frames"]))
    raise SystemExit
slots = config.asteroid.active_cap
width, remainder = divmod(metadata["observation_size"] - SHIP_FEATURES, slots)
frames, odd = divmod(width - ASTEROID_FEATURES, 2)
print(frames if remainder == 0 and odd == 0 and frames >= 0 else 0)
PYEOF
}

infer_long_history() {
  $PY - "$1" <<'PYEOF' 2>/dev/null || echo "0 8"
import json, sys
from pathlib import Path
layout = json.loads((Path(sys.argv[1]) / "metadata.json").read_text()).get("observation_layout") or {}
print(int(layout.get("history_long_frames", 0)), int(layout.get("history_long_stride", 8)))
PYEOF
}

# Say which checkpoint was chosen and what it actually scored, so the pick is not magic.
describe_checkpoint() {
  $PY - "$1" "${2:-}" <<'PYEOF' 2>/dev/null || echo "Using checkpoint: $1"
import json, sys
from pathlib import Path
checkpoint = Path(sys.argv[1])
frames = sys.argv[2] if len(sys.argv) > 2 else ""
score = None
log = checkpoint.parent / "evaluation.jsonl"
if checkpoint.name == "champion":
    state_path = checkpoint.parent / "champion_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        detail = (f" - CHAMPION from episode {state.get('episode')}: "
                  f"{state.get('completion_rate', 0):.1%} complete, "
                  f"accuracy {state.get('accuracy', 0):.3f}")
        if frames:
            detail += f", history frames {frames}"
        print(f"Using {checkpoint}{detail}")
        raise SystemExit
if log.is_file():
    episode = checkpoint.name.removeprefix("checkpoint_").lstrip("0") or "0"
    for line in log.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if str(record.get("episode")) == episode:
            score = record
detail = ""
if score:
    if "stages" in score:
        index = min(score.get("training_stage", 0), len(score["stages"]) - 1)
        stage = score["stages"][index]
        detail = (f" - held-out stage {index + 1}: {stage['completion_rate']:.1%} complete, "
                  f"accuracy {stage['mean_accuracy']:.3f}")
    else:
        detail = (f" - best held-out: {score['mean_survival_time']:.2f}s survival, "
                  f"accuracy {score['mean_accuracy']:.3f}")
else:
    detail = " - newest (this run recorded no evaluation)"
if frames:
    detail += f", history frames {frames}"
print(f"Using {checkpoint}{detail}")
PYEOF
}

require_checkpoint() {
  local checkpoint
  checkpoint="${CHECKPOINT:-$(latest_checkpoint "${2:-configs/rl.toml}" "${3:-}")}"
  if [ -z "$checkpoint" ]; then
    echo "No trained model found. Run './run.sh train' first," >&2
    echo "or point at one explicitly: CHECKPOINT=models/muzero-v4/checkpoint_010000 ./run.sh $1" >&2
    exit 1
  fi
  echo "$checkpoint"
}

# The output directory of a trainer that is actually running, if there is one. `status`
# prefers it over the newest directory on disk: after a restart the old run sits there
# looking stopped, and picking it by timestamp reports "not running" for a healthy job.
running_run() {
  $PY - <<'PYEOF' 2>/dev/null
import shlex, subprocess
for line in subprocess.check_output(["ps", "ax", "-o", "command="], text=True).splitlines():
    try:
        args = shlex.split(line.strip())
    except ValueError:
        continue
    if "asteroid_survival" not in args:
        continue
    if not any(name in args for name in ("train", "train-ppo", "train-mappo", "train-team")):
        continue
    try:
        print(args[args.index("--output") + 1])
    except (ValueError, IndexError):
        pass
    break
PYEOF
}

assert_no_other_trainer() {
  [ "${ALLOW_CONCURRENT:-0}" = "1" ] && return
  local running
  running="$($PY -c '
import shlex, subprocess
matches = []
for line in subprocess.check_output(["ps", "ax", "-o", "pid=,command="], text=True).splitlines():
    fields = line.strip().split(maxsplit=1)
    if len(fields) != 2:
        continue
    pid, command = fields
    try:
        args = shlex.split(command)
    except ValueError:
        continue
    if "asteroid_survival" not in args:
        continue
    if any(name in args for name in ("train", "train-ppo", "train-mappo", "train-team")):
        matches.append(pid)
print(" ".join(matches))
')"
  if [ -n "$running" ]; then
    echo "Another trainer is already running (pid $running)." >&2
    echo "Stop it first so timing and learning runs remain comparable." >&2
    echo "Set ALLOW_CONCURRENT=1 only if you intentionally want resource contention." >&2
    exit 1
  fi
}

cmd_play() {          # play any mode yourself
  $PY -m asteroid_survival arena "${1:-arcade}" ${2:+"$2"} --seed "${SEED:-7}" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} ${ANY_RUN:+--any-run}
}

cmd_patterns() {      # watch the trajectory shapes, with trails and labels
  $PY -m asteroid_survival patterns ${1:+"$1"} --seed "${SEED:-3}"
}

cmd_showdown() {      # you + greedy + the best compatible model, in any mode
  $PY -m asteroid_survival arena "${1:-arcade}" ${2:+"$2"} \
    --with "closest,${ALGO:-ppo}" --seed "${SEED:-11}" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} ${ANY_RUN:+--any-run}
}

is_mode() { case "$1" in arcade|endless|round|survival|survival-v2|varied) return 0 ;; *) return 1 ;; esac; }
mode_takes_round() { case "$1" in round|survival|survival-v2|varied) return 0 ;; *) return 1 ;; esac; }

# Scored, controlled runs. Unlike showdown these are not a shared arena, so the numbers are
# comparable; showdown ships help each other by clearing the same rocks.
run_compare() {
  local human="$1" mode="$2" round="$3" episodes="$4"
  local args=(--episodes "$episodes") checkpoint
  local config="${CONFIG:-configs/rl-arcade.toml}"
  if [ -n "$mode" ]; then
    args+=(--mode "$mode")
    [ -n "$round" ] && args+=(--round "$round")
    checkpoint="${CHECKPOINT:-$(best_checkpoint "$mode" "$round" ppo,recurrent_ppo,muzero)}"
    if [ -z "$checkpoint" ]; then
      echo "No checkpoint can load mode '$mode'. Train one, or pass CHECKPOINT=path." >&2
      exit 1
    fi
  else
    args+=(--config "$config")
    checkpoint="$(require_checkpoint compare "$config")"
  fi
  # Modern checkpoints record their history layout, so the config here is only a fallback.
  local frames="${HISTORY_FRAMES:-$(infer_history_frames "$checkpoint" "$config")}"
  local long stride; read -r long stride <<< "$(infer_long_history "$checkpoint")"
  describe_checkpoint "$checkpoint" "$frames"
  [ "$human" = "no" ] && args+=(--no-human)
  $PY -m asteroid_survival compare --checkpoint "$checkpoint" \
    --history-frames "$frames" --history-long-frames "$long" \
    --history-long-stride "$stride" "${args[@]}"
}

# Accepts: [mode [round]] [episodes]. A bare number is always the episode count.
scored_run() {
  local human="$1" episodes="$2"; shift 2
  local mode="" round=""
  if [ $# -gt 0 ] && is_mode "$1"; then
    mode="$1"; shift
    if mode_takes_round "$mode" && [ $# -gt 0 ]; then round="$1"; shift; fi
  fi
  [ $# -gt 0 ] && episodes="$1"
  run_compare "$human" "$mode" "$round" "$episodes"
}

cmd_compare() {       # you and the agents play the same seeds, scored side by side
  scored_run yes 5 "$@"
}

cmd_versus() {        # score several models against each other, each in its own game
  local mode="" round="" episodes=10
  if [ $# -gt 0 ] && is_mode "$1"; then
    mode="$1"; shift
    if mode_takes_round "$mode" && [ $# -gt 0 ]; then round="$1"; shift; fi
  fi
  [ $# -gt 0 ] && episodes="$1"
  local args=(--episodes "$episodes" --output "${OUTPUT:-metrics/versus.json}")
  if [ -n "$mode" ]; then
    args+=(--mode "$mode"); [ -n "$round" ] && args+=(--round "$round")
  else
    args+=(--config "${CONFIG:-configs/rl-arcade.toml}")
  fi
  # MODELS is a comma-separated list of checkpoints; without it, use the newest run.
  local models="${MODELS:-}"
  if [ -z "$models" ]; then
    models="$(best_checkpoint "${mode:-arcade}" "$round" ppo,recurrent_ppo,muzero)"
  fi
  local IFS=,
  for entry in $models; do
    [ -n "$entry" ] && args+=(--model "$entry")
  done
  unset IFS
  [ "${HUMAN:-0}" = "1" ] || args+=(--no-human)
  [ "${GREEDY:-1}" = "1" ] || args+=(--no-greedy)
  [ "${PILOT:-1}" = "1" ] || args+=(--no-pilot)
  $PY -m asteroid_survival compare "${args[@]}" --history-frames "${HISTORY_FRAMES:-8}" \
    --history-long-frames "${HISTORY_LONG_FRAMES:-8}" --history-long-stride "${HISTORY_LONG_STRIDE:-8}"
}

cmd_watch() {         # same as compare, but the agents play and you just watch
  scored_run no 20 "$@"
}

cmd_baseline() {      # score the non-learning greedy controller
  $PY -m asteroid_survival evaluate-baseline --config "${CONFIG:-configs/rl-arcade.toml}" \
    --episodes "${1:-60}" --output metrics/closest-baseline.json
}

cmd_train() {         # train a new model
  assert_no_other_trainer
  local episodes="${1:-10000}"
  local output="${OUTPUT:-models/muzero-$(date +%m%d-%H%M)}"
  echo "Training $episodes episodes into $output"
  echo "Progress: ./run.sh status $output"
  # RESUME=<checkpoint> continues from a saved model. Without this the variable was accepted
  # and silently ignored, which quietly starts a fresh run when you meant to continue one.
  local train_args=(
    --curriculum "${CURRICULUM:-configs/rl-curriculum.toml}"
    --output "$output"
  )
  if [ -n "${RESUME:-${CHECKPOINT:-}}" ]; then
    train_args+=(--resume "${RESUME:-$CHECKPOINT}")
    echo "Resuming from ${RESUME:-$CHECKPOINT}"
  fi
  if [ -n "${INITIALIZE_FROM:-}" ]; then
    train_args+=(--initialize-from "$INITIALIZE_FROM")
    echo "Initializing perception/policy from $INITIALIZE_FROM"
  fi
  if [ -n "${RESUME_LEARNING_RATE:-}" ]; then
    train_args+=(--resume-learning-rate "$RESUME_LEARNING_RATE")
    echo "Resetting resumed optimizer at $RESUME_LEARNING_RATE"
  fi
  if [ -n "${START_STAGE:-}" ]; then
    train_args+=(--start-stage "$START_STAGE")
    echo "Starting curriculum at stage $START_STAGE"
  fi
  if [ "${STOP_WHEN_MASTERED:-0}" = "1" ]; then
    train_args+=(--stop-when-mastered)
  fi
  $PY -m asteroid_survival train \
    "${train_args[@]}" \
    --episodes "$episodes" \
    --simulations "${SIMULATIONS:-50}" \
    --learning-rate "${LEARNING_RATE:-0.001}" \
    --updates-per-episode "${UPDATES_PER_EPISODE:-32}" \
    --parallel-envs "${PARALLEL_ENVS:-16}" \
    --history-frames "${HISTORY_FRAMES:-8}" \
    --history-long-frames "${HISTORY_LONG_FRAMES:-8}" \
    --history-long-stride "${HISTORY_LONG_STRIDE:-8}" \
    --log-every "${LOG_EVERY:-250}" \
    --eval-every "${EVAL_EVERY:-250}" --eval-episodes 20 \
    --checkpoint-every "${CHECKPOINT_EVERY:-250}"
}

cmd_train_ppo() {     # train PPO without tree search or replay
  local recurrent="${1:-0}"
  local episodes="${2:-10000}"
  local label="ppo-ff"
  [ "$recurrent" = "1" ] && label="ppo-lstm"
  local output="${OUTPUT:-models/${label}-$(date +%m%d-%H%M)}"
  assert_no_other_trainer
  echo "Training $label for $episodes episodes into $output"
  echo "Progress: ./run.sh status $output"
  local args=(
    --curriculum "${CURRICULUM:-configs/rl-curriculum.toml}"
    --output "$output" --episodes "$episodes" --seed "${SEED:-0}"
    --parallel-envs "${PARALLEL_ENVS:-8}"
    --history-frames "${HISTORY_FRAMES:-8}"
    --history-long-frames "${HISTORY_LONG_FRAMES:-8}"
    --history-long-stride "${HISTORY_LONG_STRIDE:-8}"
    --eval-every "${EVAL_EVERY:-250}" --device "${PPO_DEVICE:-auto}"
  )
  [ -n "${LEARNING_RATE:-}" ] && args+=(--learning-rate "$LEARNING_RATE")
  [ -n "${PPO_GAMMA:-}" ] && args+=(--gamma "$PPO_GAMMA")
  [ -n "${PPO_TARGET_KL:-}" ] && args+=(--target-kl "$PPO_TARGET_KL")
  [ -n "${PPO_ENT_COEF:-}" ] && args+=(--ent-coef "$PPO_ENT_COEF")
  [ -n "${PPO_VF_COEF:-}" ] && args+=(--vf-coef "$PPO_VF_COEF")
  [ -n "${PPO_ENTROPY_FLOOR:-}" ] && args+=(--entropy-floor "$PPO_ENTROPY_FLOOR")
  [ -n "${PPO_N_EPOCHS:-}" ] && args+=(--n-epochs "$PPO_N_EPOCHS")
  [ -n "${ENCODER:-}" ] && args+=(--encoder "$ENCODER")
  [ "${STOP_WHEN_MASTERED:-0}" = "1" ] && args+=(--stop-when-mastered)
  [ -n "${RESUME:-}" ] && args+=(--resume "$RESUME")
  [ -n "${INITIALIZE_FROM:-}" ] && args+=(--initialize-from "$INITIALIZE_FROM")
  [ -n "${START_STAGE:-}" ] && args+=(--start-stage "$START_STAGE")
  if [ "$recurrent" = "1" ]; then
    $PY -m asteroid_survival train-ppo "${args[@]}" --recurrent
  else
    $PY -m asteroid_survival train-ppo "${args[@]}"
  fi
}

cmd_ppo_screen() {    # controlled one-seed ablation, run sequentially
  local episodes="${1:-10000}"
  local stamp; stamp="$(date +%m%d-%H%M)"
  OUTPUT="models/ppo-ff-$stamp" cmd_train_ppo 0 "$episodes"
  OUTPUT="models/ppo-lstm-$stamp" cmd_train_ppo 1 "$episodes"
}

cmd_train_ppo_endless() {    # survival ladder: transfer the best model into round 1
  local episodes="${1:-15000}"
  local source="${INITIALIZE_FROM:-$(best_checkpoint survival 1 ppo,recurrent_ppo)}"
  if [ -z "$source" ]; then
    echo "No compatible PPO checkpoint found to seed the survival ladder." >&2
    exit 1
  fi
  local output="${OUTPUT:-models/ppo-endless-$(date +%m%d-%H%M)}"
  echo "Initializing the survival ladder from $source"
  CURRICULUM="configs/rl-endless.toml" INITIALIZE_FROM="$source" OUTPUT="$output" \
    cmd_train_ppo 0 "$episodes"
}

# The rung a checkpoint actually belongs on: one above the highest round it can already
# clear. Forking above that leaves the ladder unable to promote; forking below re-teaches
# what the policy already knows.
#
# Do NOT binary search from round 1. Skill is not monotone in the round number -- a policy
# that trained up to round 18 measured 75% on round 1 while passing round 12 comfortably,
# because it specialised on the crowded fields of the later rungs. A search that stops at
# the first failure sends such a policy back to the start. Instead, anchor on the stage the
# checkpoint itself recorded and walk outward from there, which is also cheaper: the anchor
# is usually right or one rung off.
measure_fork_stage() {
  $PY - "$1" "$2" "${FORK_SEEDS:-24}" <<'PYEOF'
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.ppo import require_ppo, _stage_env
from asteroid_survival.rl.evaluation import evaluate_policy

checkpoint, curriculum, seeds = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
_, PPO, _, _, _ = require_ppo()
layout = json.loads((checkpoint / "metadata.json").read_text())["observation_layout"]
spec = load_curriculum(curriculum)
model = PPO.load(str(checkpoint / "model.zip"), device="cpu")
top = len(spec.stages)
memo: dict[int, bool] = {}


def passes(round_number: int) -> bool:
    if round_number in memo:
        return memo[round_number]
    env = _stage_env(spec, round_number - 1, layout)
    scores = [evaluate_policy(
        env, lambda o: int(np.asarray(model.predict(o, deterministic=True)[0]).item()), [seed])
        ["aggregate"] for seed in range(9000, 9000 + seeds)]
    clear = float(np.mean([s["clear_rate"] for s in scores]))
    survival = float(np.mean([s["survival_fraction"] for s in scores]))
    ok = clear >= spec.promotion_clear_rate and survival >= spec.promotion_completion
    print(f"  round {round_number:>3}: clear {clear:5.1%}  survival {survival:5.1%}  "
          f"{'pass' if ok else 'fail'}", file=sys.stderr, flush=True)
    memo[round_number] = ok
    return ok


# The checkpoint knows which rung it was training on; start the search there.
anchor = 1
state = checkpoint / "curriculum_state.json"
if state.is_file():
    try:
        anchor = min(top, max(1, int(json.loads(state.read_text()).get("stage", 0)) + 1))
    except (ValueError, TypeError):
        anchor = 1
print(f"  anchored on round {anchor} (the rung this checkpoint was training on)",
      file=sys.stderr, flush=True)

budget = 15
if passes(anchor):
    highest = anchor
    while highest < top and budget > 0:
        budget -= 1
        if not passes(highest + 1):
            break
        highest += 1
    print(min(top, highest + 1))
else:
    probe = anchor - 1
    while probe >= 1 and budget > 0:
        budget -= 1
        if passes(probe):
            break
        probe -= 1
    print(max(1, probe + 1))
PYEOF
}

cmd_train_ppo_survival_v2() { # fork the protected v1 policy into observation/curriculum v2
  local episodes="${1:-5000}"
  local source="${INITIALIZE_FROM:-models/ppo-survive-0819-1852/champion}"
  if [ ! -d "$source" ]; then
    source="$(best_checkpoint survival 16 ppo,recurrent_ppo)"
  fi
  if [ -z "$source" ] || [ ! -d "$source" ]; then
    echo "No compatible solo PPO checkpoint found for survival-v2." >&2; exit 1
  fi
  local output="${OUTPUT:-models/ppo-survival-v2-$(date +%m%d-%H%M)}"
  echo "Forking survival-v2 from $source (the source run is not modified)"
  local stage="${START_STAGE:-}"
  if [ -z "$stage" ]; then
    echo "Measuring which round this policy already passes..."
    stage="$(measure_fork_stage "$source" configs/rl-survival-v2.toml)"
    echo "Starting at round $stage - the lowest round it does not already clear"
  fi
  # PPO with these small networks is ~3x slower on MPS than on CPU, and `auto` picks MPS.
  CURRICULUM="configs/rl-survival-v2.toml" INITIALIZE_FROM="$source" OUTPUT="$output" \
    PPO_DEVICE="${PPO_DEVICE:-cpu}" START_STAGE="$stage" cmd_train_ppo 0 "$episodes"
}

cmd_train_mappo() {
  assert_no_other_trainer
  local episodes="${1:-1000}" output="${OUTPUT:-models/mappo-team-$(date +%m%d-%H%M)}"
  local args=(--output "$output" --episodes "$episodes" --max-ships "${MAX_SHIPS:-8}"
              --seed "${SEED:-0}" --device "${PPO_DEVICE:-cpu}")
  [ "${PROTECT:-0}" = "1" ] && args+=(--protect)
  [ -n "${INITIALIZE_FROM:-}" ] && args+=(--initialize-from "$INITIALIZE_FROM")
  [ -n "${RESUME:-}" ] && args+=(--resume "$RESUME")
  $PY -m asteroid_survival train-mappo "${args[@]}"
}

cmd_train_team() {
  assert_no_other_trainer
  local steps="${1:-1000000}" output="${OUTPUT:-models/team-ppo-$(date +%m%d-%H%M)}"
  local args=(--output "$output" --steps "$steps" --parallel-envs "${PARALLEL_ENVS:-8}"
              --eval-every "${EVAL_EVERY:-250000}" --eval-episodes "${EVAL_EPISODES:-128}"
              --keep-checkpoints "${KEEP_CHECKPOINTS:-3}"
              --seed "${SEED:-0}" --device "${PPO_DEVICE:-cpu}")
  [ -n "${RESUME:-}" ] && args+=(--resume "$RESUME")
  [ "${STOP_WHEN_MASTERED:-0}" = "1" ] && args+=(--stop-when-mastered)
  $PY -m asteroid_survival train-team "${args[@]}"
}

cmd_test_team() {
  local checkpoint="${1:-${CHECKPOINT:-}}"
  [ -n "$checkpoint" ] || { echo "pass a team checkpoint or set CHECKPOINT" >&2; exit 1; }
  local args=(--checkpoint "$checkpoint" --episodes "${EPISODES:-64}"
              --ships "${SHIPS:-8}" --level "${LEVEL:-12}" --seed "${SEED:-20000}")
  [ "${PROTECT:-0}" = "1" ] && args+=(--protect)
  [ -n "${TEAM_STAGE:-}" ] && args+=(--stage "$TEAM_STAGE")
  $PY -m asteroid_survival evaluate-team "${args[@]}"
}

cmd_play_team() {
  local checkpoint="${1:-${CHECKPOINT:-}}"
  [ -n "$checkpoint" ] || { echo "pass a team checkpoint or set CHECKPOINT" >&2; exit 1; }
  local args=(--checkpoint "$checkpoint" --ships "${SHIPS:-8}"
              --level "${LEVEL:-12}" --seed "${SEED:-7}")
  [ "${PROTECT:-0}" = "1" ] && args+=(--protect)
  [ -n "${TEAM_STAGE:-}" ] && args+=(--stage "$TEAM_STAGE")
  $PY -m asteroid_survival play-team "${args[@]}"
}

cmd_test_ppo() {
  local checkpoint="${1:-${CHECKPOINT:-}}"
  [ -n "$checkpoint" ] || { echo "pass a PPO checkpoint or set CHECKPOINT" >&2; exit 1; }
  local stage="${2:-${ROUND:-}}" output="${OUTPUT:-metrics/ppo-final-test.json}"
  local args=(--checkpoint "$checkpoint" --episodes 128 --seed 10256 --output "$output")
  [ -n "$stage" ] && args+=(--stage "$stage")
  $PY -m asteroid_survival evaluate "${args[@]}"
}

cmd_rounds() {
  local mode="${1:-survival-v2}"
  $PY - "$mode" <<'PYEOF'
import sys
from asteroid_survival.modes import MODES
from asteroid_survival.rl.curriculum import load_curriculum
mode=MODES[sys.argv[1]]
if not mode.curriculum: raise SystemExit(f"{sys.argv[1]} has no rounds")
spec=load_curriculum(mode.curriculum)
print("round  size       patterns linear speed       amplitude wavelength spawn  spread initial")
for i,s in enumerate(spec.stages,1):
    size=s.asteroid_size if s.asteroid_size is not None else "mixed"
    print(f"{i:>5}  {str(size):<10} {len(s.patterns):>8} {s.linear_probability:>6.2f} "
          f"{s.min_speed:>4.0f}-{s.max_speed:<4.0f} {s.amplitude_min:>3.0f}-{s.amplitude_max:<3.0f} "
          f"{s.wavelength_min:>3.1f}-{s.wavelength_max:<3.1f} {s.spawn_interval:>5.2f}s "
          f"{s.spawn_spread:>5.1f} {s.initial_asteroids:>7}")
PYEOF
}

cmd_train_ppo_coop() {       # tier two of the survival ladder: two ships, one policy
  local episodes="${1:-15000}"
  local source="${INITIALIZE_FROM:-$(best_checkpoint survival 53 ppo,recurrent_ppo)}"
  if [ -z "$source" ]; then
    echo "No compatible PPO checkpoint found to seed the co-operative tier." >&2
    exit 1
  fi
  local output="${OUTPUT:-models/ppo-coop-$(date +%m%d-%H%M)}"
  echo "Initializing the two-ship tier from $source"
  CURRICULUM="configs/rl-endless-coop.toml" INITIALIZE_FROM="$source" OUTPUT="$output" \
    START_STAGE="${START_STAGE:-97}" cmd_train_ppo 0 "$episodes"
}

cmd_train_ppo_nonlinear() {  # phase 2: transfer the best FF-PPO into stronger curves
  local episodes="${1:-15000}"
  local source="${INITIALIZE_FROM:-$(best_checkpoint round 48 ppo)}"
  if [ -z "$source" ]; then
    echo "No compatible feed-forward PPO checkpoint found for nonlinear initialization." >&2
    exit 1
  fi
  local output="${OUTPUT:-models/ppo-nonlinear-$(date +%m%d-%H%M)}"
  echo "Initializing nonlinear PPO from $source"
  CURRICULUM="configs/rl-nonlinear.toml" INITIALIZE_FROM="$source" START_STAGE=29 OUTPUT="$output" \
    cmd_train_ppo 0 "$episodes"
}

cmd_train_fast() {    # lower-cost MuZero search/training; curriculum and rewards are unchanged
  SIMULATIONS="${SIMULATIONS:-24}" \
  UPDATES_PER_EPISODE="${UPDATES_PER_EPISODE:-16}" \
  cmd_train "${1:-10000}"
}

cmd_continue() {      # continue one run without losing replay, stage, or logs
  local output="${1:-$(ls -dt models/*/ 2>/dev/null | head -1)}"
  local episodes="${2:-10000}"
  output="${output%/}"
  if [ -z "$output" ] || [ ! -d "$output" ]; then
    echo "Run directory not found: $output" >&2; exit 1
  fi
  local checkpoint
  checkpoint="$($PY - "$output" <<'PYEOF'
import sys
from pathlib import Path
paths = sorted(Path(sys.argv[1]).glob("checkpoint_*"))
if paths:
    print(paths[-1])
PYEOF
)"
  if [ -z "$checkpoint" ]; then
    echo "No checkpoint found in $output" >&2; exit 1
  fi
  local saved_simulations saved_updates
  read -r saved_simulations saved_updates <<< "$($PY - "$checkpoint" <<'PYEOF'
import json, sys
from pathlib import Path
settings = json.loads((Path(sys.argv[1]) / "metadata.json").read_text())["settings"]
print(int(settings.get("num_simulations", 50)), int(settings.get("updates_per_episode", 32)))
PYEOF
)"
  SIMULATIONS="${SIMULATIONS:-$saved_simulations}" \
  UPDATES_PER_EPISODE="${UPDATES_PER_EPISODE:-$saved_updates}" \
  OUTPUT="$output" RESUME="$checkpoint" cmd_train "$episodes"
}

cmd_finish() {        # continue until every curriculum stage is mastered (or safety cap)
  local output="${1:-$(ls -dt models/*/ 2>/dev/null | head -1)}"
  local episodes="${2:-100000}"
  if [ -n "$output" ] && [ -d "$output" ]; then
    STOP_WHEN_MASTERED=1 cmd_continue "$output" "$episodes"
  else
    OUTPUT="$output" STOP_WHEN_MASTERED=1 cmd_train "$episodes"
  fi
}

cmd_graph() {
  local dir view
  case "${1:-}" in
    completion|survival|both)
      dir="$(ls -dt models/*/ 2>/dev/null | head -1)"
      view="$1"
      ;;
    *)
      dir="${1:-$(ls -dt models/*/ 2>/dev/null | head -1)}"
      view="${2:-both}"
      ;;
  esac
  dir="${dir%/}"
  $PY -m asteroid_survival graph --run "$dir" --view "$view" \
    --height "${GRAPH_HEIGHT:-20}"
}

cmd_pull() {         # copy a run's champion down from the training box, to preview locally
  # Training runs headless on the remote box; previewing needs a display, so the champion has
  # to come here. Only the champion, its state and the evaluation log: training.jsonl is
  # hundreds of megabytes of per-episode noise and is never needed locally.
  local remote="${REMOTE:-oracle}"
  local run="${1:-}"
  if [ -z "$run" ]; then
    run="$(ssh "$remote" 'ls -dt ~/Asteroids/models/*/ 2>/dev/null | head -1' \
           | sed 's:/$::;s:.*/models/:models/:')"
    [ -n "$run" ] || { echo "could not work out the newest run on $remote" >&2; exit 1; }
    echo "newest run on $remote: $run"
  fi
  run="${run%/}"
  mkdir -p "$run"
  echo "pulling $run from $remote ..."
  scp -q -r "$remote:~/Asteroids/$run/champion" "$run/" || {
    echo "no champion in $run on $remote yet" >&2; exit 1; }
  for extra in champion_state.json evaluation.jsonl curriculum_state.json; do
    scp -q "$remote:~/Asteroids/$run/$extra" "$run/" 2>/dev/null || true
  done
  # The champion is the best-scoring checkpoint, which can be far behind the live policy --
  # it only updates when a score improves, so after a promotion it can sit still for
  # thousands of episodes. `latest` fetches the newest checkpoint instead, which is what to
  # watch when the question is "how do they play now".
  if [ "${2:-}" = "latest" ]; then
    local newest
    newest="$(ssh "$remote" "ls -d ~/Asteroids/$run/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1")"
    if [ -n "$newest" ]; then
      scp -q -r "$remote:$newest" "$run/" && echo "  also pulled $(basename "$newest")"
    fi
  fi
  $PY - "$run" <<'PYEOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
meta = json.loads((run / "champion" / "metadata.json").read_text(encoding="utf-8"))
if meta.get("algorithm") == "centralized_team_ppo":
    print(f"  {int(meta.get('environment_steps', 0)):,} decisions | "
          f"centralized two-ship policy | stage {int(meta.get('stage', 0)) + 1}")
    print(f"  watch champion: ./run.sh preview {run}")
    checkpoints = sorted(run.glob("checkpoint_*"))
    if checkpoints:
        print(f"  watch latest:   ./run.sh preview {checkpoints[-1]}")
    raise SystemExit
layout = meta.get("observation_layout", {})
ships = int(layout.get("max_teammates", 0)) + 1
print(f"  episode {meta['episodes']:,} | {meta['environment_steps']:,} decisions "
      f"| observation {meta['observation_size']} | {ships} ship{'s' if ships > 1 else ''}")
print(f"  watch it:  ./run.sh preview {run}")
PYEOF
}

cmd_preview() {      # watch the best held-out checkpoint in a run (or an exact checkpoint)
  local target="${1:-$(running_run)}"
  [ -n "$target" ] || target="$(ls -dt models/*/ 2>/dev/null | head -1)"
  target="${target%/}"
  # A fresh seed each time, so repeated previews are not the same asteroid layout over and
  # over. Pass a seed to pin one, e.g. SEED=10000 to watch what the held-out score measures.
  local round="${2:-}"
  local seed="${3:-${SEED:-$RANDOM$RANDOM}}"
  echo "preview seed: $seed  (N = next seed, R = replay this one, Esc = quit)"
  # Centralized team checkpoints use a different environment and action space from legacy
  # solo/shared PPO. Dispatch them through play-team while keeping the familiar preview
  # command working for a run directory or an exact checkpoint.
  local team_checkpoint=""
  if [ -f "$target/metadata.json" ]; then
    team_checkpoint="$target"
  elif [ -f "$target/champion/metadata.json" ]; then
    team_checkpoint="$target/champion"
  fi
  if [ -n "$team_checkpoint" ] && $PY - "$team_checkpoint/metadata.json" <<'PYEOF'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("algorithm") ==
                 "centralized_team_ppo" else 1)
PYEOF
  then
    SEED="$seed" cmd_play_team "$team_checkpoint"
    return
  fi
  $PY -m asteroid_survival preview "$target" --seed "$seed" ${round:+--stage "$round"}
}

cmd_status() {        # how is training going
  local dir="${1:-$(running_run)}"
  # Newest directory, but skip runs that never produced a held-out evaluation. A run that
  # was killed early leaves a directory behind and would otherwise become the default,
  # reporting a corpse as the current state of training.
  if [ -z "$dir" ]; then
    local candidate
    for candidate in $(ls -dt models/*/ 2>/dev/null); do
      if [ -s "${candidate%/}/evaluation.jsonl" ]; then dir="$candidate"; break; fi
    done
    [ -n "$dir" ] || dir="$(ls -dt models/*/ 2>/dev/null | head -1)"
  fi
  dir="${dir%/}"
  if [ -z "$dir" ] || [ ! -d "$dir" ]; then
    echo "No model directory found yet." >&2; exit 1
  fi
  echo "== $dir =="
  if [ $# -eq 0 ]; then
    local others
    others="$(ls -dt models/*/ 2>/dev/null | sed 's:/$::' | grep -v "^${dir}$" | head -3 | tr '\n' ' ')"
    [ -n "$others" ] && echo "other runs: $others(pass one to ./run.sh status)"
  fi
  local training_pid
  training_pid="$($PY - "$dir" <<'PYEOF'
import shlex, subprocess, sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
for raw in subprocess.check_output(
        ["ps", "ax", "-o", "pid=,command="], text=True).splitlines():
    fields = raw.strip().split(maxsplit=1)
    if len(fields) != 2:
        continue
    pid, command = fields
    try:
        args = shlex.split(command)
        output = args[args.index("--output") + 1]
    except (ValueError, IndexError):
        continue
    if ("asteroid_survival" in args
            and any(name in args for name in
                    ("train", "train-ppo", "train-mappo", "train-team"))
            and Path(output).resolve() == target):
        print(pid)
        break
PYEOF
)"
  if [ -n "$training_pid" ]; then
    echo "training: RUNNING (pid $training_pid)"
  else
    echo "training: not running"
  fi
  # Data-collection jobs are not trainers, so the check above cannot see them and the box
  # reads as idle when it is in fact saturated. Report them explicitly.
  local oracle_pid
  # `|| true`: pgrep exits 1 when nothing matches, and under `set -euo pipefail` that
  # aborts the whole status command -- which is exactly when you most want it to print.
  oracle_pid="$(pgrep -f "planning_oracle.py" 2>/dev/null | head -1 || true)"
  if [ -n "$oracle_pid" ]; then
    # Newest log wins: several of these accumulate and a finished run's last line would
    # otherwise be reported as the live job's progress.
    local oracle_log oracle_progress=""
    oracle_log="$(ls -t oracle-*.log 2>/dev/null | head -1)"
    [ -n "$oracle_log" ] && oracle_progress="$(grep -E "episodes|clear" "$oracle_log" | tail -1)"
    echo "planning oracle: RUNNING (pid $oracle_pid)${oracle_progress:+ | ${oracle_progress# }}"
  fi
  if [ -f "$dir/training.jsonl" ]; then
    local progress
    progress="$($PY - "$dir/training.jsonl" <<'PYEOF'
import json, sys
from pathlib import Path
lines = Path(sys.argv[1]).read_text().splitlines()
latest = json.loads(lines[-1]) if lines else {}
if "episode" in latest:
    detail = f"{latest['episode']} total ({len(lines)} recorded in this directory)"
else:
    detail = f"{len(lines)} total"
if latest.get("environment_steps") is not None:
    detail += f" | {int(latest['environment_steps']):,} environment decisions"
print(detail)
PYEOF
)"
    echo "episodes: $progress"
  else
    echo "episodes: none yet (a fresh run writes its first block at 250 episodes)"
  fi
  $PY - "$dir" <<'PYEOF'
import json, sys
from pathlib import Path
checkpoints = sorted(Path(sys.argv[1]).glob("checkpoint_*"))
if checkpoints:
    try:
        metadata = json.loads((checkpoints[-1] / "metadata.json").read_text())
    except (OSError, ValueError):
        metadata = {}
    algorithm = metadata.get("algorithm")
    if algorithm in {"ppo", "recurrent_ppo"}:
        label = "LSTM-PPO" if metadata.get("recurrent") else "PPO"
        print("learner: {} | device {} | throughput {:.0f} decisions/s".format(
            label, metadata.get("device", "unknown"),
            float(metadata.get("decisions_per_second", 0.0))))
PYEOF
  if [ ! -s "$dir/evaluation.jsonl" ]; then
    local every
    every="$(ps -eo command= 2>/dev/null | sed -n "s/.*--eval-every \([0-9]*\).*/\1/p" | head -1 || true)"
    echo "held-out evaluations: none yet${every:+ (first at episode $every)}"
  fi
  if [ -s "$dir/evaluation.jsonl" ]; then
    if [ -f "$dir/champion_state.json" ]; then
      $PY - "$dir/champion_state.json" "$dir/evaluation.jsonl" <<'PYEOF'
import json, sys
from pathlib import Path
s = json.load(open(sys.argv[1]))
clear = s.get("clear_rate")
clear_detail = "" if clear is None else " | clear {:.1%}".format(clear)
# The round's name, not its global index. A tier that extends another starts at the parent's
# length, so co-operative round 1 has index 96 and printing "stage 97" looks like a bug.
stage_index = int(s.get("training_stage", 0))
label = str(stage_index + 1)
try:
    for line in reversed(Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        name = record["stages"][stage_index].get("name")
        if name:
            label = name
        break
except (OSError, ValueError, IndexError, KeyError):
    pass
print("champion: episode {} | {} | completion {:.1%}{} | accuracy {:.3f} | recoveries {} | retention alerts {}/{}".format(
    s.get("episode", 0), label,
    s.get("completion_rate", 0), clear_detail, s.get("accuracy", 0), s.get("recoveries", 0),
    s.get("retention_failures", 0), s.get("patience", 4)))
print("champion restorations: {}".format(s.get("restorations", 0)))
if s.get("rollbacks", 0):
    print("historical destructive rollbacks (disabled now): {}".format(s["rollbacks"]))
PYEOF
    fi
    # The promotion decision, which is pooled across a window of evaluations and is not
    # visible in any single one of them. `streak` stopped meaning "passes in a row" when
    # pooling went in -- it counts evaluations banked in the window -- so it is reported as
    # a window fill level, never as progress toward the bar.
    $PY - "$dir/evaluation.jsonl" <<'PYEOF'
import json, sys
from pathlib import Path
lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
record = None
for line in reversed(lines):
    try:
        record = json.loads(line)
        break
    except ValueError:
        continue
if record:
    pool = record.get("promotion_pool")
    clear_target = record.get("promotion_clear_rate_target")
    completion_target = record.get("promotion_completion_target")
    if pool:
        clear = pool.get("clear_rate", 0.0)
        completion = pool.get("completion_rate", 0.0)
        blockers = []
        if clear_target is not None and clear < clear_target:
            blockers.append("clear {:+.1f} pts".format((clear - clear_target) * 100))
        if completion_target is not None and completion < completion_target:
            blockers.append("completion {:+.1f} pts".format(
                (completion - completion_target) * 100))
        line = "promotion: pooled over {}/{} evaluations ({} episodes)".format(
            pool.get("evaluations", 0), pool.get("required_evaluations", 0),
            pool.get("episodes", 0))
        print(line)
        print("  clear      {:6.1%}{}".format(
            clear, "" if clear_target is None else " of {:.0%}".format(clear_target)))
        print("  completion {:6.1%}{}".format(
            completion,
            "" if completion_target is None else " of {:.0%}".format(completion_target)))
        print("  " + ("SHORT BY " + ", ".join(blockers) if blockers
                      else "both gates met - promotes on the next evaluation"))
    elif record.get("current"):
        current = record["current"]
        success = float(current.get("success_rate", 0.0))
        target = float(record.get("target", 0.0))
        state = "PROMOTED" if record.get("promoted") else "waiting"
        candidate = record.get("candidate_action")
        detail = "" if not candidate else " | candidate {} | best {:.1%} | lr {:.2g}".format(
            candidate, record.get("best_success_rate", success),
            record.get("effective_learning_rate", 0.0))
        print("promotion: {} | success {:.1%} of {:.0%} | retention {}{}".format(
            state, success, target,
            "passed" if record.get("retention_ok", True) else "failed", detail))
    else:
        streak = record.get("promotion_streak", 0)
        print("promotion: not pooled | streak {} (single-evaluation gate)".format(streak))
PYEOF
    # The critic, which is what the current experiments turn on. A value function that
    # explains less than half the return variance makes every advantage estimate noisy.
    if [ -f "$dir/ppo_updates.jsonl" ]; then
      $PY - "$dir/ppo_updates.jsonl" <<'PYEOF'
import json, statistics, sys
from pathlib import Path
rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[-200:]:
    try:
        rows.append(json.loads(line))
    except ValueError:
        continue
if rows:
    def mean(name, default=None):
        values = [float(r[name]) for r in rows if name in r]
        return statistics.fmean(values) if values else default
    entropy = mean("entropy")
    if entropy is None:
        loss = mean("entropy_loss")
        entropy = None if loss is None else -loss
    parts = ["explained_variance {:.3f}".format(mean("explained_variance", 0.0))]
    parts.append("value_loss {:.2f}".format(mean("value_loss", 0.0)))
    if entropy is not None:
        parts.append("entropy {:.3f} nats".format(entropy))
    parts.append("approx_kl {:.4f}".format(mean("approx_kl", 0.0)))
    print("critic/policy (last {} updates): {}".format(len(rows), " | ".join(parts)))
PYEOF
    fi
    echo "-- held-out evaluations (the number that matters) --"
    tail -5 "$dir/evaluation.jsonl" | $PY -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if "current" in r:
        s = r["current"]
        steps = int(r.get("environment_steps", 0))
        line = "  step {:>9,}  {:<28} success {:6.1%} of {:.0%}".format(
            steps, s.get("name", "team stage"), s.get("success_rate", 0.0),
            r.get("target", 0.0))
        line += "  kills {:5.2f}  friendly fire {:.3f}".format(
            s.get("mean_asteroids_destroyed", 0.0), s.get("mean_friendly_fire", 0.0))
        if r.get("promoted"):
            line += "  <- PROMOTED"
        elif r.get("candidate_action"):
            line += "  <- {} (best {:.1%})".format(
                r["candidate_action"].upper(), r.get("best_success_rate", 0.0))
        print(line)
        continue
    episode = r.get("episode", 0)
    if "stages" in r:
        index = min(r.get("training_stage", 0), len(r["stages"]) - 1)
        s = r["stages"][index]
        completion_target = r.get("promotion_completion_target")
        clear_target = r.get("promotion_clear_rate_target")
        label = s.get("name") or "round {}".format(index + 1)
        line = "  ep {:>6}  {:<18} completion {:6.1%}{}".format(
            episode, label, s["completion_rate"],
            "" if completion_target is None else " of {:.0%}".format(completion_target))
        clear = s.get("clear_rate")
        if clear is not None:
            line += "  clear {:6.1%}{}".format(
                clear, "" if clear_target is None else " of {:.0%}".format(clear_target))
        if s.get("mean_survival_time") is not None:
            line += "  survival {:5.1f}s".format(s["mean_survival_time"])
        line += "  accuracy {:.3f}".format(s["mean_accuracy"])
        pool = r.get("promotion_pool")
        if pool:
            line += "  pool {:.1%}".format(pool.get("clear_rate", 0.0))
        if r.get("promoted"):
            line += "  <- PROMOTED"
        print(line)
        continue
    survival = r["mean_survival_time"]
    kills = r["mean_asteroids_destroyed"]
    accuracy = r["mean_accuracy"]
    print(f"  ep {episode:>6}  survival {survival:6.2f}s  kills {kills:5.2f}  accuracy {accuracy:.3f}")
'
  fi
}

cmd_follow() {        # stream held-out evaluations as they land, instead of re-running status
  local dir="${1:-$(running_run)}"
  [ -n "$dir" ] || dir="$(ls -dt models/*/ 2>/dev/null | head -1)"
  dir="${dir%/}"
  if [ -z "$dir" ] || [ ! -d "$dir" ]; then
    echo "No model directory found yet." >&2; exit 1
  fi
  cmd_status "$dir"
  echo
  echo "-- following $dir (Ctrl-C to stop) --"
  $PY - "$dir" "${FOLLOW_EVERY:-15}" <<'PYEOF'
import json, shlex, subprocess, sys, time
from pathlib import Path

run = Path(sys.argv[1]).resolve()
interval = float(sys.argv[2])
log = run / "evaluation.jsonl"


def trainer_alive():
    """True while a trainer process is writing to this directory.

    Checked by --output rather than by a saved pid: the run may have been launched from
    another shell, and a stale pid file would report a dead run as healthy.
    """
    for raw in subprocess.check_output(["ps", "ax", "-o", "command="], text=True).splitlines():
        try:
            args = shlex.split(raw.strip())
            output = args[args.index("--output") + 1]
        except (ValueError, IndexError):
            continue
        if ("asteroid_survival" in args
                and any(name in args for name in
                        ("train", "train-ppo", "train-mappo", "train-team"))
                and Path(output).resolve() == run):
            return True
    return False


def records():
    if not log.is_file():
        return []
    out = []
    for line in log.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def describe(record, previous):
    if record.get("current"):
        stage = record["current"]
        line = (f"  step {record.get('environment_steps', 0):>9,}  "
                f"{stage.get('name', 'team stage'):<28}"
                f"  success {stage.get('success_rate', 0.0):6.1%}"
                f" of {record.get('target', 0.0):.0%}"
                f"  kills {stage.get('mean_asteroids_destroyed', 0.0):5.2f}"
                f"  friendly fire {stage.get('mean_friendly_fire', 0.0):.3f}")
        if record.get("promoted"):
            line += "  <- PROMOTED"
        return line
    stage_index = min(record.get("training_stage", 0), len(record.get("stages", [])) - 1)
    stage = record["stages"][stage_index] if record.get("stages") else {}
    target = record.get("promotion_completion_target")
    completion = stage.get("completion_rate")
    label = stage.get("name") or f"round {stage_index + 1}"
    line = f"  ep {record.get('episode', 0):>6}  {label:<18}"
    if completion is not None:
        gap = "" if target is None else f" of {target:.0%}"
        line += f"  completion {completion:6.1%}{gap}"
    clear = stage.get("clear_rate")
    if clear is not None:
        clear_target = record.get("promotion_clear_rate_target")
        gap = "" if clear_target is None else f" of {clear_target:.0%}"
        line += f"  clear {clear:6.1%}{gap}"
    if stage.get("mean_survival_time") is not None:
        line += f"  survival {stage['mean_survival_time']:5.1f}s"
    if stage.get("mean_accuracy") is not None:
        line += f"  accuracy {stage['mean_accuracy']:.3f}"
    streak = record.get("promotion_streak")
    pool = record.get("promotion_pool")
    if pool:
        # The pooled rates are the promotion decision; the per-evaluation numbers above are
        # a single noisy draw and cannot promote on their own.
        line += (f"  pool {pool.get('evaluations', 0)}/"
                 f"{pool.get('required_evaluations', 0)}"
                 f" [{pool.get('clear_rate', 0.0):.1%} clear,"
                 f" {pool.get('completion_rate', 0.0):.1%} completion,"
                 f" {pool.get('episodes', 0)} eps]")
    elif streak:
        line += f"  streak {streak}"
    marks = []
    if record.get("promoted"):
        marks.append("PROMOTED")
    elif previous is not None and record.get("training_stage", 0) < previous.get("training_stage", 0):
        marks.append("FELL BACK")
    if record.get("champion_action") == "improved":
        marks.append("new champion")
    elif record.get("champion_action") == "restored":
        marks.append("champion restored")
    if marks:
        line += "  <- " + ", ".join(marks)
    return line


# The status block above has already printed the last few evaluations, so start from
# what is on disk now and print only what arrives after it. Re-printing the newest
# record here showed the same episode twice in two different formats.
seen = records()
previous = seen[-1] if seen else None
count = len(seen)
print("  waiting for the next evaluation...", flush=True)

# Poll rather than watch the filesystem: evaluations are minutes apart, so a timer is
# simpler than an event API and costs nothing measurable.
while True:
    try:
        time.sleep(interval)
    except KeyboardInterrupt:
        break
    current = records()
    for record in current[count:]:
        print(describe(record, previous), flush=True)
        previous = record
    if len(current) > count:
        count = len(current)
    elif not trainer_alive():
        print("  -- trainer is no longer running --", flush=True)
        break
PYEOF
}

cmd_test() {          # run the test suite
  $PY -m pytest -q
}

usage() {
  cat <<'EOF'
Asteroid Survival

Modes -- every play/showdown/watch/compare command takes the same four:
  arcade            clear-the-wave arcade play
  endless           one run, difficulty ramps forever
  round N           nonlinear curriculum round, 1-48
  survival N        survival ladder round, 1-96
  survival-v2 N     nonlinear-first survival ladder, 1-96
  varied N          survival v2 plus varied rounds 97-100, 1-100

Play
  ./run.sh play [mode] [N]      play it yourself           (default: arcade)
  ./run.sh showdown [mode] [N]  you + greedy + your newest model, one shared arena
  ./run.sh watch [mode] [N]     agents only, scored on fixed seeds
  ./run.sh compare [mode] [N]   you and the agents on the same seeds, scored
  ./run.sh versus [mode] [N] [runs]  score several models, each alone in its own game
  ./run.sh pull [dir] [latest]  copy a run's champion (or newest checkpoint) down to here
  ./run.sh preview [dir] [N]    watch a run's champion, optionally on round N
  ./run.sh patterns [name]      watch trajectory shapes with trails (omit name for all)

  ./run.sh play survival 12     an endless-ladder round
  ./run.sh showdown round 48    the hardest nonlinear round, three ships
  ./run.sh watch survival 8 20  twenty scored seeds on ladder round 8

Train
  ./run.sh train [N]            MuZero, mastery curriculum (default 10000)
  ./run.sh train-fast [N]       MuZero with smaller search/update budgets
  ./run.sh train-ppo [N]        feed-forward PPO
  ./run.sh train-lstm [N]       recurrent LSTM-PPO
  ./run.sh train-ppo-nonlinear [N]  transfer FF-PPO into stronger curved motion
  ./run.sh train-ppo-endless [N]    transfer FF-PPO into the survival ladder
  ./run.sh train-ppo-coop [N]       transfer a solo model into the two-ship tier
  ./run.sh train-ppo-survival-v2 [N] fork a solo model into survival v2 (rung measured, CPU)
  ./run.sh train-team [STEPS]        centralized joint policy for the two-ship objective
  ./run.sh train-mappo-team [N]      shared policy, centralized team critic, 1-8 ships
  PROTECT=1 ./run.sh train-mappo-team [N]  gated object-protection variant
  ./run.sh ppo-screen [N]       PPO then LSTM-PPO sequentially on one seed
  ./run.sh continue DIR [N]     continue a run with replay/stage/logs intact
  ./run.sh finish DIR [N]       continue until every stage passes

Inspect
  ./run.sh status [dir]         training progress, newest run by default
  ./run.sh follow [dir]         status once, then stream evaluations as they land
  ./run.sh graph [dir] [view]   terminal graph (`both`, `completion`, or `survival`)
  ./run.sh baseline [N]         score the greedy controller
  ./run.sh test                 run the test suite
  ./run.sh test-team CHECKPOINT score a shared team policy (SHIPS/LEVEL overrides)
  ./run.sh play-team CHECKPOINT watch N copies of the shared actor (SHIPS/LEVEL)
  ./run.sh test-ppo CHECKPOINT [ROUND]  run the untouched 128-seed PPO test panel
  ./run.sh rounds survival-v2   print every round's exact difficulty

Overrides (environment variables)
  SEED=11           starting seed
  CHECKPOINT=path   use a specific model instead of the auto-selected one
  ANY_RUN=1         auto-select across every run by score, not just the newest
  ENCODER=set       permutation-invariant encoder over the asteroid/projectile sets
  MODELS=a,b        checkpoints for `versus` (default: the newest run's champion)
  HUMAN=1           include yourself in `versus`
  GREEDY=0          leave the greedy baseline out of `versus`
  PILOT=0           leave the scripted pilot baseline out
  ALGO=muzero       showdown against MuZero instead of PPO
  CONFIG=path       raw config for compare/watch/baseline when no mode is given
  OUTPUT=path       where to write a new training run
  CURRICULUM=path   curriculum for training (default configs/rl-curriculum.toml)
  HISTORY_FRAMES=8  asteroid history frames; must match how the model was trained
  SIMULATIONS=24    tree-search simulations per decision
  UPDATES_PER_EPISODE=16  gradient batches after each episode
  INITIALIZE_FROM=path  transfer weights while resetting task-specific training
  LEARNING_RATE=0.00025 Adam rate for a fresh or policy-initialized run
  START_STAGE=5     one-based curriculum stage for a continuation
  EVAL_EVERY=500    held-out evaluation interval
  PPO_DEVICE=cpu    override PPO device (auto, cpu, or mps)
  TEAM_STAGE=7      one-based stage for test-team/play-team on a centralized checkpoint
  PPO_ENT_COEF=0.0025 entropy bonus for PPO, including initialized/resumed runs
  PPO_VF_COEF=1.0   value-loss coefficient for PPO, including resumed runs
  PPO_ENTROPY_FLOOR=0.8 hold PPO entropy near N nats by adapting the entropy bonus
  FOLLOW_EVERY=15   seconds between `follow` polls
  FORK_SEEDS=24     seeds per probe when measuring a fork rung
  ALLOW_CONCURRENT=1 bypass the trainer process guard

Aliases kept for muscle memory: `endless` = `play endless`,
`play-round N` = `play round N`, `play-endless N` = `play survival N`.
EOF
}

menu() {
  echo "Asteroid Survival"
  echo "  1) play arcade"
  echo "  2) play endless"
  echo "  3) play a survival ladder round"
  echo "  4) showdown - me vs greedy vs the best model"
  echo "  5) watch the agents, scored"
  echo "  6) training status"
  echo "  7) score the greedy baseline"
  echo "  8) train a model"
  echo
  read -r -p "pick [1-8]: " choice
  case "$choice" in
    1) cmd_play arcade ;;
    2) cmd_play endless ;;
    3) read -r -p "round [1-96]: " round; cmd_play survival "${round:-1}" ;;
    4) cmd_showdown ;;
    5) cmd_watch ;;
    6) cmd_status ;;
    7) cmd_baseline ;;
    8) cmd_train ;;
    *) echo "nothing picked"; exit 1 ;;
  esac
}

command="${1:-}"
[ $# -gt 0 ] && shift || true
case "$command" in
  "")          menu ;;
  play)        cmd_play "$@" ;;
  showdown)    cmd_showdown "$@" ;;
  patterns)    cmd_patterns "$@" ;;
  endless)      cmd_play endless ;;                 # alias
  play-round)   cmd_play round "${1:-1}" ;;         # alias
  play-endless) cmd_play survival "${1:-1}" ;;      # alias
  compare)     cmd_compare "$@" ;;
  versus)      cmd_versus "$@" ;;
  watch)       cmd_watch "$@" ;;
  train)       cmd_train "$@" ;;
  train-fast)  cmd_train_fast "$@" ;;
  train-ppo)   cmd_train_ppo 0 "${1:-10000}" ;;
  train-lstm)  cmd_train_ppo 1 "${1:-10000}" ;;
  train-ppo-nonlinear) cmd_train_ppo_nonlinear "${1:-15000}" ;;
  train-ppo-endless)   cmd_train_ppo_endless "${1:-15000}" ;;
  train-ppo-survival-v2) cmd_train_ppo_survival_v2 "${1:-5000}" ;;
  train-ppo-coop)      cmd_train_ppo_coop "${1:-15000}" ;;
  train-team)          cmd_train_team "${1:-1000000}" ;;
  train-mappo-team)    cmd_train_mappo "${1:-1000}" ;;
  test-team)           cmd_test_team "$@" ;;
  play-team)           cmd_play_team "$@" ;;
  test-ppo)            cmd_test_ppo "$@" ;;
  rounds)              cmd_rounds "${1:-survival-v2}" ;;
  ppo-screen)  cmd_ppo_screen "${1:-10000}" ;;
  continue)    cmd_continue "$@" ;;
  finish)      cmd_finish "$@" ;;
  status)      cmd_status "$@" ;;
  follow)      cmd_follow "$@" ;;
  baseline)    cmd_baseline "$@" ;;
  graph)       cmd_graph "$@" ;;
  pull)        cmd_pull "$@" ;;
  preview)     cmd_preview "$@" ;;
  test)        cmd_test "$@" ;;
  -h|--help|help) usage ;;
  *)           echo "unknown command: $command" >&2; echo; usage; exit 1 ;;
esac
