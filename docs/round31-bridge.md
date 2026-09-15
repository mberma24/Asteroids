# The round-31 bridge — prepared 2026-09-15, not yet deployed

Round 31 takes the rock mix from three-quarters large straight to all large. Measured across
four champions on 64 held-out seeds per round, that costs **-0.102 clear [-0.188, -0.012]** —
the largest single-round drop on the ladder — and v21's champion plays round 31 at **0.516**
against a 0.75 gate. Rounds 29-30 already bridge the previous composition step for exactly
this reason (`a04da3b`), and this applies the same fix one step later.

`configs/rl-survival-v3-bridge31.toml` inserts one phase: rounds 31-32 run seven-eighths
large, and the full all-large step moves to round 33. **No rounds are added or renumbered**,
and no speed/amplitude/period ramp changes — `stepped()` depends only on the round index.

## Why it is a separate file and not an edit

`task_hash` digests every stage of the ladder, and `--resume` rejects a mismatch. The service
resumes from its newest checkpoint on every restart, so editing `configs/rl-survival-v3.toml`
while v21 runs would crash-loop it on the next reboot or OOM kill. The live files are
therefore untouched; `test_bridge31_matches_v3_everywhere_except_rounds_31_32` pins the copy
against the original so it cannot drift.

## Before deploying — three checks

Deploy only once v21 has actually promoted to round 31.

```bash
ssh oracle && cd ~/Asteroids
# 1. It really promoted, and to round 31 rather than anywhere else.
./run.sh status models/oracle-survival-v3-v21-action-fire | grep -E "champion|Stage|promotion"
grep -c '"promoted": true' models/oracle-survival-v3-v21-action-fire/evaluation.jsonl

# 2. How the champion actually plays round 31 and the bridged rounds, on held-out seeds.
#    If it already clears round 31 near 0.70, the bridge is unnecessary -- do not deploy.
PYTHONPATH=src .venv/bin/python scripts/benchmark_checkpoint.py \
  models/oracle-survival-v3-v21-action-fire/champion /tmp/r31.json \
  --names retention31 --seed 1000004000 --count 64 --workers 2

# 3. Nothing else is competing for the box.
systemctl is-active asteroids ; pgrep -fa "planning_oracle|benchmark_checkpoint"
```

## Deploying

```bash
ssh oracle && cd ~/Asteroids
mkdir -p experiments/bridge31
# A copy, so the live v21 run stays intact as the rollback target.
cp -a models/oracle-survival-v3-v21-action-fire/champion experiments/bridge31/source
sudo cp /etc/systemd/system/asteroids.service.d/bridge.conf experiments/bridge31/v21-bridge.conf
sudo systemctl stop asteroids
sudo install -m 644 cloud/asteroids-v22-bridge31.conf \
  /etc/systemd/system/asteroids.service.d/bridge.conf
sudo systemctl daemon-reload && sudo systemctl start asteroids
./run.sh status models/oracle-survival-v3-v22-bridge31
```

The first evaluation should land near the round-31 score from check 2, since the policy is
transferred unchanged; a much lower reading means the wrong `START_STAGE` or curriculum.

## Rolling back

```bash
sudo systemctl stop asteroids
sudo install -m 644 experiments/bridge31/v21-bridge.conf \
  /etc/systemd/system/asteroids.service.d/bridge.conf
sudo systemctl daemon-reload && sudo systemctl start asteroids
```

v21 resumes from its own newest checkpoint; nothing in `models/oracle-survival-v3-v21-action-fire`
is modified by the fork.

## How to judge it

On the frozen benchmark (`scripts/benchmark_checkpoint.py`, `current` = round 29, 256 seeds
from 1,000,000,832), not on training clear rates. The v21 champion scores 0.746 there. The
bridge is working if the bridged run holds that and improves on round 31; it is failing if
round-29 performance slides, which is what happened to v16 on a rung it could not hold.
