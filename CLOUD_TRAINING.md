# Training somewhere else

The solo PPO workload is **Python physics plus a small network**, so it is bound by CPU
cores, not by matrix throughput. This is measured, not assumed: the same run does 799
decisions/s on CPU and 294 on Apple MPS, and stable-baselines3 prints a warning saying so.

**A free GPU is worth nothing here.** Rank free offers by cores and by how long a session
survives, and ignore the accelerator.

| Where | Cores | Session limit | Throughput | Notes |
|---|---:|---|---:|---|
| This Mac | 10 | none | **799/s** (measured) | the baseline |
| Oracle Cloud Always Free | 4 ARM | **none** | ~300-400/s (estimated) | free permanently, runs 24/7 |
| Kaggle Notebooks | 4 | 12h, ~30h/week | ~300-400/s (estimated) | documented weekly quota |
| Colab free | ~2 | ~12h, idle-disconnects | ~150-250/s (estimated) | its selling point is a GPU you cannot use |

Cloud figures scale the measured per-core rate; only the Mac number is measured directly.

## Oracle Cloud Always Free (recommended)

4 Ampere A1 cores and 24 GB, free permanently, with **no session limit** -- the only free
tier that runs a multi-day job unattended. Roughly half the Mac's rate, but it trains while
the Mac is asleep, so it adds throughput rather than replacing it.

```bash
ssh ubuntu@YOUR_INSTANCE_IP
git clone https://github.com/mberma24/Asteroids.git
cd Asteroids
./cloud/setup.sh
```

`cloud/setup.sh` installs Python 3.12 and the CPU-only torch wheel, runs the test suite as a
self-check, and starts training inside `tmux` so an SSH disconnect does not kill it. Attach
with `tmux attach -t asteroids`, detach with Ctrl-B then D.

To fork from a policy you already trained, copy its champion up first (9 MB):

```bash
scp -r models/ppo-survival-v2-0819-2258/champion \
       ubuntu@YOUR_INSTANCE_IP:~/Asteroids/models/source/champion
INITIALIZE_FROM=models/source/champion ./cloud/setup.sh
```

The fork rung is then **measured** -- `train-ppo-survival-v2` binary searches for the lowest
round that policy cannot already clear and starts there. Do not hardcode a stage: an earlier
version of this file suggested `START_STAGE=16`, which dropped a policy five rungs above
where it could promote and left it grinding.

### Caveats worth knowing before you sign up

- **A1 capacity is often unavailable** in busy regions; "Out of host capacity" is the usual
  first experience. Retry, or pick a quieter home region -- the home region is fixed at
  signup.
- A **credit card is required for identity verification**. Always Free shapes do not charge
  it; be careful not to provision anything outside them.
- Oracle reclaims **idle** Always Free compute. A box running PPO continuously is not idle.
- The instance is `aarch64`, the same architecture as Apple Silicon, and checkpoints are
  plain torch state dicts -- they move between the two unchanged.

## Kaggle and Colab

Both work, both are interrupted. Keep the run directory on persistent storage (a Kaggle
dataset, or Drive on Colab) and let the 250-episode checkpoints bound the loss. Colab's
notebook lives at `notebooks/asteroids_colab.ipynb`.

Resume a chunked run by pointing `RESUME` at the newest complete `checkpoint_*`, keeping the
same curriculum and output directory:

```bash
OUTPUT=/content/drive/MyDrive/Asteroids/models/ppo-survival-v2 \
RESUME=/content/drive/MyDrive/Asteroids/models/ppo-survival-v2/checkpoint_XXXXXX \
CURRICULUM=configs/rl-survival-v2.toml PPO_DEVICE=cpu ./run.sh train-ppo 2000
```

`RESUME` refuses if the curriculum's task hash has moved, which is what stops a run silently
continuing on a different game. After a curriculum change use `INITIALIZE_FROM` instead.

## Keeping the local Mac training while you are away

Nothing cloud about it, and it is the fastest machine you have:

```bash
caffeinate -is -w $(pgrep -f 'asteroid_survival train-ppo')   # attach to the running trainer
```

`caffeinate -w` holds sleep off until that process exits. Closing the lid still sleeps an
Apple Silicon Mac unless it is on power with an external display attached.

## Compute accounting

Durable metadata records about 21 training wall hours: 12.5 CPU and 8.5 Apple-MPS, with
**zero cloud GPU hours**. Do not project mastery from episode counts -- dwell time on hard
rungs and evaluation cost dominate. Recalculate from each checkpoint's measured
`decisions_per_second`.
