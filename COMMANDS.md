# Commands

Run everything from the project directory:

```bash
cd /Users/michaelberman/Asteroids
source .venv/bin/activate
```

`run.sh` already uses `.venv/bin/python` when it exists, so activation is convenient but
not required.

## Current survival and team commands

| Command | Purpose |
|---|---|
| `./run.sh play survival-v2 N` | Play any of the 96 nonlinear-first survival rounds |
| `./run.sh rounds survival-v2` | Print every round's exact difficulty |
| `./run.sh train-ppo-survival-v2 N` | Fork the protected round-16 PPO into v2 |
| `./run.sh test-ppo CHECKPOINT [ROUND]` | Run the untouched 128-seed final panel |
| `./run.sh train-team N` | Train one centralized joint policy for exactly two ships |
| `./run.sh train-mappo-team N` | Train a shared 1–8 ship actor and team critic |
| `PROTECT=1 INITIALIZE_FROM=TEAM_CHECKPOINT ./run.sh train-mappo-team N` | Train protection after survival |
| `SHIPS=8 LEVEL=12 ./run.sh test-team CHECKPOINT` | Evaluate a team checkpoint |
| `SHIPS=8 LEVEL=12 ./run.sh play-team CHECKPOINT` | Watch the shared actor fly every ship |

The default v2 source is `models/ppo-survive-0819-1852/champion` (episode 22,000,
human round 16). Override it with `INITIALIZE_FROM`, `START_STAGE`, and `OUTPUT`. Never use
cross-task `RESUME`: v2 has a new task hash and observation layout.

`train-team` is the current two-ship path. Its action is one 256-way joint choice (all
16×16 action pairs), with at most one ship firing per decision. It starts with four finite
waves that always require shooting, adds ship collisions and then friendly fire, crosses
three 30-second survival warmups, and finally trains all 96 real survival rounds. Training
episodes are 80% current stage, 10% foundational wave rehearsal, and 10% uniformly sampled
older stages. Held-out promotion uses only this learned policy; greedy and pilot have no
episode weight. Set `TEAM_STAGE=N` to select a one-based stage for `test-team` or `play-team`.

Legacy MAPPO levels 1–12 map to solo-v2 rounds 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, and 96.
`MAX_SHIPS` defaults to 8; half the episodes use that frontier count and half rehearse
smaller teams. The feed-forward solo model is usually CPU/environment-bound, so benchmark
before reserving a GPU.

## Modes

Five commands — `play`, `showdown`, `watch`, `compare`, `versus` — share one mode
vocabulary, so there is a single thing to learn:

| Mode | Takes a round? | Meaning |
|---|---|---|
| `arcade` | no | clear-the-wave arcade play (the default) |
| `endless` | no | one run whose difficulty ramps forever |
| `round N` | yes, 1–48 | nonlinear curriculum round `N` |
| `survival N` | yes, 1–96 | endless survival ladder round `N` |

Arguments are positional and always in this order:

```
./run.sh <command> [mode] [round] [runs]
```

`round` is only accepted by `round` and `survival`. A bare number is always `runs`, so
`./run.sh watch 20` means twenty runs of the default mode, while
`./run.sh watch survival 8 20` means twenty runs of survival round 8.

## Play and watch

| Command | Parameters | Defaults | What it does |
|---|---|---|---|
| `play [mode] [round]` | mode, round | `arcade` | You alone |
| `showdown [mode] [round]` | mode, round | `arcade` | You + greedy + your newest model, one shared arena |
| `patterns [name]` | a pattern name | all of them | Watch trajectory shapes with motion trails |
| `preview [dir] [round] [seed]` | run directory, round, seed | live run, its stage, random seed | Watch a run's champion play |
| `watch [mode] [round] [runs]` | mode, round, runs | `20` runs | Agents only, scored |
| `compare [mode] [round] [runs]` | mode, round, runs | `5` runs | You and the agents on identical seeds |
| `versus [mode] [round] [runs]` | mode, round, runs | `10` runs | Several models, each alone, ranked |

```bash
./run.sh play                    # arcade, alone
./run.sh play survival 12        # endless-ladder round 12
./run.sh play round 48           # the hardest nonlinear round
./run.sh showdown survival 12    # you + greedy + newest model
./run.sh patterns serpentine     # one trajectory shape, with trails
./run.sh preview                 # the run currently training
./run.sh preview models/my-run 30   # that run's champion on round 30
./run.sh watch survival 8 20     # twenty scored runs on ladder round 8
```

Controls: A/D or Left/Right rotate, W or Up thrusts, Space fires, P pauses, R restarts,
Escape quits. In `preview`, N advances to a new seed and R repeats the current one.

## Baselines

Two scripted opponents, neither of which learns, both deterministic:

| Baseline | What it does |
|---|---|
| `greedy` | Rotates toward the nearest rock's *current* position and fires. Never thrusts. |
| `pilot` | Leads its shots by solving the intercept, and thrusts away from rocks whose closest approach is inside its own radius. |

Greedy stops discriminating between policies exactly where the ladder gets interesting,
because it cannot dodge and cannot hit a crossing target. The pilot is the harder yardstick.
Measured over 20 seeds per round on survival-v2:

| Round | greedy clears | pilot clears |
|---:|---:|---:|
| 1-15 | 100% | 100% |
| 20 | 80% | **100%** |
| 25 | 35% | **70%** |
| 30 | 10% | **40%** |
| 40 | 0% | **15%** |

Both appear automatically in `watch`, `compare`, and `versus`. Drop either with `GREEDY=0`
or `PILOT=0`. `pilot` is also a lineup name, so `./run.sh play` and `showdown` can use it.

**Shared arena versus separate games.** `showdown` puts everyone in one arena, so one
player's kills help the others — it is a demo, not a measurement. `compare`, `watch`, and
`versus` give every contender its own run of the same seeds, which is what to trust.

Model selection is automatic: the **newest run**, then the best held-out checkpoint inside
it. That is deliberately not "best score across every run" — an abandoned run that reached a
later curriculum stage would outrank the run you are actually training.

```bash
CHECKPOINT=models/my-run/champion ./run.sh showdown round 48   # pin one exactly
ANY_RUN=1 ./run.sh showdown survival 12                        # search every run by score
SEED=10000 ./run.sh play survival 12                           # repeatable layout
ALGO=muzero ./run.sh showdown                                  # MuZero instead of PPO
```

## Comparing models

`versus` scores any number of contenders. Each plays **alone**, on its own run of the same
seeds, so nobody clears anyone else's asteroids. Results are ranked by mean survival and
written to `metrics/versus.json`, which also keeps every episode's full metrics.

```bash
./run.sh versus survival 12 10                     # newest model + greedy, ten runs each
MODELS=a/champion,b/champion ./run.sh versus round 41 8
HUMAN=1 ./run.sh versus endless 5                  # play the same seeds yourself
GREEDY=0 MODELS=a,b ./run.sh versus round 48 20    # models only, no baseline
```

| Variable | Default | Effect |
|---|---|---|
| `MODELS` | newest run's champion | Comma-separated checkpoints to score |
| `HUMAN` | `0` | `1` adds you to the lineup |
| `GREEDY` | `1` | `0` drops the greedy baseline |
| `OUTPUT` | `metrics/versus.json` | Where the report is written |

Contenders are labelled from their run directory with the launch timestamp stripped, so
`models/ppo-nonlinear-v2-0818-1152/champion` shows as `nonlinear-v2`. The timestamp comes
back only if two runs would otherwise collide.

## Survival v2 rounds (`rl-survival-v2.toml`)

The curriculum the current training runs use. Ninety-six rounds, **thirty seconds each**.
Not the same as the endless ladder in the next section: v2 introduces sine at round 3 and
all eleven patterns at round 23, where the endless ladder flies straight until round 38.

```bash
./run.sh rounds survival-v2      # every round's exact difficulty, from the config itself
./run.sh play survival-v2 23     # play round 23 yourself
./run.sh watch survival-v2 23    # agents only, fixed seeds, scored
./run.sh showdown survival-v2 23 # you + greedy + newest model, one shared arena
./run.sh compare survival-v2 23  # you and the agents on identical seeds, scored
./run.sh preview models/oracle-lowent 23   # a specific run's champion on round 23
```

`rounds` reads the same config the trainer reads, so it can never drift from what is
actually being trained. Everything else takes `survival-v2 N` exactly like any other mode.

| Rounds | Movement pool | Straight | Spawn size |
|---:|---|---:|---|
| 1–2 | linear foundation | 100% | small |
| 3–6 | sine | 50% | small |
| 7–10 | sine, arc, S-curve | 25% | small |
| 11–16 | smooth three | 25% | small → medium over three steps |
| 17–22 | smooth three + zigzag, sawtooth, lane-change | 14.3% | medium |
| 23–25 | **all eleven patterns** | 8.3% | 75% medium / 25% large |
| 26–28 | all eleven patterns | 8.3% | 50% medium / 50% large |
| 29–52 | all eleven patterns | 8.3% | large |
| 53–82 | all eleven patterns | 8.3% | 75% large / 25% medium |
| 83–96 | all eleven patterns | 8.3% | random mixed size |

Every numeric knob — speed, amplitude, wavelength, spawn interval, spread, initial count —
ramps linearly from round 1 to round 96, so no round is ever easier than the one before it.
What jumps is **composition**: the pattern pool changes only at rounds 3, 7, 17, and 23, and
the size mix only at 11, 13, 15, 23, 26, 29, 53, and 83. Round 23 is the largest single step
in the whole curriculum — the only one that changes the pattern pool (6 → 11) and the size
mix at the same time. Worth watching before and after:

```bash
./run.sh play survival-v2 22     # six patterns, all medium
./run.sh play survival-v2 23     # eleven patterns, large rocks appear
```

## The endless survival ladder and its tiers

Ninety-six rounds, **thirty seconds each**. A round is cleared by staying alive to the
limit — the limit only bounds episode cost, the objective is to survive as long as possible.

| Rounds | Config | Size | Motion |
|---:|---|---|---|
| 1–10 | `rl-survival-small.toml` | small | straight lines |
| 11–16 | `rl-survival-bridge.toml` | small, 1 medium in 4 | straight lines |
| 17–26 | `rl-survival-medium.toml` | medium (splits into 2) | straight lines |
| 27–38 | `rl-survival-large.toml` | large (splits into 6) | straight lines |
| 39–52 | `rl-survival-curves.toml` | large | 3 curves + straight, equally likely |
| 53–82 | `rl-survival-full.toml` | large | all twelve trajectories, equally likely |
| 83–96 | `rl-endless.toml` | **mixed** | all twelve trajectories, equally likely |

Chained by `extends`, so loading `rl-endless.toml` gives the whole ladder. One new thing per
tier, the way the arcade curriculum does it: size first (a small rock does not split, so an
early round never becomes a crowd), then curvature, then the full pattern set, then mixed
sizes. Every join is continuous — no tier starts easier than the one before it ended. The
bridge tier exists because small → all-medium was the ladder's steepest single step: it
mixes one medium into every four rocks first.

"All twelve" means the eleven curved patterns plus a straight line, each at 1/12 — straight
is one trajectory among equals, not a privileged share.

`rl-endless-coop.toml` adds rounds 97–126: the full-pattern tier replayed from round 53's
difficulty with **two ships both running the same policy**, where bumping kills both and a
shot landing on a teammate kills it. Deliberately not the hardest tier — when a new axis
appears, the others reset to a level already mastered.

```bash
./run.sh train-ppo-endless 40000   # the solo ladder
./run.sh train-ppo-coop 15000      # the two-ship tier
./run.sh play survival 1           # play any round yourself
```

Two calibration facts worth knowing, both measured over 24 seeds per round. **A do-nothing
policy clears at most 25% of a round** (round 1; 12% by round 10, 0% from round 52 on) — at
thirty seconds the field has half as long to fill, so idling is much harder to defeat than
at sixty, and the guard is `initial_asteroids` (eight rocks already in flight at round 1)
rather than a tight spawn clock. **The greedy baseline** clears 100% through round 27, first
falls under the 90% gate around round 38, and reaches zero by round 60 — so most of the
ladder is above where a hand-written policy stops.

## Trajectory patterns

Eleven shapes, listed in `PATTERN_NAMES`. Rounds 29+ of the nonlinear curriculum sample all
of them uniformly, per asteroid. The *endless* ladder introduces them gradually: straight
lines to round 38, three curves through round 52, then the full set from round 53 on.
**Survival v2 is much earlier** — sine at round 3, six patterns by round 17, all eleven at
round 23.

```bash
./run.sh patterns              # all of them at once, each labelled, with motion trails
./run.sh patterns brownian     # one at a time
```

`sine` `zigzag` `sawtooth` `s_curve` `lane_change` `serpentine` `arc` `corkscrew`
`figure_eight` `spiral` `brownian`

The ship is invulnerable in this view and asteroids never split, so the paths stay legible.
`serpentine` is the hardest of the set and `brownian` the most aimless; the README has the
measured per-pattern difficulty table.

## Training

Every training command takes one parameter: the **episode budget**. On a resume that means
*additional* episodes, not a target total.

| Command | Parameter | Default | What it does |
|---|---|---|---|
| `train [N]` | episodes | `10000` | MuZero on the mastery curriculum |
| `train-fast [N]` | episodes | `10000` | MuZero with 24 searches / 16 updates instead of 50 / 32 |
| `train-ppo [N]` | episodes | `10000` | Feed-forward PPO |
| `train-lstm [N]` | episodes | `10000` | Recurrent LSTM-PPO |
| `train-ppo-nonlinear [N]` | episodes | `15000` | Transfers the best PPO into rounds 29+ |
| `train-ppo-endless [N]` | episodes | `15000` | Transfers the best PPO into the survival ladder |
| `train-ppo-survival-v2 [N]` | episodes | `5000` | Safely widens/forks the protected solo PPO into v2 |
| `train-ppo-coop [N]` | episodes | `15000` | Transfers a solo model into the two-ship tier |
| `train-mappo-team [N]` | episodes | `1000` | Shared decentralized actor and centralized team critic |
| `ppo-screen [N]` | episodes | `10000` | PPO then LSTM-PPO sequentially on one seed |
| `continue DIR [N]` | run directory, episodes | newest run, `10000` | Continue a run with replay, stage, and logs intact |
| `finish DIR [N]` | run directory, episodes | newest run, `100000` | Continue until every stage is mastered |

```bash
./run.sh train-ppo 10000
./run.sh train-ppo-endless 15000
./run.sh continue models/my-run 5000
```

Only one trainer runs at a time; a second is refused so timings stay comparable. Set
`ALLOW_CONCURRENT=1` if you genuinely want the contention.

### Variables that shape a training run

| Variable | Default | Effect |
|---|---|---|
| `OUTPUT` | `models/<label>-<MMDD-HHMM>` | Where the run is written |
| `CURRICULUM` | `configs/rl-curriculum.toml` | Which curriculum to train on |
| `RESUME` | — | Continue a checkpoint with its optimizer, stage, and logs |
| `INITIALIZE_FROM` | — | Take the weights only, resetting optimizer and task state |
| `START_STAGE` | — | One-based curriculum stage to begin at |
| `STOP_WHEN_MASTERED` | `0` | `1` stops as soon as the final stage passes its gate |
| `LEARNING_RATE` | `0.001` MuZero / checkpoint's own for PPO | Adam rate; on a PPO resume this overrides only the rate |
| `SEED` | `0` | Training seed |
| `PARALLEL_ENVS` | `16` MuZero / `8` PPO | Environments stepped together |
| `EVAL_EVERY` | `250` | Episodes between held-out evaluations |
| `FOLLOW_EVERY` | `15` | Seconds between `follow` polls |
| `PPO_DEVICE` | `auto` | `cpu` or `mps` to override. **Use `cpu`** — `auto` picks MPS, which is ~2.2× slower here |
| `SIMULATIONS` | `50` | MuZero tree searches per decision |
| `UPDATES_PER_EPISODE` | `32` | MuZero gradient batches per episode |
| `HISTORY_FRAMES` | `8` | Past asteroid positions the policy sees |
| `ENCODER` | MLP | `set` swaps the flat MLP for a permutation-invariant set encoder |
| `CHECKPOINT_EVERY` | `250` | Episodes between checkpoint writes |
| `LOG_EVERY` | `250` | Episodes between log lines |
| `RESUME_LEARNING_RATE` | — | Reset a resumed optimizer to this rate |

`RESUME` and `INITIALIZE_FROM` are mutually exclusive, and the difference matters. `RESUME`
requires the task to be unchanged — it refuses if the curriculum's task hash has moved,
which is what stops a run silently continuing on a different game. `INITIALIZE_FROM` keeps
the policy weights and deliberately discards everything task-specific, which is the right
tool after a curriculum or trajectory change.

```bash
# Continue the same run untouched.
RESUME=models/my-run/checkpoint_015000 OUTPUT=models/my-run ./run.sh train-ppo 10000

# Carry the weights into a changed task, starting part-way up the curriculum.
INITIALIZE_FROM=models/my-run/champion START_STAGE=35 \
  CURRICULUM=configs/rl-nonlinear.toml OUTPUT=models/next ./run.sh train-ppo 25000
```

## Inspecting a run

| Command | Parameters | Default | What it does |
|---|---|---|---|
| `status [dir]` | run directory | the live run, else newest | Progress, champion, recent held-out evaluations |
| `follow [dir]` | run directory | the live run, else newest | `status` once, then every new evaluation as it lands |
| `graph [dir] [both\|completion\|survival]` | run directory, view | newest, `both` | Prints held-out progress as a terminal line chart; saves nothing |
| `baseline [N]` | episodes | `60` | Scores the greedy controller |
| `test` | — | — | Runs the test suite |
| `test-team CHECKPOINT` | checkpoint | — | Scores a centralized or legacy team policy |

`follow` is `status` that keeps going. It prints the usual block, then one line per
held-out evaluation as the trainer writes it — round, completion against the promotion gate,
mean survival, accuracy, promotion streak — flagging `PROMOTED`, `FELL BACK`, and
`new champion` as they happen. It stops on Ctrl-C, or by itself once the trainer exits and
there is nothing left to arrive. `FOLLOW_EVERY` (default 15s) sets the poll interval;
evaluations are minutes apart, so there is no reason to poll hard.

`graph` uses Unicode Braille as a portable 2x4-pixel terminal canvas and expands to the full
terminal width. Its vertical range automatically zooms to the observed minimum and maximum;
bright dots mark actual held-out evaluations, while the Braille strokes interpolate between
them. `P` marks the evaluation that promoted the policy. Set `GRAPH_HEIGHT=30` for a taller
chart.

```bash
./run.sh follow                       # the run currently training
./run.sh follow models/my-run         # a specific run
FOLLOW_EVERY=60 ./run.sh follow       # check in once a minute
```

`status` and `preview` with no argument prefer whichever run has a **live trainer**, falling
back to the newest directory. Right after a restart the stopped run would otherwise be
reported as "not running", which reads as a failure rather than an intentional stop.
`status` also lists the other runs so they are easy to reach.

## Variables that apply anywhere

| Variable | Default | Effect |
|---|---|---|
| `SEED` | `7` play / `11` showdown / `3` patterns / random preview | Starting seed |
| `CHECKPOINT` | auto | Use one exact model rather than the auto-selected one |
| `ANY_RUN` | — | `1` auto-selects across every run by score, not just the newest |
| `ALGO` | `ppo` | `muzero` to put MuZero in a showdown |
| `CONFIG` | `configs/rl-arcade.toml` | Raw config for `compare`/`watch`/`baseline` when no mode is given |
| `PY` | `.venv/bin/python` | Interpreter, overridable so the dispatch table can be tested |
| `ALLOW_CONCURRENT` | `0` | `1` bypasses the single-trainer guard |

---

This file is the command reference. For what the commands are *doing* — the curriculum
stages and their gates, the reward equation, the endless ladder's difficulty table, and the
measured per-pattern difficulty ranking — see `README.md`.
