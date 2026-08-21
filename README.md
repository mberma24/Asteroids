# Asteroid Survival

A deterministic Asteroids-style game with arcade waves, local controllers, a compact
JAX/Flax MuZero-style learner, and feed-forward/recurrent PPO research baselines.

## End-state architecture

The target is one shared learned policy controlling **1–8 ships** against boundary-spawned
random nonlinear asteroids. Each ship acts from its local observation; a centralized team
critic exists only during training. Object protection is a later, goal-conditioned phase
that retains pure-survival practice.

The path is staged: finish solo motion prediction with `survival-v2`, train genuine shared
multi-agent PPO, then unlock protection. The old frozen-companion co-op task remains
reproducible, but `train-mappo-team` is the cooperative trainer: every living ship contributes
experience and an individual death does not end the team episode.

```bash
./run.sh train-ppo-survival-v2 5000  # safe fork of the protected round-16 champion
./run.sh play survival-v2 29         # first entirely nonlinear round
./run.sh rounds survival-v2          # exact values for all 96 rounds
./run.sh train-mappo-team 1000       # shared actor, centralized critic, teams 1-8
PROTECT=1 INITIALIZE_FROM=models/TEAM/champion ./run.sh train-mappo-team 1000
SHIPS=8 LEVEL=12 ./run.sh test-team models/MY-RUN/checkpoint_001000
SHIPS=8 LEVEL=12 ./run.sh play-team models/MY-RUN/checkpoint_001000
```

V2 introduces sine motion at round 3, every nonlinear family by round 23, and all eleven
nonlinear patterns with zero straight-line probability from round 29 onward. Observation
layout v5 appends sixteen difficulty/threat features without moving the legacy 1,235 inputs,
so the v4 feed-forward policy can be widened with zero-filled input columns.

See [CURRICULUM.md](CURRICULUM.md) for rounds/rewards and
[CLOUD_TRAINING.md](CLOUD_TRAINING.md) for interruption-safe online training.

## Run it

```bash
cd /Users/michaelberman/Asteroids
source .venv/bin/activate
./run.sh play                 # arcade, alone
./run.sh play endless         # difficulty ramps forever
./run.sh play survival 12     # endless-ladder round 12
./run.sh play round 48        # nonlinear curriculum round 48
./run.sh showdown survival 12 # you + greedy + best model, one arena
./run.sh watch survival 8 20  # agents only, twenty scored seeds
./run.sh versus round 41 10   # several models, each alone, ranked
./run.sh preview              # watch the best held-out checkpoint
./run.sh train-ppo 10000      # feed-forward PPO
./run.sh train-ppo-endless 15000  # transfer FF-PPO into the survival ladder
./run.sh status               # latest held-out results
./run.sh test
```

Every play-style command (`play`, `showdown`, `watch`, `compare`) takes the same four
modes: `arcade`, `endless`, `round N` (1–48), and `survival N` (1–96). They are resolved in
one place, `modes.py`, so a checkpoint that fits the observation layout can play any of
them and there are no hand-maintained showdown configs to drift out of step.

Install from scratch with Python 3.11+ using `pip install -e '.[dev,rl,ppo]'`. The `rl`
and `ppo` extras are independent if only one learner is needed.

Controls are A/D or Left/Right to rotate, W or Up to thrust, and Space to fire. Press P to
pause, R to restart, and Escape to quit. The renderer keeps deterministic 900×900 physics
and scales it to the largest square that fits the display.

Play any foundation or extended round yourself with its exact asteroid composition and
motion parameters, straight from the curriculum rather than from a copy of it:

```bash
./run.sh play round 1
./run.sh play round 48
SEED=10000 ./run.sh play round 48
```

Rounds 1–48 come from `configs/rl-nonlinear.toml`, which inherits the original first 28.
Clearing the configured wave ends the round. `R` restarts the same round and seed.

## Endless mode

```bash
./run.sh endless              # a new random seed every launch
SEED=10000 ./run.sh endless   # the same run twice, for a fair comparison
```

Endless mode is the open-ended survival test: `configs/endless.toml`. There are no waves and
no round to clear, so a run ends only when the ship is hit, and the single number it
produces is how long that took.

Every asteroid samples one of all eleven nonlinear patterns. Difficulty arrives in
twenty-second tiers: it holds completely still inside a tier, then every knob jumps at the
boundary. Thirty seconds in is already demanding and one minute in is punishing — a pace
set by what an episode costs to train on, and one that keeps the observation distribution
stationary within a tier instead of drifting under a learner on every frame.

| Tier | From | Speed | Spawn | Cap | Amplitude | Period | Spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0s | 39–60 | 4.31s | 8 | ≤50 | 3.1–4.77s | 130° |
| 2 | 20s | 62–96 | 2.93s | 15 | ≤90 | 2.5–3.9s | 90° |
| 3 | 40s | 84–132 | 1.54s | 22 | ≤130 | 1.9–3.03s | 50° |
| 4 | 60s | 101–159 | 0.80s | 26 | ≤150 | 1.6–2.6s | 30° |
| 6 | 100s | 123–194 | 0.66s | 26 | ≤150 | 1.6–2.6s | 30° |
| 10 | 180s | 167–264 | 0.48s | 26 | ≤150 | 1.6–2.6s | 30° |

Spread is how far a spawn's heading is scattered away from the arena centre, so a wide
opening spread means most rocks drift harmlessly past and a narrow late spread means they
come at you. Speed, spawn interval, active cap, amplitude, period, and spread all
interpolate over the four tiers of the first minute. From tier 4 on,
`endless_pressure_per_minute` keeps multiplying speed and dividing the spawn interval on
the same twenty-second cadence, forever, so no policy survives indefinitely and any two
runs can be ranked. Amplitude, cap, and spread stop at their tier-4 values — those are
bounded by the arena size and by the fixed observation layout, not by how hard the run
should get.

`ramp_seconds` is 50 rather than 60 because difficulty is evaluated at the tier boundary
below the clock, so the ramp is sampled at 0s, 20s, and 40s — progress 0, 0.4, 0.8 — and
tier 4 clamps to the full target. That spaces the four tiers evenly across the first minute.

The orange HUD line under the controls shows the tier and the difficulty currently in
force, so the ramp is visible while playing rather than inferred from the clock.

Measured over twenty seeds with no learned model involved:

| Controller | Median survival | Dies around |
|---|---:|---|
| do nothing | 25.6s | tier 2 |
| random | 18.1s | tier 1 |
| greedy baseline | 34.3s | tier 2 |

Note the compression: the greedy baseline now outlives doing nothing by only 8.7s, where
an earlier and gentler version of this ramp separated them by 33.4s. The schedule through
the first thirty seconds is fixed by design; if that gap proves too thin to rank policies
during training, the knob to reach for is `endless_pressure_per_minute`, which governs
everything after tier 4 and is not constrained by the early schedule.

## Endless survival ladder (training)

`configs/rl-endless.toml` turns endless mode into a mastery curriculum. Each round is a
**fixed** difficulty, and clearing it means staying alive for thirty seconds rather than
clearing a wave. Promotion, retention, rehearsal, and champion tracking are the existing
machinery, unchanged — `completed_stage` simply becomes "survived to the decision limit".

```bash
./run.sh play-endless 1        # play round 1 yourself
./run.sh play-endless 12       # ...or any round up to 96
./run.sh train-ppo-endless 15000   # transfer the best nonlinear PPO into the ladder
```

This resolves the tension the in-episode ramp had. Episode cost is bounded by the survival
target — 450 decisions at every rung, no matter how strong the policy gets — while
difficulty stays constant inside a round, so the observation distribution is stationary.
Policies are ranked by *which round they reach*, which is unbounded and monotonic, instead
of by seconds survived in one run.

| Round | Starts with | Speed | Amplitude | Period | Spawn | Spread |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 26–38 | — | — | 2.00s | 170° |
| 27 | 9 | 34–50 | — | — | 1.60s | 168° |
| 39 | 9 | 40–60 | 8–16 | 3.4–5.2s | 1.40s | 160° |
| 53 | 10 | 46–70 | 12–34 | 3.4–5.2s | 1.30s | 152° |
| 82 | 16 | 100–158 | 12–130 | 1.8–2.9s | 0.90s | 45° |
| 96 | 20 | 126–196 | 12–170 | 1.45–2.35s | 0.75s | 24° |

Amplitude and period are blank until round 39 because everything below it flies straight.

A round opens with its field **already occupied**. Interval spawning otherwise starts on an
empty arena, so the first seconds would be free time that says nothing about the round.

Asteroids wrap rather than leaving the arena, so `spawn_interval` is the real difficulty
knob: it sets how fast the field fills. `active_cap` is deliberately **not** a per-round
knob — the observation layout is sized from it and has to stay identical across rounds, the
same reason the wave curriculum pins it at 26.

Opening asteroids are placed anywhere in the arena but never near the ship, and that
clearance is mostly **time-based** rather than a fixed distance: `spawn_safe_radius` sets a
60px floor and `spawn_safe_seconds` adds 1.8 seconds of the asteroid's own travel on top. A
flat radius would be a shrinking reaction window as the ladder speeds up — 180px is over
four seconds of warning at round 1 and under a second and a half at the top, which put a
third of late-round deaths inside the opening three seconds. Clearance is checked against
each asteroid's true position at age zero (a pattern's lateral offset can be as large as its
amplitude) and across the arena wrap.

From round 39 on, asteroids sample the curved patterns **and a straight line with exactly
the same probability as any one curve** — 1/4 in the curves tier (3 patterns + straight),
1/12 in the full tier (11 + straight). A field of nothing but curves is its own narrow
distribution — a policy can learn to always expect a turn — so plain linear motion stays in
the mix, but it is not privileged either.

### The ladder's seven tiers

These tiers are **`configs/rl-endless.toml`**, not the survival-v2 curriculum. Both are
96 rounds and both are survival ladders, so check which one a run was launched with
before reading a round number off this table. Everything here flies straight until
round 39; survival-v2 introduces sine at round 3 and all eleven patterns at round 23.
For that schedule see `CURRICULUM.md`.

Ninety-six rounds, **thirty seconds each**. The limit only bounds episode cost — the
objective is to survive as long as possible, and a shorter round simply buys twice as many
rungs.

| Rounds | Size | Motion | What it introduces |
|---:|---|---|---|
| 1–10 | small | straight | interception, with rocks that never split |
| 11–16 | small, 1 medium in 4 | straight | splitting, one rock at a time |
| 17–26 | medium | straight | splitting everywhere: each rock becomes two |
| 27–38 | large | straight | the full split chain, seven hits per rock |
| 39–52 | large | 3 curves + straight, equally likely | curvature, nothing erratic yet |
| 53–82 | large | all twelve, equally likely | the long tier; every knob tightens |
| 83–96 | **mixed** | all twelve, equally likely | sizes rolled per spawn |

The bridge tier (11–16) was added after measurement: small → all-medium was the steepest
single step in the ladder, and mixing one medium into every four rocks turns it into two.

One new thing per tier, the way the arcade curriculum does it. Size comes first because a
small rock does not split — shoot it and it is gone — so an early round never becomes a
crowd. Every join is continuous: no tier begins easier than the one before it ended.

Mixed sizes is the honest top axis rather than a smalls-only tier, which would be *easier*
than the tier before it: a small is the least dangerous size per rock and costs one bullet
instead of seven.

**Stationary rounds were considered and rejected.** In a survival round, motionless asteroids
never reach the ship and spawns are already guarded away from it, so doing literally nothing
clears such a round **24 times out of 24 with zero kills** — measured. Every round is
therefore survivable only by acting.

Two calibration facts, both measured over 24 seeds per round:

- **Idling clears at most 25%** — that is round 1; it is 12% by round 10 and 0% from round
  52 on. At thirty seconds this is much harder to guarantee than at sixty, because the field
  has half as long to fill — the lever is `initial_asteroids` (eight rocks in flight at
  round 1), *not*
  `spawn_spread`. Narrowing spread aims rocks at the arena centre rather than at the ship,
  so it is dodged by simply not being in the middle; spread keeps a monotone 170° → 24°
  envelope across the whole ladder and is never spent on the foundation.
- **Greedy** clears 100% through round 27, slips under the **90%** promotion gate around
  round 38 as curvature arrives, and reaches zero by round 60 — so roughly the top 60% of
  the ladder is above where a hand-written policy stops.

The gate is 90%, not the 80% the wave curriculum uses, because a thirty-second round is
markedly easier to survive: measured on greedy, the same round reads 9–19 points higher at
thirty seconds than at sixty where the gate bites, and up to 47 points higher on the hardest
rounds. Retention stays at 75%, so the ladder promotes high and falls back only on real decay.

### Tier two: two ships, one policy

`configs/rl-endless-coop.toml` adds rounds 97–126: the full-pattern tier replayed from round
53's difficulty envelope with **two ships both flying the same policy**. Bumping into each
other kills both; a shot that lands on a teammate kills it. Rounds 1–96 are inherited
unchanged and stay in the retention set, so solo skill is not traded away for co-operation. A third tier is another file extending this one with
`ships = 3`.

```bash
./run.sh train-ppo-coop 15000
```

Three things make the tiers one continuous curriculum rather than separate games:

- The observation reserves **teammate slots**, sized for the busiest round and appended
  last. A policy cannot avoid ships it cannot see, and appending leaves every earlier input
  weight pointing at the same feature.
- Companions load a **snapshot of the learner**, refreshed at each evaluation. Environments
  run in separate processes, so the live network cannot be shared; a companion is a slightly
  stale copy of the learner rather than a scripted bystander.
- A solo model seeds the tier by **widening its policy**: learned weights keep their columns
  and the new teammate inputs start at zero, so the transferred policy behaves identically
  until it learns to use them.

Reward is survival-first. `active_time_penalty` and `timeout_penalty` are both zero,
because in a survival round they are exactly backwards: one charges for staying alive, the
other penalises reaching the decision limit, which is how a round is cleared.

| Term | Value |
|---|---:|
| survival | +0.10 per decision (+45.0 over a full round) |
| round cleared | +10.0 |
| asteroid hit | +0.15 / +0.30 / +0.60 by size |
| miss | −0.02 |
| death | −5.0 |

Staying alive dominates deliberately: a full round of survival pays +45, while destroying
ten large rocks pays +6. Hits are worth something because a field left uncleared eventually
kills you, but they are never worth trading survival for. Ordering by size is the way round
— a large rock pays more than the two mediums it becomes — so breaking rocks up is not
itself the reward.

Greedy baseline over 24 seeds per round, which anchors the ladder:

| Round | 1 | 27 | 33 | 38 | 39 | 52 | 53 | 60+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Completion | 100% | 100% | 96% | 92% | 71% | 46% | 38% | 0% |

Greedy stalls at the 90% promotion gate around **round 38**, leaving nearly sixty rounds of
headroom above it.

## Arcade game rules

The main play configs now use movement and inertia. Wave 1 contains four large rocks; later
waves add two, capped at eleven. No wave begins until every prior fragment is gone. Fresh
rocks travel in uniformly random directions at 40–70 px/s instead of aiming at the ship;
children are progressively faster. The active cap is 26, with a two-second opening grace
and a 1.5-second inter-wave pause.

| Arcade wave | Starting large rocks |
|---:|---:|
| 1 | 4 |
| 2 | 6 |
| 3 | 8 |
| 4 | 10 |
| 5 and later | 11 |

Large rocks split into two medium rocks and each medium splits into two small rocks. The
26-rock active cap can suppress a split when the arena is already full.

Older stationary, continuous-spawn, ramp, and protection configs remain for legacy
experiments. `heading_mode` is `aimed`, `spread`, or `random`. `motion_mode` is `linear`,
`specific`, or `pool`; pool mode can mix linear paths with `linear_probability`.

## Mastery curriculum

`./run.sh train`, `train-ppo`, and `train-lstm` use
`configs/rl-curriculum.toml`. Every stage contains one wave. `S`, `M`, and `L` below are the
rocks present at the start; “hits” includes every child created by splitting (`S=1`, `M=3`,
`L=7`). A turret can rotate and fire but has zero thrust acceleration.

| Stage | Lesson | Start | Hits | Motion, speed | Ship | Limit / no-hit |
|---:|---|---|---:|---|---|---:|
| 1 | stationary-1-small | S | 1 | stationary | turret | 30s / 20s |
| 2 | stationary-2-small | S+S | 2 | stationary | turret | 30s / 20s |
| 3 | stationary-1-medium | M | 3 | stationary | turret | 30s / 20s |
| 4 | stationary-1-medium-1-small | M+S | 4 | stationary | turret | 30s / 20s |
| 5 | stationary-2-medium | M+M | 6 | stationary | turret | 28s / 20s |
| 6 | stationary-2-medium-1-small | M+M+S | 7 | stationary | turret | 32s / 20s |
| 7 | stationary-1-large | L | 7 | stationary | turret | 32s / 20s |
| 8 | stationary-1-large-1-small | L+S | 8 | stationary | turret | 36s / 20s |
| 9 | stationary-1-large-1-medium | L+M | 10 | stationary | turret | 42s / 20s |
| 10 | stationary-2-large | L+L | 14 | stationary | turret | 55s / 20s |
| 11 | safe-linear-1-large | L | 7 | linear, 20–30 | invulnerable turret | 25s / 34s |
| 12 | safe-linear-2-large | L+L | 14 | linear, 20–30 | invulnerable turret | 50s / 30s |
| 13 | safe-linear-3-large | L+L+L | 21 | linear, 20–30 | invulnerable turret | 65s / 54s |
| 14 | linear-1-large | L | 7 | linear, 20–30 | turret | 25s / 34s |
| 15 | linear-2-large | L+L | 14 | linear, 20–30 | turret | 50s / 30s |
| 16 | linear-3-large | L+L+L | 21 | linear, 20–30 | turret | 65s / 54s |
| 17 | sine-1-large | L | 7 | sine, 25–35 | turret | 40s / 20s |
| 18 | sine-2-large | L+L | 14 | sine, 25–35 | turret | 65s / 22s |
| 19 | sine-3-large | L+L+L | 21 | sine, 25–35 | turret | 85s / 21s |
| 20 | mobile-sine-1-large | L | 7 | sine, 25–35 | mobile | 40s / 20s |
| 21 | mobile-sine-2-large | L+L | 14 | sine, 25–35 | mobile | 65s / 22s |
| 22 | mobile-sine-3-large | L+L+L | 21 | sine, 25–35 | mobile | 85s / 21s |
| 23 | mobile-arc-1-large | L | 7 | arc, 30–40 | mobile | 45s / 26s |
| 24 | mobile-arc-2-large | L+L | 14 | arc, 30–40 | mobile | 100s / 25s |
| 25 | mobile-arc-3-large | L+L+L | 21 | arc, 30–40 | mobile | 100s / 93s |
| 26 | mobile-s-curve-1-large | L | 7 | S-curve, 30–45 | mobile | 55s / 20s |
| 27 | mobile-s-curve-2-large | L+L | 14 | S-curve, 30–45 | mobile | 90s / 20s |
| 28 | mobile-s-curve-3-large | L+L+L | 21 | S-curve, 30–45 | mobile | 85s / 20s |

Every 250 episodes, the current stage is evaluated on 96 fixed seeds and ten earlier stages
— sampled on rotation, not all of them — on 8 retention seeds each; future stages are not
evaluated. An evaluation passes when the current stage reaches at least 80% completion and
5% mean accuracy while the sampled prior stages retain 75% completion *pooled*, with no
single stage under a 50% floor. Two passing evaluations within the latest four promote the
learner. Training samples the current stage 75% of the time and a random earlier stage 25%.

The survival ladder uses the same machinery with its own numbers: 64 evaluation seeds, 16
retention seeds, and a 90% promotion gate.

### Fragments outrun their parent

Every mode now splits asteroids into debris that moves faster than the rock it came from:
medium at 1.15x and small at 1.35x. The wave curriculum used to leave both at 1.0, so a
policy trained on it had never seen this and died to its own splits in arcade play — it shot
*better* than the greedy baseline there (48.6 kills against 40.4, 2.0 waves against 1.7) and
still died at 41.7s while greedy survived the full sixty. It was mispredicting the pieces.

Dry-spell windows (`no_hit_seconds`) are now derived from measurement rather than written by
hand: the longest gap between kills the greedy baseline actually needs on each stage, over
sixteen seeds, with 60% headroom and a 20-second floor. Several were far too tight even
before the fragment change — stage 13 needed 33 seconds and allowed 17 — so competent play
was being cut off as though it had stalled.

## Curriculum reward

The per-episode reward is the sum of these components:

| Event | Reward |
|---|---:|
| Destroy any large, medium, or small rock | `+1` per hit |
| Clear the wave | `+5` |
| Clear-speed bonus | `+6 × max(0, 1 − clear_seconds / 45)` |
| Wave accuracy bonus | `+2 × hits / shots` |
| Turn toward the nearest target | `+0.15 × reduction in aim error (radians)` |
| Active time | `−0.02 × seconds` |
| Missed or unresolved projectile | `−0.03` each |
| Episode/no-hit timeout | `−8` |
| Ship destroyed | `−5` |

The aim term is signed, so turning away loses the same amount that turning toward earns.
Projectiles still flying at termination are resolved as misses. Because each split child is
a hit, fully clearing one medium pays `+3` in hit rewards and one large pays `+7`, before
clear, speed, accuracy, aim, time, or terminal terms.

## Advanced nonlinear phase

The original 28 stages deliberately isolate basic aiming, splitting, collision survival,
movement, and three gentle curve types. They top out at 30–45 px/s forward speed and
25–50 px lateral amplitude, so they do not become more visibly curved after stage 28.

`configs/rl-nonlinear.toml` extends that foundation with rounds 29–48. It inherits the exact
first 28 definitions without modifying their task manifest. Training starts a new run at
round 29 by transferring the mastered feed-forward policy with a fresh optimizer:

```bash
./run.sh train-ppo-nonlinear 15000
```

The launcher automatically selects the best compatible feed-forward PPO. To pin the source
and output names:

```bash
INITIALIZE_FROM=models/ppo-ff-0817-1739/champion \
OUTPUT=models/ppo-nonlinear-v1 \
./run.sh train-ppo-nonlinear 15000
```

Every rock in every added round independently samples one of all eleven nonlinear patterns:
sine, zigzag, sawtooth, arc, S-curve, lane change, serpentine, corkscrew, figure eight,
spiral, or brownian. Round 29 begins at round 28's exact speed/amplitude/period envelope. Thereafter,
forward min/max speed rises by 1 px/s, amplitude rises by 3/5 px, and oscillation periods
shrink by 0.02/0.04 seconds per round. Starting count adds one large rock every two rounds
until reaching the arcade cap of 11 starting rocks; the last four rounds hold 11 while the
continuous motion increments continue.
`A` is lateral amplitude and `T` is pattern period.

| Round | Start | Speed | A | T | Limit / no-hit |
|---:|---|---:|---:|---:|---:|
| 29 | 3L | 30–45 | 25–50 | 3–4.5s | 90s / 20s |
| 30 | 3L | 31–46 | 28–55 | 2.98–4.46s | 96s / 21.5s |
| 31 | 4L | 32–47 | 31–60 | 2.96–4.42s | 102s / 23s |
| 32 | 4L | 33–48 | 34–65 | 2.94–4.38s | 108s / 24.5s |
| 33 | 5L | 34–49 | 37–70 | 2.92–4.34s | 114s / 26s |
| 34 | 5L | 35–50 | 40–75 | 2.9–4.3s | 120s / 27.5s |
| 35 | 6L | 36–51 | 43–80 | 2.88–4.26s | 126s / 29s |
| 36 | 6L | 37–52 | 46–85 | 2.86–4.22s | 132s / 30.5s |
| 37 | 7L | 38–53 | 49–90 | 2.84–4.18s | 138s / 32s |
| 38 | 7L | 39–54 | 52–95 | 2.82–4.14s | 144s / 33.5s |
| 39 | 8L | 40–55 | 55–100 | 2.8–4.1s | 150s / 35s |
| 40 | 8L | 41–56 | 58–105 | 2.78–4.06s | 156s / 36.5s |
| 41 | 9L | 42–57 | 61–110 | 2.76–4.02s | 162s / 38s |
| 42 | 9L | 43–58 | 64–115 | 2.74–3.98s | 168s / 39.5s |
| 43 | 10L | 44–59 | 67–120 | 2.72–3.94s | 174s / 41s |
| 44 | 10L | 45–60 | 70–125 | 2.7–3.9s | 180s / 42.5s |
| 45 | 11L | 46–61 | 73–130 | 2.68–3.86s | 186s / 44s |
| 46 | 11L | 47–62 | 76–135 | 2.66–3.82s | 192s / 45.5s |
| 47 | 11L | 48–63 | 79–140 | 2.64–3.78s | 198s / 47s |
| 48 | 11L | 49–64 | 82–145 | 2.62–3.74s | 204s / 48.5s |

### The eleven trajectory patterns

From round 29 on, every asteroid independently samples one of eleven patterns uniformly, so
a round mixes all of them. Each displaces the asteroid from the straight line it would otherwise
fly, using a **lateral** component perpendicular to its drift and an **along** component
parallel to it. The along component is what makes real shape variety possible: a
lateral-only offset can only ever produce "constant forward speed plus a wiggle", so
circles, loops, and figure eights are unreachable without it.

| Pattern | Shape | Curves along its path |
|---|---|---|
| sine | smooth, even oscillation | no |
| zigzag | constant lateral speed, instant reversals | no |
| sawtooth | ramps across and snaps back, at irregular intervals | no |
| s_curve | long rounded sweeps that pause at each extreme | no |
| lane_change | near-square steps that dwell in a lane, then snap | no |
| serpentine | erratic full-width sweeps, never repeats — **the hardest** | no |
| arc | one wide, continuous, sweeping turn | yes |
| corkscrew | tight repeated loops | yes |
| figure_eight | a true figure eight that crosses its own path | yes |
| spiral | loops widening from nothing to full amplitude | yes |
| brownian | an aimless wander with no shape and no period | yes |

Every shape stays inside `amplitude`, so a curriculum's amplitude setting remains a real
bound. Peak displacement speeds range from 0.2x to 2.6x a plain sine's, which
`PEAK_SPEED_FACTOR` records so observation normalisation does not saturate on the fastest
ones.

Two are deliberately unpredictable rather than merely fast. `serpentine` is a triangle wave
whose rate is itself modulated at an irrational ratio to its own frequency: full-amplitude
sweeps like a sine, sharp corners like a zigzag, and no period to learn — every swing is a
different length from the last. `sawtooth`'s teeth are stretched and squeezed by a slow
incommensurate wobble, so its snaps never fall on a beat. `brownian` is an aimless drift with no recognisable shape, built by spectral synthesis
rather than by accumulating random steps -- a pattern is a pure function of time, so there
is nowhere to keep a walker's state and it must return an exact velocity as well as a
position. Its component amplitudes fall as 1/f, which is what gives Brownian motion its
character: mostly slow, large, directionless drift with finer detail on top. Component
phases derive from each asteroid's own phase, so no two wander alike.

All three are still pure functions of time, amplitude, frequency, and phase: erratic to
watch, exactly reproducible from a seed. **None of them ever teleports.** `sawtooth` used to:
a true sawtooth snaps from one extreme to the other between consecutive frames, which reads
as a rock vanishing and reappearing across the arena. It is now an asymmetric ramp -- a long
drift and a fast but flyable run back -- and `test_no_pattern_ever_teleports` checks every
pattern for frame-to-frame continuity.

Difficulty per pattern, measured with the greedy baseline on endless round 10 over 24 seeds
(lower completion is harder):

| serpentine | sine | arc | zigzag | lane_change | s_curve | figure_eight | sawtooth | brownian | corkscrew | spiral |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **12%** | 21% | 29% | 29% | 33% | 33% | 38% | 50% | 58% | 62% | 62% |

That ranking was not obvious in advance. Difficulty here comes from large coherent sweeps,
not from agitation: two earlier erratic designs built from high-frequency jitter were the
*easiest* patterns in the set at 67%, because small rapid shakes cover little ground and
oscillating along the direction of travel cancels forward progress.

They were previously far less distinct than the names suggest: `figure_eight` was
`sin(x)cos(x)` laterally, which is exactly a half-amplitude sine at double frequency and
correlated **1.000** with `sine`, and eight pairs sat above 0.96. The worst pair is now
0.51. `test_no_two_patterns_are_near_duplicates` keeps it that way.

The spiral pattern is now amplitude-bounded. Previously its envelope grew forever inside an
episode, which made survival difficulty drift independently of the selected round. With the
cap, round parameters remain the actual source of difficulty.

## Training and saving

Quick end-to-end run:

```bash
python -m asteroid_survival train \
  --curriculum configs/rl-curriculum.toml \
  --output models/curriculum-smoke \
  --episodes 10 --simulations 12 --parallel-envs 2 \
  --history-frames 8 --history-long-frames 8 --history-long-stride 8 \
  --checkpoint-every 10 --eval-every 10
```

Runs contain `training.jsonl`, frozen `evaluation.jsonl`, resumable checkpoints and replay,
and `curriculum_state.json`. Generate `progress.svg` with:

```bash
./run.sh graph models/my-run
```

Every evaluation saves a matching checkpoint and prints a command for watching it. Watch an
exact checkpoint, or let the run directory select its best held-out checkpoint:

```bash
./run.sh preview models/my-run/checkpoint_001000
./run.sh preview models/my-run
```

Training keeps two model roles. The newest `checkpoint_*` is the durable learner snapshot and
the normal resume target. `champion/` is a protected copy of the best eligible held-out model
and is the normal play/preview target. PPO never restores champion weights inside a running
rollout: doing that would train on samples from a different policy. Retention failure changes
lesson focus only. An intentional rollback is a new run created with
`INITIALIZE_FROM=.../champion`. Champion ranking uses survival, full-round clearing, and
accuracy when the curriculum defines all three gates.

Replay is partitioned by curriculum stage. Sixty percent of each batch comes from the current
stage and the remainder is divided among mastered stages; half of replay capacity is reserved
for those earlier lessons. If retention fails repeatedly, training temporarily focuses the
weakest retained stage without resetting weights, optimizer state, promotion streak, or replay.
Only the champion and the newest two resumable checkpoints are retained by default.

For faster experiments, `./run.sh train-fast 10000` uses 24 tree-search simulations and 16
gradient updates per episode instead of 50 and 32. The curriculum also avoids evaluating
unseen stages, which saves substantial time without reducing training quality.

## PPO experiment

PPO uses the exact same game, reward, 16 actions, trajectory history, curriculum sampling,
held-out seeds, promotion gates, and champion policy as MuZero. It removes tree search and
replay, making it the controlled test of whether the learning algorithm—not the task—is the
bottleneck. Feed-forward PPO uses two 256-unit layers. Recurrent PPO adds a 256-unit LSTM and
resets its hidden state at every episode boundary.

```bash
./run.sh train-ppo 10000       # feed-forward ablation
./run.sh train-lstm 10000      # memory-enabled policy
./run.sh ppo-screen 10000      # same seed, sequential (never simultaneous)
```

The launcher refuses to start any trainer while another is running. This prevents resource
contention from corrupting throughput comparisons; `ALLOW_CONCURRENT=1` is an explicit escape
hatch. PPO status reports environment decisions as well as episodes, and its graphs use
environment decisions on the x-axis. PPO checkpoints contain `model.zip` and no replay file.

**Run PPO on the CPU.** `PPO_DEVICE` defaults to `auto`, which picks MPS, and MPS is
*slower* here: benchmarked at the real observation size, one learn cycle takes 0.78s on CPU
against 4.81s on MPS, and end to end the run goes from ~470 to ~1030 decisions/s. Launch
with `PPO_DEVICE=cpu` until the default changes.

### Optional set encoder

`ENCODER=set` replaces the flat MLP over the 1,235-float vector with a permutation-invariant
encoder (`src/asteroid_survival/rl/networks.py`): each asteroid, projectile, and teammate
slot is embedded independently, then pooled by masked mean and max. Slot order then carries
no information, so the policy cannot overfit to which slot a rock happens to land in, and
the encoder is much smaller than the flat first layer. It is an experiment rather than the
default — a from-scratch head-to-head against the MLP is what decides it.

Resume feed-forward PPO in the same output directory. The episode count is additional:

```bash
OUTPUT=models/my-run \
RESUME=models/my-run/checkpoint_005000 \
./run.sh train-ppo 5000
```

Use `./run.sh train-lstm 5000` instead for a recurrent checkpoint. MuZero has a convenience
wrapper that finds the newest checkpoint and restores replay and saved settings:

```bash
./run.sh continue models/my-muzero-run 5000
```

Resume loads the exact checkpoint named by `RESUME`; it does not silently substitute
`champion/`. Use the same algorithm, curriculum, output directory, and history layout.

Do not resume legacy stationary checkpoints into this curriculum. They remain loadable for
matching legacy configs, but new models use fine/coarse turning actions, ship velocity,
26 asteroid slots, explicit movement/invulnerability inputs, and different reward semantics.

## MuZero memory and correctness

This is a custom compact MuZero-style learner; `mctx` supplies tree search, not the network
or replay/training implementation. Learned dynamics rolls hypothetical futures forward but
is not memory of earlier real observations.

The model receives eight dense history samples plus eight samples at stride eight, reaching
about 4.7 seconds back. Tracks are keyed by asteroid ID and wrap-aware. Ship velocity and
weapon cooldown are observable. Dynamics also predicts continuation so search discounts
death and successful completion. Versioned checkpoint manifests record action names,
capacity, and exact history offsets and reject incompatible games.

## Metrics and comparison

Logs include completion, waves, clear time, survival, shots fired/resolved/hit/missed, raw
and resolved accuracy, score, size-specific kills, every reward component, and terminal
reason. `closest` is the deterministic no-prediction controller; it rotates toward the
nearest current rock and fires but does not thrust.

```bash
python -m asteroid_survival evaluate-baseline \
  --config configs/rl-arcade.toml --episodes 100 \
  --output metrics/closest-arcade.json
```

Model selection for `showdown`, `compare`, `watch`, and `preview` is two-stage: the newest
run, then the best held-out checkpoint inside it. Ranking by score across all runs instead
looks reasonable and is quietly wrong -- an abandoned run that reached a later curriculum
stage outranks the run being trained, so a stale model keeps getting picked. `ANY_RUN=1`
restores the old behaviour, and `CHECKPOINT=path` overrides both.

`showdown` displays human, greedy, and a selected RL agent in one shared arena. It uses
MuZero by default; pass `ppo` to select the best compatible PPO or LSTM-PPO checkpoint.
Pass `ppo-nonlinear` for the round-48 target arena: all eleven nonlinear patterns at 49–64
px/s, 82–145 px amplitude, and 2.62–3.74 second periods.
`compare` runs them alone on identical seeds and is the controlled measurement.

## Frequently needed answers

- **Did training finish?** `./run.sh status models/my-run` distinguishes a stopped process
  from a mastered curriculum. An exhausted episode budget can stop before stage 28 passes.
- **What stage is next?** Read `stage` in `curriculum_state.json`; it is zero-based, while
  terminal output and this README use one-based stage numbers.
- **Which model should I play?** Pass the run directory to `preview`, or use `champion/`
  explicitly. It is the protected held-out winner.
- **Which model should I resume?** Use the numerically newest `checkpoint_*`, not
  `champion/`, unless deliberately branching from the champion.
- **How long should I resume?** Each stage needs at least two evaluation opportunities, so
  remaining stages require at least `remaining × 500` episodes in the ideal case. Budget
  roughly twice that when retention or harder mechanics may delay promotion.
- **Does `10000` mean total or extra on resume?** Extra. Resuming episode 12,000 for 10,000
  episodes targets episode 22,000.
- **Why is the champion behind the learner?** The learner can advance while the protected
  champion remains at the last evaluation that improved the eligible held-out score.
- **Can showdown use PPO?** Yes, and it is the default. `ALGO=muzero ./run.sh showdown`
  picks MuZero instead. Showdown works in every mode: `./run.sh showdown survival 12`.
- **Where are rewards and wave definitions?** The exact values, 28 foundation rounds, and
  extended rounds 29–48 are documented above. The sources of truth are
  `configs/rl-curriculum.toml` and `configs/rl-nonlinear.toml`; the 96-round survival ladder
  is `configs/rl-survival-small.toml` through `configs/rl-endless.toml`, chained by
  `extends`, with the two-ship tier in `configs/rl-endless-coop.toml`.
- **Why does completion stall below 100%?** Completion is `exp(-hazard x seconds)`, so
  80% -> 90% over a thirty-second round means roughly *halving* the death rate, and 100%
  means a death rate of exactly zero. It also means a sixty-second target is the square of
  the thirty-second one: 90% at 30s is 81% at 60s.

## Architecture

`Simulation` is Pygame-independent and deterministic:

```python
state = simulation.reset(seed=123)
result = simulation.step({"alpha": Action.THRUST_FIRE})
```

The RL adapter repeats actions for four frames. New curriculum observations contain 1,235
floats (1,243 in the two-ship tier, which appends one 8-float teammate slot): mobile ship state plus relative asteroid geometry, velocity, bearing,
closing/tangential speed, two-tier trajectory history, and active projectiles.
