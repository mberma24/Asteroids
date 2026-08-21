# Worklog

Running record of what has been changed and what was learned, so work can resume across
sessions without re-deriving it. Newest context at the top; the detail is chronological
below.

Last updated: 2026-08-21.

---

## 2026-08-21: the plateau is a promotion gate, not a learning ceiling

**Numbering:** every "stage N" below is the **zero-based** index used by
`curriculum_state.json` and `evaluation.jsonl`. The training log and the stage names
are one-based rounds, so index 20 is `survival-v2-round-21`. Subtract one when reading
the log, add one when reading the JSON.

**The flat scores are two separate problems.** Within a stage the agent barely improves:
fitting a trend to each stage's own evaluations gives +0.003 completion per 1,000 episodes
against a per-evaluation noise of sigma ~= 0.035. Stage 16 of `ppo-v2-kl-0820-0044` got
9,750 episodes and moved by less than one evaluation's noise; stage 19 of
`oracle-survival-v2` got 12,750 episodes at a mean clear rate of 0.723 and then promoted on
a 0.828 draw. Two independent runs, same signature.

**Promotion is noise-gated.** The gate is `promotion_completion = 0.90` AND
`promotion_clear_rate = 0.80`, needing 2 passes in a 4-evaluation window over 64 episodes.
At a true clear rate of 0.707 the odds a single evaluation reads >= 0.80 are 0.039, which
predicts 60-90 evaluations of waiting; stage 16 took 39 and stage 19 took 32. The dwell time
is explained by the binomial, not by learning.

**Which gate binds.** Across 105 Oracle evaluations: 16.2% pass both, 6.7% pass completion
but fail clear, 1.0% pass clear but fail completion. Clear rate blocks 7x more often.
Retention, the third condition in `consider_promotion`, has never blocked a promotion (0 of
105).

**Why they differ so much on the same episodes.** `clear_rate` is binary per episode
(`completed_stage`), while `completion_rate` for a survival curriculum is
`survival_fraction` = mean of `min(1, survival_time / limit)` -- partial credit. Dying at 26s
of a 30s cap scores 0.87 completion and 0.00 clear. The agent almost always *nearly*
survives, so the 0.90 partial-credit gate is mild and the 0.80 binomial gate is the wall.

**Entropy was pinned.** `entropy_loss` sat at 1.07 nats for 25,000 episodes without drift
(max is ln(16) = 2.77 for the mobile action set). `ent_coef = 0.01` holds the policy at
~3-way random every decision in a game where one bad decision is fatal. `explained_variance`
also sat flat at ~0.35, and `target_kl = 0.02` was early-stopping every update.

**Experiment.** Added `--ent-coef` to `train-ppo` (commit c9bb424), applied after model load
like `--learning-rate` so `--resume` still restores the checkpoint's other settings. Two
local arms from `ppo-v2-kl-0820-0044/checkpoint_025500`, same seed and eval seeds, on stage
18:

| arm | evals | completion | clear | promoted |
|---|---|---|---|---|
| ent_coef 0.01 | 7 (1,500 eps) | 0.869 | 0.717 | no |
| ent_coef 0.0025 | 5 (1,000 eps) | 0.905 | 0.772 | yes, ep 26,750 |

The low-entropy arm then posted 0.828 and 0.781 clear on stage 19 immediately. The gap is
~1.8 sigma (p ~= 0.08), so **not formally significant** -- it is corroborated by three
independent signals (completion, clear, an actual promotion) and did not decay as
evaluations accumulated, but it has not been established.

**Running overnight.** `models/oracle-survival-v2` was stopped at episode ~33,250 and forked
from its `checkpoint_033000` (stage 20) into `models/oracle-lowent` with
`--ent-coef 0.0025`, everything else identical. The old directory is intact and resumable;
`models/fork-point-032000` is a preserved copy in case checkpoint pruning removes 033000.
Oracle has only 4 cores, so the two runs cannot run concurrently -- the banked 105-evaluation
history of `oracle-survival-v2` is the control.

**Next.** Compare `oracle-lowent` stage-20 clear rate against the banked
`oracle-survival-v2` stage-20 baseline (4 evaluations, mean clear 0.656). Independently of
the entropy result, the gate should judge the pooled ~256 episodes across the promotion
window rather than requiring two individually lucky evaluations; that reclaims statistical
power already being paid for. If entropy decay does not move it, revisit
`explained_variance` 0.35 and `target_kl` 0.02.

---

## 2026-08-19: survival v2 and the real multi-agent endpoint

- Preserved `models/ppo-survive-0819-1852/champion`: 22,000 episodes, 8,561,128 environment
  steps, human round 16. V2 forks from it instead of resuming across tasks.
- Fixed SB3 learning-rate persistence across model, schedule, optimizer, metadata, and
  champion tracking. Removed automatic PPO restoration from evaluation callbacks because
  swapping policies under an on-policy rollout invalidates that rollout.
- Added four rotating validation panels, an untouched final-test band, hybrid survival/clear
  mastery, PPO update telemetry, layout-v5 difficulty/threat inputs, and time-prorated
  survival shaping.
- Added a separate 96-round `rl-survival-v2.toml`: curves begin on round 3; every round after
  28 contains all eleven nonlinear movement types and no linear samples.
- Added a simultaneous shared-policy environment and MAPPO-style centralized critic. Every
  ship contributes actions, rewards are normalized by team size, individual deaths do not
  end the episode, and asteroid/projectile/teammate sets are pooled invariantly.
- Added collision, friendly-fire, cooldown-mask, team survival, and protected-object metrics
  and rewards for 1–8 ships.
- Durable PPO metadata totals 67,446,556 steps and about 20.95 wall hours: approximately
  12.50 CPU hours and 8.45 Apple-MPS hours. CUDA cloud GPU hours used so far: zero.

---

## 2026-08-19: straight restored to an equal share in v2; fork rungs are now measured

Two fixes to the survival-v2 line codex added.

**Straight is one trajectory among equals again.** v2 shipped with `linear_probability`
tapering to **0.0 from round 29 on**, so 68 of 96 rounds contained no straight-line motion
at all. Every phase now sets `1/(len(patterns) + 1)`, so a rock is exactly as likely to fly
straight as to fly any one curve; the share falls only because the pool grows (25% with
three curves, 14.3% with six, 8.3% with eleven). Pinned by
`test_straight_line_is_one_trajectory_among_equals`, and codex's test asserting
`linear_probability == 0.0` after round 28 was rewritten to assert the full eleven-pattern
pool instead.

**`train-ppo-survival-v2` measures its fork rung instead of hardcoding 16.** It now binary
searches for the lowest round the source policy cannot already clear, and defaults
`PPO_DEVICE=cpu`. Both defaults had bitten in the same evening: the fork landed at round 16
where the source cleared only 56%, five rungs above where it could promote, and `auto`
picked MPS at 294 dec/s against 799 on CPU.

### The champion tracker inflated a champion again

Measured on identical seeds (24 per round), the champion of a 2,000-episode run at round 11
was **worse than the checkpoint it was initialized from**:

| round | parent champion | child champion |
|---:|---:|---:|
| 1 | 83.3% | 66.7% |
| 8 | 95.8% | 70.8% |
| 11 | 83.3% | 50.0% |

The child's own held-out record read 92.3% survival / 81.2% clear at the episode it was
crowned. Same failure mode as the round-41 phantom champion: the champion is the maximum
over many noisy evaluations, so it is biased upward, and `evaluation_panels = 4` means the
crowned reading came from one rotating seed panel rather than the whole distribution. The
fork search is the cheap guard -- it re-measures a candidate on fixed seeds before any
weights are reused.

Relaunched as `models/ppo-v2-equal-0819-2348` from the *parent* champion, which the search
placed at **round 13** (round 12 passes at 90.6% clear, round 13 fails at 81.2% clear
against a 90% survival gate).

Also worth knowing for v2: greedy clears rounds 1-11 at 100%, and idling clears 16-42% of
the early rounds -- looser than the old ladder's 25% ceiling, though the 80/90% gate still
makes idling unpromotable.

---

## 2026-08-19: documentation re-audited against the code

`README.md`, `COMMANDS.md`, and `run.sh --help` had drifted from the ladder as it actually
loads. Everything below was re-measured, not re-read:

- The ladder is **96 rounds**, not 90. Tiers: 1-10 small, **11-16 bridge** (one medium in
  four), 17-26 medium, 27-38 large, 39-52 curves, 53-82 full, 83-96 mixed. Co-op is
  **97-126**, replaying the full tier from round 53's envelope.
- **Idling clears at most 25%** (round 1), 12% by round 10, 0% from round 52 on -- 24 seeds
  per round. The docs claimed "0-17%", and claimed twelve opening rocks where the config
  has eight.
- **Greedy** clears 100% through round 27, 96/92% at rounds 33/38, then 71% at 39 as
  curvature arrives, 38% at 53, and 0% from round 60 on. The docs claimed it held to round
  20 and hit zero at 66; the real crossing of the 90% gate is round 38.
- The README's survival reward table was still the pre-survival-ladder one (+0.02/decision,
  +5 clear, 0.2/0.5/1.0 hits). Actual: **+0.10/decision (+45 a round), +10 clear,
  0.15/0.30/0.60 by size, -5 death, -0.02 miss**.
- `ENCODER=set` and the CPU-vs-MPS finding were documented only in `run.sh --help` and this
  worklog; both are now in the README and COMMANDS.

One real bug fell out of the audit: `cmd_train_ppo_coop` defaulted `START_STAGE=91`, left
over from the 90-round ladder, so the co-op tier would have started five rounds inside the
*solo* mixed tier instead of at round 97. It also seeded itself from `best_checkpoint
survival 30`, now round 53, where the tier actually begins.

Also recorded in the README: completion is `exp(-hazard x seconds)`, which is why the gate
feels like a wall. Going 80% -> 90% on a thirty-second round is halving the death rate, and
100% requires it to be exactly zero. Measured on round 10 with a deterministic champion:
identical outcomes on 64/64 seeds across two passes, 57/64 cleared, and greedy clears **all
7** of the seeds the model failed -- so the gap is policy quality, not unwinnable layouts.

---

## 2026-08-19: straight-line motion is one trajectory among equals

`linear_probability` on the survival ladder is now `1/(len(patterns) + 1)`, so a straight
line is sampled exactly as often as any curve rather than being singled out with its own
share:

- curves tier (3 patterns + straight): 0.5 -> **0.25**
- full and mixed tiers (11 patterns + straight): 0.25 -> **0.0833**
- the co-op tier follows the full tier

Verified by counting spawns over 120 episodes: straight lands at 25.9% in the curves tier
against an expected 25%, and 9.0% in the full tier against an expected 8.3%.

The old 0.25 was the same mistake as an all-curves field, in reverse: a field of nothing but
curves teaches a policy to always expect a turn, and over-weighting straight teaches it the
opposite. Neither is the distribution the game actually has.

This is a stage-spec change, so the task hash moves and the run could not be resumed.
Relaunched as `models/ppo-ladder-v2-0819-1021` with `INITIALIZE_FROM` on the previous
champion and `START_STAGE=9`, the round it had reached.

### Unrelated, from the same session: PPO was on the wrong device

`PPO_DEVICE` defaults to `auto`, which selects MPS, and stable-baselines3 prints an explicit
warning that PPO with an MLP policy is slower on a GPU than on CPU. That warning appeared in
every training log for hours and was filtered out as noise.

Benchmarked at the real observation size (1235 inputs, 256x256 nets, 8 envs):

| | inference, 8-env batch | one 2048-step learn cycle |
|---|---:|---:|
| CPU | 0.31 ms | 0.78 s |
| MPS | 1.52 ms | 4.81 s |

End to end the run went from **470 to ~1030 decisions/s**, a 2.2x speedup for free. The
learn cycle dominates and is 6x slower on MPS: the networks are far too small for a GPU to
pay for its transfer overhead. Every survival run since is launched with `PPO_DEVICE=cpu`.

The default should probably change from `auto` to `cpu`; left as is for now so the switch
stays visible in the launch command.

---

## 2026-08-18: survival promotion gate raised to 90%

`promotion_completion = 0.90` on the survival ladder (set in `rl-survival-small.toml`, the
root of the chain, so every tier and the co-op ladder inherit it). Retention stays at 0.75,
giving a wide hysteresis: promote high, fall back only on real decay.

Chosen by measurement, not feel. A thirty-second round is markedly easier to survive than the
sixty it replaced, so the old 80% gate no longer demanded the same competence. Greedy
baseline on the same round at both limits:

| round | 30s | 60s | inflation |
|---:|---:|---:|---:|
| 26 | 88% | 78% | +9 |
| 32 | 84% | 66% | +19 |
| 36 | 59% | 41% | +19 |
| 44 | 41% | 9% | +31 |
| 47 | 59% | 12% | +47 |

In the region where the gate actually bites -- rounds 26-32, where completion sits near the
old bar -- the shorter round reads 9-19 points higher. 90% restores roughly the competence
80% demanded at sixty seconds. 85% would have recovered only about half of it.

Noise is not the binding constraint at this gate, which is what makes 90% safe rather than a
repeat of the stall. At 64 evaluation episodes, chance of promoting within one
four-evaluation window:

| true rate | gate 80% | gate 85% | gate 90% |
|---:|---:|---:|---:|
| 85% | 99% | 69% | 9% |
| 90% | 100% | 100% | 74% |
| 92% | 100% | 100% | 95% |

A 90% gate needs a true completion of 91.4% to promote reliably -- 1.4 points of margin,
against 2.6 at an 80% gate. The 32-episode noise trap that once pinned a run does not apply
at 64 episodes.

`promotion_completion` is a `[curriculum]` setting and not part of the hashed payload, so the
task hash is unchanged and the in-flight run was resumed rather than restarted.

---

## 2026-08-18: survival ladder rebuilt -- 90 rounds of 30 seconds, size ladder foundation

Run: `models/ppo-ladder-0818-2209`, seeded from the previous survival champion, 40,000-episode
budget. Entry point is still `configs/rl-endless.toml`, which now pulls the whole chain.

### Why

The previous ladder stalled on round 1: the model climbed 41% -> 57% then flattened at ~54%
against an 80% gate, with **zero promotions in 4,000 episodes**, dying at ~45s of 60 as the
field saturated. A foundation is supposed to be walked through, not ground on.

### Six tiers, one new thing each

| Rounds | File | Size | Motion |
|---:|---|---|---|
| 1-10 | `rl-survival-small.toml` | small | straight |
| 11-20 | `rl-survival-medium.toml` | medium | straight |
| 21-32 | `rl-survival-large.toml` | large | straight |
| 33-46 | `rl-survival-curves.toml` | large | 3 gentle curves, half straight |
| 47-76 | `rl-survival-full.toml` | large | all eleven, 25% straight |
| 77-90 | `rl-endless.toml` | mixed | all eleven, 25% straight |

Co-op rounds move to 91-120 and mirror the *curved* tier, not the hardest one: when a new
axis appears, the others reset to a level already mastered, as the arcade curriculum does.

### `asteroid_size`, and why it is not called `spawn_size`

`game_config` hardcoded `spawn_size = 3` for survival stages, so the ladder could only spawn
large. New `CurriculumStage.asteroid_size` (survival-only, default 3, accepts a size name,
1-3, or `mixed`/`None` for a per-spawn roll).

**Not** named `spawn_size`: that key already means something else in a hand-written stage
table (`curriculum.py:352`), where it is the legacy per-wave fallback that desugars into a
composition. Reusing it would have collided silently. Default 3 means `_specified` strips it
and no existing task hash moves -- verified against digests captured before the change, and
now frozen in `test_the_wave_curricula_task_hashes_are_frozen`.

### What 30-second rounds actually change

- Round 1 goes from 54% to 90.6% for the same policy. Measured beforehand: the shortened
  limit alone does most of the unsticking; the size ladder is what makes the first thirty
  rounds a shaped progression rather than one wall with a shorter clock.
- **Idling gets much easier**, because the field has half as long to fill. On the old round 1
  settings at 30s, a do-nothing policy cleared 38-75%.
- The lever that fixes it is `initial_asteroids`, not `spawn_spread`. Measured at 30s: with 6
  rocks in flight idling cleared 50-58% at *every* spread from 60 to 150 degrees; with 10 it
  fell to 21-33%. Round 1 now starts with twelve.
- `spawn_spread` is the wrong knob for a different reason too: `heading_mode = "aimed"` points
  rocks at the arena *centre*, not at the ship, so a narrow spread is dodged by not being in
  the middle. Opening the ladder there would have taught corner-camping as the foundation
  skill. Spread keeps a monotone 170 -> 24 degree envelope and is never spent early.
- Large rocks are the *most* idle-hazardous size, not the least -- three times a small's
  radius. Size is a gentle axis for a policy that plays, not for one that stands still, which
  is why the size ladder needs the `initial_asteroids` floor underneath it.
- Reward rates are unchanged. Halving the round halves both the survival total (90 -> 45) and
  the kills available, so the survival:hits ratio the ladder was tuned with is preserved.
  `round_clear` and `death_penalty` become ~2x more significant per unit time, which is a
  reasonable direction for a survival task.

### Calibration (24 seeds per round)

| round | 1 | 10 | 20 | 32 | 40 | 47 | 56 | 66 | 90 |
|---|---|---|---|---|---|---|---|---|---|
| idle | 12% | 8% | 12% | 4% | 4% | 4% | 0% | 0% | 0% |
| greedy | 100% | 100% | 100% | 75% | 46% | 33% | 4% | 0% | 0% |

Idling never exceeds 17% anywhere; the do-nothing test bar was tightened from 50% to 30%,
because at thirty seconds the old bar was not a real guard. Greedy stalls at the 80% gate
around round 32, leaving nearly sixty rounds of headroom.

### Still open

`promotion_completion` is 0.80, and at 30s each episode carries half the evidence, so the gate
is a structurally weaker bar than it was at 60s. Raising it to 0.85 for the survival ladder is
worth considering, but as a separate change so this one stays measurable on its own.

---

## 2026-08-18: survival became its own ladder -- three tiers, survival-dominant reward

Run: `models/ppo-survival-0818-2011`, seeded from the wave champion, 30,000-episode budget.

### Reward is survival-dominant now

`survival_bonus` 0.02 -> 0.10 per decision (90 over a full round), `round_clear` 5 -> 10,
hits cut to 0.1/0.25/0.5. Measured on the old weights, kills were **77% of positive reward**
and staying alive only 19% -- the opposite of what the ladder scores. Hits remain meaningful
shaping, because the field fills until it is unsurvivable and clearing it is how breathing
room is won.

### Three tiers, from straight lines to every curve

| Rounds | Motion |
|---:|---|
| 1-8 | straight lines only |
| 9-16 | half straight, three gentle curves (`sine`, `arc`, `s_curve`) |
| 17-46 | all eleven patterns, 25% straight |

Chained through `extends`: `rl-endless-linear.toml` -> `rl-endless-curves.toml` ->
`rl-endless.toml`. Each tier needs its own progression block because patterns and motion mode
cannot be stepped within one.

### Stationary rounds were proposed, measured, and rejected

A survival round is passed by staying alive, motionless asteroids never reach the ship, and
spawns are already guarded away from it -- so **doing literally nothing clears a stationary
survival round 24 times out of 24, with zero kills**. Under a survival-dominant reward that
is worth more than playing. The foundation is linear instead: rocks that actually close on
the ship.

Every round now fails a do-nothing policy: 19% at round 1, 0% from round 17.

### The joins had to be bridged carefully

The first tuning made tier one *denser* than tier three's opening (4.2s spawn interval
against 5.0s), so difficulty stepped backwards at the join. Fixed by taking early density
from asteroids already in flight (6 at round 1, easing to 4) rather than from a tight spawn
clock, which lets the interval fall monotonically 5.9 -> 5.2 -> 5.0 into the curved tier.
That is why `initial_asteroids` decreases across the foundation and only ramps in tier three.

Greedy baseline on the finished ladder: 100% to round 22, 56% at round 27, 0% from round 30
-- it stalls at the 80% gate around round 25, leaving twenty rounds of headroom.

The co-operative tier now starts at round 47 (`START_STAGE` default updated), and its ladder
is 76 stages.

---

## 2026-08-18: switched to the survival ladder; spawn and deadline fixes

- Training moved off the wave curriculum onto `configs/rl-endless.toml`, starting at round 1,
  seeded from the wave champion (`ppo-nonlinear-v3-0818-1803/champion`, 61.5% on round 40).
  Run: `models/ppo-endless-0818-1838`, learning rate 2e-4, stops when round 30 is mastered.
- Reasoning, with the numbers that drove it. The wave curriculum's gate is "clear one wave
  perfectly, four times in five", which diverges from surviving as long as possible the
  further up it goes. Greedy on the rounds ahead, with corrected fragments: round 40 41.7%,
  round 41 16.7%, round 44 4.2%, **round 48 0.0%**. Reaching round 48 is days of training
  toward a milestone that does not measure the objective.
- The model transfers usefully: on the survival ladder it already reads 58.3% at round 5,
  37.5% at round 10, 4.2% at round 15, so there is a real ladder above it.
- Known weakness carried over: accuracy 0.48 against greedy's 0.63. It out-kills greedy by
  spraying. Survival rewards staying alive rather than shooting well, so this will likely
  persist; fixing it is a reward-shaping change (`miss_penalty`, or the accuracy bonus).

### Asteroids could spawn on top of the ship

- The safe-radius guard only covered the opening field. Interval spawns come from the arena
  edges, which is safe only while the ship is away from an edge -- and the arena wraps.
  Measured worst clearance was **-21.9px**: asteroids materialising *inside* the ship, nine
  times over forty episodes. Unavoidable deaths, not missed dodges.
- Every spawn is now checked, using the trajectory's real position at age zero. After 32
  failed attempts it spawns anyway rather than skipping, because silently thinning the field
  would make a crowded round easier exactly when it is hardest.
- Config-gated on `spawn_safe_radius`/`spawn_safe_seconds`, so wave curricula that leave them
  at zero are byte-identical and the running job was unaffected.
- After: 0 spawns within 60px over 520 spawns; worst clearance 162px at round 15 and 220px at
  round 30, the difference being the speed-scaled component.

### Survival rounds no longer see the clock

- The observation's fifth feature is `elapsed / episode limit`. In a survival round that
  reveals exactly when the sixty-second cutoff arrives, inviting a policy to key on the
  deadline -- coasting into it, or giving up past it -- which is the opposite of surviving
  indefinitely. The limit exists only to bound episode cost.
- Survival rounds now read 0.0 there; wave rounds still see it, since clearing quickly is
  genuinely rewarded. The slot is kept either way, so the observation layout is unchanged and
  no checkpoint is invalidated.

### `run.sh` had two commands dispatched but not defined

- The mode-unification refactor sliced out everything between `cmd_play` and `cmd_baseline`,
  which silently took `cmd_train_ppo_endless` with it; a later edit anchored on that missing
  function, so `cmd_train_ppo_coop` never landed either. Both dispatch entries still existed,
  so `./run.sh train-ppo-endless` failed with "command not found" at the moment it was needed.
- Restored, and `test_every_dispatched_command_actually_exists` now checks the script's
  internal consistency -- nothing else did.
- `retention_sample = 10` also added to the survival ladder, for the same reason as the wave
  curriculum: the full prior-stage sweep approaches the cost of the training it interleaves.

---

## 2026-08-18: fragment speeds fixed, linear mixed back in, two-ship tier built

### Fragments now outrun their parent everywhere

- `CurriculumStage.game_config` set `medium_speed_multiplier` and `small_speed_multiplier` to
  1.0 for wave stages while arcade play, endless mode, and the survival ladder all use 1.15
  and 1.35. A policy trained on the wave curriculum had therefore never seen debris move
  faster than the rock it came from.
- That is what made it die in `showdown`: on arcade it out-shot the greedy baseline (48.6
  kills against 40.4, 2.0 waves against 1.7) and still died at 41.7s while greedy survived
  the full sixty. It was not bad at the game and it had **not** forgotten linear motion --
  measured 93.8% and 100% completion on the linear foundation rounds -- it was mispredicting
  its own splits.
- Now 1.15/1.35 in every mode. Cost to the current model is close to nil: re-measured under
  the corrected speeds the champion reads 59.4% on round 41 and 84-91% on rounds 37-39.

### Dry-spell windows are measured, not guessed

- Faster debris takes longer to chase down, which pushed three stages past their
  `no_hit_seconds` window and tripped the balance guard.
- Rather than hand-tuning, every window is now derived: the longest real gap between kills
  the greedy baseline needs on that stage over sixteen seeds, with 60% headroom and a 20s
  floor, measured with the cutout disabled so the number is not censored by the cutout
  itself.
- This exposed that several windows were already far too tight, before any of today's
  changes: stage 13 needed 33s and allowed 17, stage 25 needed 58s and allowed 68 only after
  a first pass. Competent play was being cut off as though it had stalled.

### Survival rounds mix linear motion back in

- `linear_probability = 0.25` on the survival ladder, alongside all eleven curves; measured
  at 25.7% over sixty episodes. The progression generator hardcoded `0.0`, so this was not
  previously expressible from a config at all.

### Tier two of the survival ladder: two ships, one policy

- `configs/rl-endless-coop.toml`: sixty stages, the first thirty inherited solo rounds and
  the next thirty the same difficulty envelope flown by two ships. `friendly_collisions =
  "full"`, so bumping kills both ships and a shot that lands on a teammate kills it.
- `CurriculumStage.ships` drives it; `CurriculumSpec.max_teammates` sizes the observation for
  the busiest round in the curriculum, because sizing per stage would change the observation
  shape part-way through a run.
- `TEAMMATE_FEATURES` are appended **last** in the encoding, after the projectile block, so
  every earlier input weight keeps its meaning. A single-ship encoding is byte-identical to
  what it was.
- Companions run `SnapshotPolicy`, which reloads `companion.zip` whenever the trainer
  rewrites it. Environments are separate processes under `SubprocVecEnv`, so the live network
  cannot be shared with them; a companion is a slightly stale copy of the learner, which is
  ordinary self-play practice.
- `widen_policy` lets a solo model seed the tier despite the wider observation: matching
  parameters are copied, and a first-layer weight matrix that has grown columns keeps its
  learned columns and zero-fills the new ones, so the transferred policy behaves identically
  until it learns to use teammates.
- `./run.sh train-ppo-coop [N]` transfers from the best survival checkpoint and starts at
  round 31.

**All of this changes the task hash**, deliberately: fragment speeds and dry-spell windows are
part of what a stage configures. The in-flight run cannot be `RESUME`d across it and needs
`INITIALIZE_FROM`.

---

## 2026-08-18: half of every run was going into evaluation; retention now rotates

- Measured, at curriculum stage 41: one evaluation costs ~100,700 decisions (40 prior stages
  x 8 episodes = 66,700, plus 96 current-stage episodes = 34,100) against ~99,300 decisions
  of training in the 250 episodes it interleaves with. **Evaluation was 50% of every
  decision the run made**, and the prior-stage sweep was two thirds of that.
- `retention_sample` (10 in `configs/rl-curriculum.toml`) scores a rotating subset of prior
  stages instead of all of them. The rotation is deterministic, so coverage is even and runs
  stay reproducible: with 40 prior stages every one is revisited every fourth evaluation.
- Safe because retention is already judged on the pooled sample. Skipped stages are recorded
  with `episodes: 0`, which `retention_holds` and `forgotten_stage` both treat as "no
  evidence" rather than as a failure -- that guard is what makes the sampling correct rather
  than merely cheaper.
- Evaluation share drops from 50% to 34% of decisions: about **33% more training per second**
  at the same CPU.

### Has round 41 actually been learning? No.

Measured from the training log, which is a far better instrument than the evaluations
(n ~ 900 per block, standard error 1.7%, against 32-episode evaluations swinging +-17%):

| episodes | completed | accuracy | survival |
|---|---:|---:|---:|
| 8753-9953 | 57.2% | 0.523 | 25.5s |
| 9953-11153 | 52.9% | 0.525 | 25.1s |
| 11153-12353 | 49.2% | 0.536 | 23.6s |
| 12353-13553 | 57.0% | 0.523 | 25.8s |
| 13553-14753 | 54.6% | 0.527 | 25.1s |

Flat across 6,000 episodes. Accuracy creeps up (0.523 -> 0.548) while completion does not, so
the policy is refining its shooting and not its survival -- consistent with the 47.8% death
rate and 0% timeout rate.

**But the learning rate was pinned at 3.75e-5 for most of that window**, so "the task is too
hard" and "the step size was too small" are still confounded. The control fixes and the LR
reset landed at episode 14750; the clean experiment is to re-run this same table after a few
thousand episodes. Only if it is still flat does the gate or the composition growth need
tapering.

## 2026-08-18: champion selection made noise-aware; run unpinned and resumed

Implemented the four fixes from the slump diagnosis below.

1. **Champion selection is smoothed.** `PPOChampionTracker` keeps a rolling window of the
   last `SMOOTHING_WINDOW = 3` evaluations at the current stage and compares the mean, not a
   single reading. A fluke is now worth a third of the signal instead of crowning a phantom.
   The window resets on promotion: completion on a new stage says nothing about the old one.
2. **Restoration is skipped when training is level with the champion.** Falling back only
   happens if the smoothed rate is genuinely *below* the champion's estimate. Previously five
   restorations discarded weights that measured as good as, or better than, the snapshot they
   returned to.
3. **Learning-rate floor raised from initial/8 to initial/3.** The old floor was reached
   inside a single curriculum round, after which the run could not move. Cutting the rate is
   the right response to instability, not to a hard stage where progress is merely slow.
4. **`evaluation_episodes` 32 -> 96** in `configs/rl-curriculum.toml`. Standard error at the
   completion rates that matter falls from ~8.4 points to ~5.0, so control decisions stop
   being driven by noise. Costs 64 extra episodes per evaluation.

Restart details:

- Resumed from `checkpoint_014750`, not the champion: measured on 96 episodes the champion
  was 57.3% and the latest checkpoint 58.3%.
- The stale champion is set aside as `champion-phantom-84pct` / `champion_state-phantom.json`
  rather than deleted. Its recorded 84.4% was the artefact; keeping it makes the failure
  reproducible. It must not be left named `champion`, or the tracker reloads the phantom bar
  on resume and re-pins the run.
- Learning rate overridden to 2e-4, above the new floor of 1e-4 and below the original 3e-4.
- Four tests pin the new behaviour: a single spike does not install; a sustained improvement
  does; a promotion always installs; restoration is skipped when level but still fires on a
  genuine collapse.

## 2026-08-18: the round-41 slump is a phantom champion, not a learning failure

Measured on `models/ppo-nonlinear-v2-0818-1152`, round 41, 21 evaluations of 32 episodes.

- Observed completion swings 43.8% to 84.4% with no trend, mean 64.7%. The standard error of
  a single 32-episode evaluation at that rate is 8.4%, so **+-17% is routine noise**.
- The champion was crowned at episode 11500 on a reading of 84.4%. Re-evaluated on 96
  episodes it is actually **57.3%**. A single reading that high is a 1.2% event at the true
  rate; across 21 evaluations, seeing at least one is 23% likely. The champion's score is a
  max over noisy draws, not a level the policy ever reached.
- The latest checkpoint measures **58.3%** on the same 96 episodes -- as good as or better
  than the champion. So the five restorations have been discarding equal-or-better weights to
  return to a lucky snapshot.
- Because nothing can beat a phantom 84.4%, "evaluations since improvement" never resets, the
  plateau controller keeps firing, and the learning rate is pinned at its floor of 3.75e-5,
  eight times below where it started. The run cannot move.
- Promotion needs two readings >= 80% inside a four-evaluation window. At a true rate of 57%
  that is roughly a 0.7% chance per window -- the gate is now waiting on a fluke rather than
  on learning.
- Accuracy is stable and slightly rising (0.50 -> 0.54), so aiming is still improving. The
  failure is survival in a dense field, as the 47.8% death rate and 0% timeout rate show.

Fixes to make before restarting, in order of importance:

1. Do not crown a champion on one evaluation. Require the improvement to exceed the standard
   error, or compare moving averages of the last few evaluations. This is the root cause.
2. Re-measure a champion before restoring to it, so a phantom cannot pin the run.
3. Raise the learning-rate floor, and do not cut the rate when the metric is noisy-flat
   rather than declining.
4. Raise the current-stage evaluation from 32 episodes; at 96 the standard error falls from
   8.4% to 4.8%. Evaluation already costs 1.3 episodes per training episode, so this is not
   free, but control decisions are currently being driven by noise.

Open design question, separate from the bugs: true completion is 57-58% against an 80% gate,
so the policy does have ~22 points to find. The curriculum adds a large rock every two
rounds; tapering the gate for later rounds, or slowing composition growth, would keep
progress moving if the control fixes alone are not enough.

---

## 2026-08-18: COMMANDS.md rewritten as a parameter reference

- Every command now documents its positional parameters and their real defaults, taken from
  `run.sh` and then verified by dry-running each one rather than transcribed from memory.
- New up-front section on the shared mode grammar: `./run.sh <command> [mode] [round] [runs]`,
  which modes accept a round, and the rule that a bare number is always the run count.
- Tables for `versus` (`MODELS`, `HUMAN`, `GREEDY`, `OUTPUT`), for the training variables,
  and for the variables that apply everywhere.
- Documents the `RESUME` vs `INITIALIZE_FROM` distinction explicitly, since that is the pair
  most likely to be reached for after a curriculum or trajectory change and the one where
  picking wrong silently continues a run on a different game.

---

## 2026-08-18: `versus` scores several models against each other

- `./run.sh versus [mode] [round] [runs]`. Every contender plays alone on its own run of the
  same seeds, so nobody clears anyone else's asteroids -- the distinction from `showdown`,
  which is one shared arena. Ranked by mean survival; full per-episode metrics land in
  `metrics/versus.json`.
- `comparison.compare` took a single checkpoint; it now takes a list, labels each from its
  run directory (timestamp stripped unless two runs collide), and can drop greedy or the
  human from the lineup. `MODELS=`, `HUMAN=1`, `GREEDY=0` drive it from `run.sh`.
- First run, endless round 6, six seeds each: greedy 60.00s (clears), old nonlinear champion
  49.51s, v2 champion 44.67s, original feed-forward champion 41.66s. On nonlinear round 41,
  eight seeds: v2 27.16s, greedy 26.72s, ff 24.11s, old nonlinear 22.93s.
- Worth noting from that table: greedy still beats or matches every learned model on these
  measures, and does it with markedly better accuracy (0.63 against ~0.50 on round 41). The
  learned policies destroy more asteroids but aim worse and die sooner.

---

## 2026-08-18: automatic model selection follows the newest run

- `showdown`/`compare`/`watch`/`preview` were resolving to `ppo-nonlinear-0817-2340`, the
  abandoned run trained on the old trajectory dynamics, because `_latest_checkpoint` ranked
  by held-out score across every run and that run had reached curriculum stage 40 while the
  live one was at 35. Hence having to pass `CHECKPOINT=` by hand every time.
- Selection is now two-stage: newest **run** across runs, best held-out **checkpoint** within
  it. Best-within-run is still right -- training is noisy and the final checkpoint is often
  worse than an earlier one -- but best-across-runs never was.
- `ANY_RUN=1` (`--any-run`) restores score-ranking across every run; `CHECKPOINT=` still
  overrides everything.
- `./run.sh status` and `./run.sh preview` with no argument now prefer whichever run has a
  live trainer, falling back to newest on disk. Picking by directory timestamp meant that
  right after a restart the stopped run was reported as "not running", which reads as a
  failed job rather than an intentional stop. `status` also lists the other runs now.

---

## 2026-08-18: eleven patterns integrated into both curricula; nonlinear run restarted

- `PATTERN_NAMES` holds eleven shapes. `configs/rl-nonlinear.toml`, `configs/rl-endless.toml`,
  and `configs/endless.toml` all list every one, and `AsteroidConfig.pattern_pool` defaults to
  the full set, so nothing samples a stale subset.
- Swept every remaining "ten patterns" reference out of `run.sh`, `README.md`, `COMMANDS.md`,
  both curriculum configs, and the tests. Where the count appears in code it is now derived
  from `len(PATTERN_NAMES)` rather than written out, so it cannot drift again.
- Balance re-measured on both curricula with eleven patterns. Endless ladder greedy
  completion 100/100/83/42/17/0 at rounds 1/5/7/9/11/13; nonlinear 100/100/42/8/0 at rounds
  29/34/39/44/48. Both still span usefully, so no re-tuning was needed.
- The old nonlinear run was stopped at 20,500 episodes. It could not be resumed -- an
  eleventh pattern is a real task change and the hash correctly refuses -- so the new run
  uses `INITIALIZE_FROM` on its champion, which keeps the policy weights while resetting
  optimizer and task state.
- Restart stage chosen by measurement, not by guessing. The champion evaluated under the new
  dynamics: round 34 100%, round 37 75%, round 39 66.7%, round 40 66.7%. It was on round 40
  under the old patterns; the harder `serpentine` and the added `brownian` cost it roughly a
  round's worth of margin, so the new run starts at round 35 to re-consolidate before
  climbing. First evaluation confirmed the choice: 96.9% on round 35 at episode 250.
- New run: `models/ppo-nonlinear-v2-0818-1152`, initialized from the old champion, start
  stage 35, learning rate 1e-4, 25,000-episode budget, stopping when round 48 is mastered.
- Deliberately a **new output directory**. Evaluation numbers recorded before and after the
  pattern change are not comparable, and mixing them in one `evaluation.jsonl` would make the
  progress graph lie. `models/ppo-nonlinear-0817-2340` is left intact as the old-dynamics
  record.

---

## 2026-08-18: the ten patterns were near-duplicates; rewritten around along-track motion

- Confirmed first: rounds 29-48 do sample all ten patterns uniformly (~10% each over 200
  seeds). Round 28 and earlier use a single pattern. So the sampling was never the problem.
- The shapes were. Measured pairwise correlation of the old set: `sine` vs `figure_eight`
  **1.000** -- `sin(x)cos(x)` is identically a half-amplitude sine at double frequency --
  with `zigzag`/`arc` at 0.993 and eight pairs above 0.96.
- Root cause was structural, not cosmetic. `trajectory()` only applied a **lateral** offset
  perpendicular to a fixed drift direction, so every pattern was "constant forward speed
  plus a wiggle". A circle, a loop, or a figure eight is unreachable in that parametrisation
  no matter what function is chosen.
- `patterns.py` now returns an along-track offset as well as a lateral one, and `trajectory`
  applies both. Four patterns use it (`arc`, `corkscrew`, `figure_eight`, `spiral`); the
  other six stay pure lateral, which a test pins.
- Worst pair is now 0.509, mean 0.108. Peak speeds span 0.2x-2.6x a sine's, recorded as
  `PEAK_SPEED_FACTOR = 3.0` and used to normalise the velocity observation, which previously
  assumed `amplitude * frequency` was the maximum.
- `s_curve` was briefly a one-shot lane shift that settled onto a straight line. That broke
  `test_no_hit_window_never_cuts_off_the_baseline`: on stage 28 greedy killed 20 of 21 rocks
  and then stalled, because the survivor drifted away in a straight line and never returned.
  It is now a slow repeating sweep. The guard caught a real balance regression.
- Curriculum balance re-measured after the rewrite. Endless ladder greedy completion is
  unchanged within noise (100/92/92/42/17/0 at rounds 1/5/7/9/11/13, previously
  100/96/79/50/12/0). Nonlinear rounds: 100% at 29 and 34, 58% at 39, 0% at 44 and 48.

### `sawtooth` was teleporting; added `brownian`, an eleventh pattern

- A true sawtooth is discontinuous: its position snaps from one extreme to the other between
  consecutive frames, which on screen is a rock vanishing and reappearing across the arena.
  Replaced with an asymmetric ramp (`_ramp`, 82% rise): the same lopsided drift-and-return
  feel, continuous. Max frame step went from a full 2-amplitude jump to 0.05 amplitudes.
- `test_no_pattern_ever_teleports` now checks frame-to-frame continuity for every pattern,
  and `test_reported_velocity_matches_how_the_position_actually_changes` checks each stated
  velocity against a finite difference of its own position. The second is the stronger guard:
  the RL observation feeds asteroid velocity to the policy and the greedy controller leads
  its shots with it, so a wrong derivative silently poisons both.
- New `brownian`: aimless drift, no shape, no period. Built by spectral synthesis (six
  components, golden-ratio-spaced rates, amplitudes falling as 1/f) rather than by
  accumulating random steps, because a pattern is a pure function of time -- there is nowhere
  to keep walker state and an exact velocity is required. Component phases derive from the
  asteroid's own phase, so no two wander alike.
- Its along-axis share is 0.45, not an even split. An even split made it by far the easiest
  of the eleven (75% greedy completion): wandering along the direction of travel cancels
  forward progress, and a rock that never closes cannot threaten. At 0.45 it sits at 58%.
- `PATTERN_NAMES` now has eleven entries and the three curricula list all of them.
- **This changes the task hash, deliberately.** An eleventh pattern is a real task change, so
  `models/ppo-nonlinear-0817-2340` can no longer be `RESUME`d against `rl-nonlinear.toml`.
  `INITIALIZE_FROM` still works, which is what the endless transfer uses anyway.
- The golden-digest test was rewritten. It pinned the digest of `configs/rl-nonlinear.toml`,
  so adding a pattern broke it even though the hashing logic was fine. It now pins the digest
  of a curriculum constructed inline in the test: freezing the output requires freezing the
  input.

### Two patterns made erratic; `serpentine` is now measurably the hardest

- `serpentine` is a triangle wave whose rate is modulated at an irrational ratio to its own
  frequency (`rate 1.7`, `warp 1.6`, divided by the golden ratio). Full-amplitude sweeps,
  sharp corners, and no period: consecutive sweeps differ in length by up to 3.8x.
- `sawtooth` keeps its ramp-and-snap identity but its teeth are stretched and squeezed by a
  slow incommensurate wobble, so the snaps never land on a beat.
- **Erratic did not mean hard, and that had to be measured.** Two earlier designs -- pure
  high-frequency jitter, and jitter riding on a slow sweep -- came out as the *easiest*
  patterns in the set: 67% greedy completion on endless round 10, against 21% for a plain
  sine. Small rapid shakes cover little ground, and oscillating along the direction of travel
  cancels forward progress, so the rocks mill about instead of closing. Difficulty here comes
  from large coherent sweeps. The accepted design keeps those and only makes their timing
  unguessable; it lands at 12%, the hardest of the ten.
- Per-pattern greedy completion, endless round 10, 24 seeds: serpentine 12, sine 21, arc 29,
  zigzag 29, lane_change 33, s_curve 33, figure_eight 38, sawtooth 50, corkscrew 62,
  spiral 62.
- An earlier attempt spaced the rates 0.80/2.09/3.38/5.47 as "golden ratio spaced". Every
  ratio was an exact fraction, so the path quietly repeated every 48.8s. The constant is now
  computed, not rounded, and a test pins it.
- Bounds still hold: worst excursion is exactly 1.000 amplitudes across all patterns and 13
  phases; worst speed 2.50 a*w against the declared `PEAK_SPEED_FACTOR` of 3.0. Worst
  pairwise similarity is unchanged at 0.550.
- Curriculum balance shifted, as expected from a harder pattern. Endless ladder greedy
  completion: 100/100/67/58/0/0 at rounds 1/5/7/9/11/13 (was 100/92/92/42/17/0). Nonlinear:
  100/100/33/8/0 at rounds 29/34/39/44/48 (was 100/100/58/0/0). The ladder still spans
  usefully; nothing needed re-tuning.
- Two of my own tests had to be corrected while doing this, both because they measured the
  wrong quantity: `jerk` on speed magnitude reports a triangle's direction reversal as
  perfectly smooth (the speed is unchanged through the corner), and a lead-error metric
  ranked the jitter designs as hardest when in-game they were the easiest.

### This changes the task, and nothing detects it

The task hash covers configuration values, not `patterns.py`. Editing a pattern silently
changes the dynamics of every stage that uses it, and no manifest check will notice. The
trainer running at the time kept the old module in memory, so everything it trained after
episode 15,000 was trained on dynamics that no longer exist in the tree.

Measured transfer cost for the champion, old-dynamics weights evaluated under the new
patterns: round 37 holds at 87.5% completion; round 40 drops to 66.7% from 75%. So the
weights remain useful and re-adaptation is cheap -- but a run should be restarted onto the
new patterns rather than continued.

## 2026-08-18: the nonlinear run was stalled by its retention gate, not by learning

Diagnosis, from `evaluation.jsonl` rather than from intuition:

- Nine promotions took it from round 29 to round 38, but the last was at episode 10,750 and
  nothing moved for the following 4,250 episodes (previous longest gap: 2,250).
- The current round was **not** the problem: it cleared the 80% gate in 6 of the last 8
  evaluations (93.8%, 90.6%, 81.2%, 84.4% most recently).
- Promotion also required *every* prior stage to read >= 75%, each measured on **8 episodes**.
  With 37 prior stages that is a conjunction of 37 noisy tests. A stage genuinely retained at
  85% reads below 75% about 10.5% of the time on 8 samples, so all 37 pass together roughly
  1.6% of the time -- and it gets stricter with every promotion.
- Pooling the last 10 evaluations (80 episodes per stage) shows all 37 prior stages at or
  above 75%, mean 95.9%, weakest round 9 at 85%. Retention was healthy the whole time.
- The plateau controller read the resulting non-promotion as "no improvement" and cut the
  learning rate to its floor (3.75e-5 from 3e-4), with 5 restorations. So the artifact had
  started genuinely slowing learning.

Fix: `curriculum.retention_holds` replaces the per-stage conjunction in all three places it
was duplicated (`CurriculumManager`, and both champion trackers). Retention now requires the
**episode-weighted pooled** completion across prior stages to clear `retention_completion`,
plus no single stage below a new `retention_floor` (0.50) to catch genuine collapse.

Validated against the logged evaluations before relaunching: of 45 evaluations that passed
the current-stage gate, the old rule promoted on 24 and the new rule promotes on 44. It still
blocks every evaluation where the current round genuinely failed. On relaunch it promoted on
the first evaluation (stage 38 -> 39 at 84.4%).

Also fixed while relaunching:

- `reward_matches` in `curriculum.py`. The resume check compared the stored reward dict to
  the fresh one wholesale, so the two survival reward terms -- both at default 0.0 -- blocked
  resuming with "observation or curriculum manifest does not match", even though every value
  the checkpoint recorded was identical. Now stored fields must match and fields the
  checkpoint never saw must be at their defaults. Same class of bug as the task hash; this is
  the third place a schema addition broke a stored-vs-fresh comparison.
- `--learning-rate` for `train-ppo` (env `LEARNING_RATE`), applied after settings are
  resolved so a resume keeps every other setting. Note `model.learning_rate = x` alone is a
  no-op on a loaded SB3 model: the rate comes from `lr_schedule`, which `load` restores, so
  the override rebuilds the schedule and sets the live optimizer groups.
- `--stop-when-mastered` for `train-ppo` (env `STOP_WHEN_MASTERED`), so the run halts at
  round 48 instead of spending its remaining budget.

Relaunched: resumed from `checkpoint_015000` at learning rate 1e-4, 25,000-episode budget,
stopping when round 48 is mastered. Endless-ladder transfer waits for that.

### Evaluation cost is now larger than training cost

Each evaluation runs 37 prior stages x 8 episodes + 32 on the current stage = 328 episodes,
every 250 training episodes -- a ratio of 1.31 evaluation episodes per training episode, and
it grows with every promotion. Not addressed yet. The fix is to score a rotating subset of
prior stages rather than all of them, which suits pooled retention well since pooling across
evaluations recovers the sample size. Worth doing before the endless ladder, which adds 30
more stages.

---

## 2026-08-18: one mode vocabulary across every play command

- `src/asteroid_survival/modes.py` is now the single place a playable configuration comes
  from. Four modes: `arcade`, `endless`, `round N` (1-48), `survival N` (1-30).
- `play`, `showdown`, `watch`, and `compare` all take the same `[mode] [N]` arguments, via a
  new `arena` CLI subcommand and `--mode/--round` on `compare`. Answering "how do I play an
  endless round myself" is now `./run.sh play survival 12`.
- **Deleted `showdown.toml`, `showdown-ppo.toml`, and `showdown-ppo-nonlinear.toml`.**
  Lineups are generated by `modes.roster()` instead. Those files duplicated curriculum
  numbers as literals and had to be hand-synced; the nonlinear one was a copy of round 48's
  parameters. Now a showdown in any mode reads the curriculum directly, so it cannot drift.
- Transfer sources for `train-ppo-nonlinear`/`train-ppo-endless` now resolve through a new
  `best-checkpoint` CLI subcommand instead of pointing at a showdown TOML.
- `preview` takes an optional round: `./run.sh preview models/my-run 30`.
- `run.sh` honours a `PY` override, which makes the dispatch table testable without
  launching a window. Verified every path that way, and found one real bug: `watch survival
  8 20` was reading 8 as the episode count rather than the round.
- Old names kept as thin aliases: `endless`, `play-round N`, `play-endless N`.
- First scored cross-mode run, `./run.sh watch survival 3 4`: greedy 60.0s (clears), the
  phase-2 PPO champion 31.8s. The champion has never trained on survival, so this is a
  baseline for the ladder rather than a verdict on the model -- but it does confirm the
  ladder measures something the wave curriculum does not.

---

## 2026-08-18: survival rounds open with the field already populated

- `AsteroidConfig` gained `initial_asteroids`, `spawn_safe_radius`, and `spawn_safe_seconds`;
  `Simulation._populate_field` places them at reset. Survival rounds set them; wave stages
  are untouched and still open on an empty arena.
- Clearance is checked against each asteroid's **actual position at age zero**, not the
  requested origin: a pattern's lateral offset at t=0 can be as large as its amplitude, so
  origin-based checks would let rocks materialise on top of the ship. It is also checked with
  `wrapped_distance`, since a point near an edge can be close across the wrap.
- Clearance is mostly time-based (60px floor + 1.8s of the asteroid's own travel). A flat
  180px was over four seconds of warning at round 1 but under 1.5s by round 30, and measurably
  so: 8 of 24 round-30 deaths happened within three seconds. After the change that is 0-3 of
  24 across almost the whole ladder.
- Pre-population made every round harder, so the progression steps were re-tuned gentler
  (speed, amplitude, period, spawn interval, and spread all step about 20-25% less per round).
  Greedy now stalls at the 80% gate around round 7, versus round 9-10 before; 23 rounds of
  headroom remain. Rounds 20+ give greedy about 5s, so the top of the ladder is only
  meaningful for a policy substantially better than greedy.
- This was the first real test of the `_LEGACY_FIELDS` allowlist: three new hashed fields
  landed and the existing checkpoint still verified, with no edit to the snapshot table.
- The legacy-digest test is now a hardcoded golden value taken from a real checkpoint
  (`d6bf6d2a...`). The previous version enumerated fields added since and needed editing on
  every schema change -- the same maintenance trap that caused the original breakage twice.

---

## 2026-08-18: the task-hash guard broke checkpoints twice; now schema-proof

- Adding fields to `AsteroidConfig` broke `preview` and `resume` for every existing
  checkpoint, because the task hash covered the whole `asdict(GameConfig)`.
- The first fix used an **exclusion list** of the four added field names, and it was claimed
  that list could never need to grow. That was wrong. It broke again within the hour, as
  soon as the endless ladder added `survival`/`spawn_interval`/`spawn_spread` to
  `CurriculumStage` and `survival_bonus`/`round_clear` to `RewardConfig` -- neither of which
  the exclusion list covered.
- The real fix is `_LEGACY_FIELDS`, an **allowlist snapshot** of the exact field names the
  original digest covered, per dataclass. A field added anywhere in future is simply absent
  from the snapshot and cannot move the legacy digest. Do not add names to that table.
- `preview` now warns and continues on a hash mismatch instead of exiting. It is a tool for
  looking at a model, and refusing to show one is worse than showing it against a curriculum
  that has drifted -- particularly when a code change alone has produced the false alarm
  twice. `resume` stays fatal, because continuing a run on a changed task corrupts it.
- Separately, `CurriculumStage.game_config` was setting `spawn_interval` for wave stages as
  well as survival ones, overwriting the base value. It was harmless behaviourally (wave mode
  ignores it) and invisible in the hash only because the base config happened to use the same
  default. It now touches survival knobs only, so a wave stage's config is byte-identical to
  what it was before survival rounds existed.
- Two tests pin this: one drifts every post-snapshot field and asserts the legacy digest is
  unmoved while a covered field still moves it, the other asserts preview gets past the gate.

---

## 2026-08-18: endless survival ladder (built, not yet trained)

Endless difficulty became a mastery curriculum instead of an in-episode ramp:
`configs/rl-endless.toml`, thirty rounds, each a fixed difficulty, cleared by surviving
sixty seconds. This is the resolution of the cost-vs-separation tension recorded below --
episode cost is bounded by the survival target (900 decisions at every rung), difficulty is
constant inside a round, and policies rank by which round they reach rather than by seconds
in one episode.

- Almost nothing new was needed downstream. Promotion, retention, rehearsal, champion
  tracking, evaluation, and plotting all key off `completion_rate`, which is the mean of
  `completed_stage`; making that mean "survived to the decision limit" was enough.
- `EpisodeMetrics.survived_to_limit` already existed and is exactly the round-clear signal.
- `AsteroidsRLEnv` gained `completion="waves"|"survival"`, threaded from
  `CurriculumStage.completion` through `ppo.py`, `training.py`, and `preview.py`.
- `CurriculumStage` gained `survival`, `spawn_interval`, and `spawn_spread`. Survival stages
  build an interval-spawn config with `max_waves = None` and aimed headings.
- `active_cap` is deliberately NOT a per-round knob. The observation layout is sized from
  it, so varying it per round would change the observation size mid-curriculum. Asteroids
  wrap and never leave, so `spawn_interval` controls crowding instead. This is the same
  reason the wave curriculum pins `active_cap = 26`.
- Reward is survival-first: `survival_bonus` 0.02 per decision (18.0 over a full round),
  `round_clear` 5.0, death -5.0, hits 0.2/0.5/1.0, miss -0.02. `active_time_penalty` and
  `timeout_penalty` are zero because in a survival round they are exactly backwards -- one
  charges for staying alive, the other penalises clearing the round.
- Measured reward composition for a cleared round 1 (greedy, 8 seeds): hits 45.9, survival
  18.0, clear 5.0, miss -2.5. **Hits still out-earn the survival terms roughly 2:1.** That
  may be fine, since shooting is instrumentally required once the field fills, but if the
  policy starts farming split chains at the cost of position, cut `large/medium/small` or
  raise `survival_bonus` before touching anything else.
- Greedy baseline completion, twelve seeds per round: 100% at rounds 1-5, 83% at 7-9, 42% at
  11, 8% at 13, 0% from 15. Greedy stalls at the 80% promotion gate around round 9-10, so
  that is the number a learned policy has to beat, with twenty rounds of headroom above it.
- Smoke run verified end to end: a fresh PPO on CPU, 200 episodes, 4 envs, reached 10.9%
  completion on round 1 (up from 7.8% at episode 100) at ~800 decisions/s.
- No long training launched. `./run.sh train-ppo-endless 15000` transfers the best compatible
  nonlinear PPO into round 1; the phase-2 run (pid 46077) was left alone.
- `./run.sh play-endless N` plays any round 1-30 by hand. Human play of a survival round sets
  `objective.max_steps` so the round ends where training scores it.

---

## 2026-08-17: endless mode added (playable; not yet a training target)

- `./run.sh endless` plays `configs/endless.toml`: interval spawning, no waves, no step
  limit, every rock sampling one of all ten nonlinear patterns. A run ends only on a hit,
  so the only score is survival time.
- Difficulty arrives in twenty-second tiers (`ramp_step_seconds = 20`) and reaches its
  targets at tier 4 / 60s. It is constant inside a tier, which keeps the observation
  distribution stationary rather than drifting every frame. Tier 1: 39-60 px/s, one rock
  per 4.31s, cap 8, amplitude <= 50, 3.1-4.77s periods, 130-degree spread. Tier 4: 95-150
  px/s before pressure, one per 0.85s, cap 26, amplitude <= 150, 1.6-2.6s periods,
  30-degree spread.
- `ramp_seconds` is 50, not 60, and that is deliberate: progress is evaluated at the tier
  boundary below the clock, so the ramp samples 0s/20s/40s as progress 0/0.4/0.8 and tier 4
  clamps to the target. Setting it to 60 would waste a tier. Anyone retuning the tier count
  has to redo this arithmetic.
- `endless_pressure_per_minute` (0.35) keeps multiplying speed and dividing the spawn
  interval on the same twenty-second cadence after the ramp, so no policy survives forever
  and any two runs rank. Amplitude, cap, and spread deliberately stop at their tier-4
  values: cap is bound to the fixed observation layout, the other two to the arena.
- The oscillation-period window is now rampable (`wavelength_min_start`/`wavelength_max_start`)
  and `Difficulty` carries it, so spawn-time period comes from the ramp rather than the
  static config. `Difficulty.tier` is one-based, and `None` for a continuous ramp.
- `WorldSnapshot.difficulty` is populated only when `AsteroidConfig.is_ramped`; the renderer
  draws it as an orange HUD line. Wave-mode and curriculum displays are unchanged.

### What endless difficulty actually costs in training time

Measured, not assumed: twelve greedy episodes per schedule, timing the simulation loop.

| Schedule | greedy survives | decisions/episode | sim ms/episode | sim ms/decision |
|---|---:|---:|---:|---:|
| 600s continuous | 217.0s | 3255 | 1245 | 0.38 |
| 120s ramp, 20s tiers | 70.2s | 1054 | 370 | 0.35 |
| 50s ramp, 20s tiers (current) | 33.6s | 504 | 139 | 0.28 |

- Per *episode* the current schedule is ~9x cheaper, which is why raising difficulty felt
  like it sped training up: `--episodes` budgets by episode, so a harder game finishes a
  nominal budget far sooner.
- Per *decision* -- the unit learning actually consumes -- it is only ~1.4x cheaper. Almost
  all of the 9x is buying less experience, not cheaper experience. 15,000 episodes is 48.8M
  decisions on the old schedule and 7.7M on the current one.
- Simulation is a minority of training cost anyway. The live PPO run reports 390 decisions/s
  end to end (2.6 ms/decision) against ~0.3 ms/decision of simulation, so difficulty moves
  total training time by a few percent at a fixed decision budget. (Caveat: that run uses 8
  parallel envs, so the split is approximate.)
- A simulated second is 3.7x more expensive at tier 4 than tier 1 (11.0 vs 3.0 ms, 26 vs 2.4
  rocks on screen), so difficulty partly pays for itself in the wrong direction.
- Conclusion: `max_decisions` (already wired through the curriculum as `max_seconds * 15`)
  and `no_hit_seconds` are the correct cost levers, because they cap episode length without
  flattening the difficulty curve. Difficulty should be chosen for how well it separates
  policies. Using it for both jobs is what collapsed the greedy-vs-idle gap from 33.4s to
  8.7s.

### Task hashing now ignores unset config fields

- Adding the four endless-mode knobs to `AsteroidConfig` broke `preview` and `resume` for
  every previously trained checkpoint, including the live nonlinear run: the task hash was
  a sha256 over the full `asdict(GameConfig)` of every stage, so a new field with a default
  changed it even though no curriculum used the field.
- `task_hash` moved into `curriculum.py` (it was duplicated across `preview.py`, `ppo.py`,
  and `training.py`) and now hashes only fields that differ from their dataclass default.
  A future unused option can no longer invalidate anything.
- `task_hash_matches` also accepts the pre-existing hash, reproduced by excluding the four
  added fields via `_LEGACY_EXCLUSIONS`. That list is frozen and must never grow -- new
  fields do not need to be added to it, because the canonical hash now ignores unset fields.
- Both resume paths compare through `task_hash_matches` instead of exact layout equality,
  so a run started before this change still resumes.
- Note the trainer process loads its code at start, so the run launched on 08-17 keeps
  writing legacy-hash checkpoints until it is restarted; the compatibility path covers them.

### Pacing was tuned in four passes, each re-measured over fixed seeds

| Schedule | do nothing | random | greedy |
|---|---:|---:|---:|
| 600s continuous ramp | 34.5s | 23.4s | 217.0s |
| 420s continuous ramp, pressure 0.07 | 33.9s | 19.9s | 149.8s |
| 120s ramp, 20s tiers, pressure 0.35 | 36.6s | 18.1s | 70.0s |
| 50s ramp, 20s tiers, pressure 0.35 (current) | 25.6s | 18.1s | 34.3s |

- The last pass shifted the whole schedule thirty seconds earlier by request: what had been
  the 30s difficulty became tier 1, and what had been the 60s difficulty became tier 2. The
  two anchors are exact, not approximate.
- **The current schedule compresses the baseline spread badly.** Greedy outlives doing
  nothing by 8.7s, against 33.4s one pass earlier. Survival time may therefore be a weak
  ranking signal for weak-to-middling policies. The first thirty seconds are fixed by
  request, so the lever for restoring headroom is `endless_pressure_per_minute`, which only
  governs tier 4 onward. Check this before trusting endless survival as a training metric.
- Not wired into training yet. The remaining question is the reward/termination shape for a
  pure-survival endless task; episode length is no longer a blocker, since a greedy-grade
  policy resolves in about 35s of simulated time.

## 2026-08-17: feed-forward PPO reached the final curriculum block

- `models/ppo-ff-0817-1739` mastered all 28 foundation stages by the episode-20,500
  evaluation. The final-stage promotion API reports `promoted = false` because there is no
  stage 29, but `curriculum_state.json` records `mastered = true`; that is the authoritative
  completion signal. The remaining scheduled episodes are retention rehearsal.
- The protected champion at that point was episode 20,250 on stage 28. Training was still
  running when inspected, so use `./run.sh status` for the live episode count.
- The original 1,000-episode architecture screen favored LSTM-PPO slightly on stage 1
  (71.9% completion versus feed-forward PPO's 65.6%), but the long feed-forward run then
  demonstrated that recurrence is not required to traverse nearly the entire curriculum.
- The curriculum now has 28 one-wave stages: ten stationary size-ladder stages, three safe
  linear, three lethal linear, three turret sine, three mobile sine, three mobile arc, and
  three mobile S-curve stages. All current gates use 80% completion, 5% accuracy, and 75%
  retention across prior stages.
- Live showdown now accepts PPO and recurrent PPO through `./run.sh showdown ppo`; plain
  `./run.sh showdown` remains MuZero. The live adapter reconstructs the saved observation
  history and acts at the same four-frame cadence used in training.
- The old nonlinear envelope was too subtle (up to 30–45 px/s and 25–50 px amplitude).
  Added rounds 29–48 through `configs/rl-nonlinear.toml`. Every added round samples all ten
  nonlinear patterns; speed, amplitude, oscillation rate, time limits, and starting count
  increase by small explicit increments. Count adds one large rock every two rounds until
  reaching 11. Round 48 uses 11 large rocks at 49–64 px/s, 82–145 px amplitude, and
  2.62–3.74 second periods.
- `./run.sh train-ppo-nonlinear 15000` transfers the best feed-forward policy into a new run
  at round 29 while resetting optimizer/task progress. The extended config inherits the
  original 28 definitions, while the original mastered run keeps its old hash/resumability.
- `./run.sh showdown ppo-nonlinear` exposes the round-48 nonlinear target immediately for
  visual testing; the foundation PPO may struggle there until phase-two training completes.
- `./run.sh play-round N` lets a human play any exact round 1–48 using the same generated
  configuration as training, which makes subjective difficulty checks reproducible by seed.
- `README.md` now records the exact reward equation, all 28 stage compositions and limits,
  arcade wave sizes, and recurring checkpoint/resume questions. `COMMANDS.md` separates PPO
  resume from the MuZero-only `continue`/`finish` helpers.

### Current next steps

1. Let the already-mastered foundation job reach its requested episode budget or stop it, then
   launch `./run.sh train-ppo-nonlinear 15000`; never run both trainers concurrently.
2. Run a controlled `compare` on fixed seeds; use showdown only as a visual demo because
   its three ships share kills and asteroid state.
3. Train the LSTM variant beyond the 1,000-episode screen only if a direct architecture
   comparison is still useful; feed-forward PPO is already a viable full-curriculum model.
4. Re-measure the greedy baseline against the final 28-stage curriculum before claiming the
   learned model wins or loses overall.

The entries below are historical research notes. Their quoted stage counts, run status, and
reward values describe the experiments at that time; use the current sections in `README.md`
and `configs/rl-curriculum.toml` for today's behavior.

---

## 2026-08-16: champion/latest training was made recoverable

- Episode 39,250 passed the stage-4 gate (81.25% completion, 0.1436 accuracy) but the old
  completion-first champion comparison rejected it, reset its promotion streak, deleted replay,
  and overwrote the passing checkpoint. This directly identified the rollback controller as a
  blocker rather than a hypothetical concern.
- The newest checkpoint is always the resumable learner. `champion/` remains a protected
  evaluated copy for preview. Current plateau/recovery handling may restore champion policy
  weights, but it retains logs, curriculum state, and replay rather than deleting the run.
- Champion selection is aligned with both curriculum gates. Gate-passing candidates outrank
  candidates that merely set a higher completion high-water mark.
- Replay transitions now record curriculum stage. Storage reserves half its capacity for prior
  stages, and batches draw 60% current-stage data with the rest balanced across older stages.
- Repeated retention failure starts a focused rehearsal window on the weakest old stage.
  Current code can restore champion weights and lower the learning rate, but does not delete
  replay, logs, or curriculum progress.

---

## 2026-08-14: arcade-first curriculum implemented

- Play now uses movement, random linear headings, slower clear-before-next-wave rounds, and a 26-rock cap.
- Training now has five mastery-gated stages, frozen per-stage evaluation, prior-stage rehearsal, and 8 dense + 8 sparse history samples.
- Reward now separates size hits, wave clear, speed, accuracy, misses, active time, and death.
- Added mobile velocity observations, learned terminal discounts, semantic checkpoint manifests, legacy checkpoint loading, and dependency-free SVG graphs.
- Verified a two-environment MuZero smoke run and the 74-test suite.

## Historical baseline before the current curriculum

**The agent has never beaten the greedy baseline.** That is the headline. Everything else
is infrastructure built to find out why.

| | do nothing | greedy baseline | best trained model |
|---|---|---|---|
| `rl.toml` (constant) | 6.18s | **8.33s** | 7.27s (v2, 10k episodes) |
| `rl-ramp.toml` | 12.72s | **29.55s** | not yet run to completion |
| `rl-wave.toml` | 23.8s | **57.8s**, wave 2.7 | not yet run |

A uniformly random policy scores 7.45s on `rl.toml`. Two full 10,000-episode runs (v2, v3)
landed at or below that. The agent learned essentially nothing about survival, though
accuracy did creep from 0.17 to ~0.24.

### 2026-08-15: reward was non-monotonic — fixed

The curriculum reward punished the agent for playing at its own skill level. Computed
payoffs for stage 1 (one large rock, 7 bodies, 60s):

| strategy | old reward | new reward |
|---|---|---|
| do nothing | **-2.20** | -9.20 |
| agent's actual skill (acc 0.11) | **-3.76** | **+8.45** |
| decent (acc 0.30) | +8.74 | +14.08 |
| greedy (acc 0.68) | +13.42 | +17.18 |

Idling beat playing, so the agent correctly learned to idle: v3 collapsed to 0% completion
and a 100% timeout rate. The fix makes reward strictly increasing in accuracy:
all hits pay `1.0` (a first large-rock hit paid `0.2` while a miss cost `0.28`),
`accuracy 2.0 -> 4.0`, `miss_penalty 0.28 -> 0.10`, `timeout_penalty 1.0 -> 8.0`.

**Whenever the reward changes, check it is monotonic in skill before launching.** Cheap to
compute, and two runs were lost to not doing it.

All 21 stages are now verified passable: greedy scores 100% completion on every one, clears
every accuracy gate (worst margin +0.079), and earns +13.9 to +30.1 reward.

### 2026-08-16: training seeds were on a collision course with the eval set

Training seeds advance by one per episode from `--seed` (default 0); the held-out evaluation
uses frozen seeds `10000-10031`. Any run longer than ~10,000 episodes therefore starts
training on the exact levels it is scored against, quietly corrupting the only trustworthy
metric in the project.

Fixed with `training_seed()` in `rl/training.py`, which skips the reserved band, so held-out
numbers stay honest at any run length. Two tests pin it, including one that walks 40,000
episodes past the band.

### Historical run that was in flight

- **`models/curriculum-v4`** — running. **Cleared stage 1 at episode 2,500** (held-out 93.8%
  completion, accuracy 0.395 against a 0.30 gate) — the first curriculum gate any model in
  this project has passed. Accuracy rose 0.011 -> 0.395 over 2,500 episodes.
  Stage 2 (two large rocks) is slower: completion drifting 28% -> 41% but accuracy has gone
  *backwards*, 0.122 -> 0.085 over 2,000 episodes. If accuracy is still under 0.15 by episode
  7,000, revisit the stage-2 gate.
  It predates the seed fix, so a guard stops it at 9,500 episodes, before its seeds reach the
  evaluation band. Restart from a checkpoint after that to continue cleanly.

### Next steps recorded at that time

1. Retrain with `--history-long-frames 8 --history-long-stride 8`. The current run used
   history deltas that were not wrap-aware, so its history feature was partly noise.
2. Run a real training job on `rl-wave.toml` (nothing has trained on waves yet).
3. Re-measure baselines whenever the task changes — every task change so far has
   invalidated them.

---

## Open questions

- ~~Does long-term memory help?~~ **Resolved: yes.** See "Long-term memory wins" below.
- ~~Does the tail get worse with wider memory?~~ **No** — that was the wrap bug.
- ~~Is a stationary turret the right task at all?~~ **Resolved:** stages 20–28 enable
  movement, and the live arcade game is mobile. Earlier turret stages remain useful for
  isolating aiming before adding thrust.
- **`configs/default.toml` duplicates `showdown.toml`** (both are human + greedy + muzero).
  Left as-is deliberately. `configs/solo.toml` exists for genuine single-player.

---

## What was learned, with numbers

These are the findings worth keeping. Each was measured, not assumed.

### The task was nearly unlearnable

At `rl.toml`'s constant difficulty, the gap between doing nothing (6.18s) and playing well
(8.33s) is **2.15s**, while seed-to-seed standard deviation is **~2.5s**. The noise was
larger than the entire skill range, so ~90% of the reward was paid out regardless of what
the agent did. This is the best explanation for two flat 10,000-episode runs.

Ramping difficulty inside each episode widened that gap to 16.8s (`rl-ramp`) and 34.0s
(`rl-wave`). Ramping *within* an episode rather than between episodes keeps the task
stationary, so the target does not move under the learner.

### Every asteroid was aimed at the ship

Measured while tuning waves: **100% of spawns headed within 20° of the arena centre**
(median 5.4°), where the stationary ship sits. Nothing ever drifted harmlessly past. Real
Asteroids scatters them, which is why waves are survivable there. Adding `spawn_spread`
took greedy from 17.1s to 57.8s. It is now a ramped difficulty knob: wide (arcade-like)
early, narrowing as a run goes on.

### The agent could not see its own weapon cooldown

`ShipSnapshot` had no cooldown field, so `FIRE` was a no-op for ~3.6 consecutive decisions
with no way to know. Fixed by exposing it in the observation.

### Aiming needs bearing, not world coordinates

The observation gave asteroid positions in world frame with `x` and `y` divided by
*different* constants (`width/2` vs `height/2`), which skews every angle. The network had
to undo that distortion before any rotation it learned was valid. Now each asteroid carries
`sin`/`cos` of its bearing relative to the ship's heading, plus closing and tangential
speed, and positions are scaled isotropically.

This did **not** fix learning — run v3 with bearing features tracked v2 almost exactly
(accuracy 0.204 vs 0.203 at episode 3000). Hypothesis falsified.

### Asteroid paths are unpredictable from one frame

Paths are deterministic functions of `(pattern, t, amplitude, frequency, phase)`, none of
which appear in a snapshot. Predicting 0.5s ahead from a single frame lands a median 33px
from the truth, against asteroid radii of 13/24/39px — leading a shot is impossible.

With history, a learned predictor reaches 13px median at 12 frames, below a small
asteroid's radius. Accuracy improves steeply to ~8 frames and plateaus by 12; 16 and 24 are
slightly worse.

Default memory is **8 frames = 0.53s**, covering only 12–31% of one oscillation
(wavelengths are 1.7–4.5s). That motivated the two-tier memory, now validated below.

### Loss going up is not a bug

MuZero's targets are non-stationary, so total loss rises during healthy training. In the v2
run the rise was entirely `policy_loss` and `consistency_loss`; `value_loss` stayed flat at
0.08 and `reward_loss` at 0.0001 across 10,000 episodes. Divergence would have shown up in
value loss. **Read `evaluation.jsonl`, not loss.**

### Long-term memory wins, once wrapping is handled

Predicting 0.5s ahead, all schemes trained to convergence and calibrated against analytic
linear extrapolation (18.3px median) on the same held-out split:

| scheme | slots | span | median | p90 |
|---|---|---|---|---|
| 8 dense (shipped default) | 8 | 0.47s | 5.0px | 23.4px |
| 16 dense | 16 | 1.00s | 4.3px | 24.8px |
| **8 dense + 8 stride 8** | 16 | **4.73s** | **3.9px** | **15.6px** |

Two-tier memory wins on both the median *and* the tail, at the same slot count. The tail is
the bigger win: p90 drops 37% against dense-16. Enable with
`--history-long-frames 8 --history-long-stride 8`.

### A wrapping bug hid all of this

`encode_observation` computed history deltas by raw subtraction while every other position
in the file used `wrapped_delta`. An asteroid crossing the screen edge therefore produced a
~900px jump, when the physical maximum over one decision is ~73px. **8.6% of samples were
corrupted this way.**

Fixing it improved prediction roughly tenfold (48.6px → 5.0px median for the shipped 8-frame
default) and reversed the apparent conclusion that wide memory hurts the tail. The training
run started 2026-08-14 (`models/muzero-0814-0100`) used the buggy feature throughout.

### Probes must be calibrated or they lie

Three prediction probes produced confident, wrong answers. Each scored *worse than analytic
linear extrapolation*, which a 512-wide MLP should beat trivially. I first blamed underfitting
and threw more optimisation steps at it; the real cause was the wrapping bug above, whose
outliers dominated the MSE objective. Any future probe must report an analytic baseline on
the same held-out split and be discarded if the learned model loses to it — that guard is
what eventually surfaced the bug.

Related: an early conclusion that "history does not help" came from testing *quadratic*
extrapolation, which cannot represent a sine. That was a strawman and the conclusion was
wrong.

---

## Changes made

### Performance

The original trainer ran at **78s per episode**. It is now **~0.4s** — about 200x.

- **JIT-compiled the search.** `mctx.gumbel_muzero_policy` was being traced from scratch on
  every decision (~110 times per episode), and `recurrent_fn` was a closure defined inside
  `search()`, so even adding `@jax.jit` would have re-traced constantly. Hoisted it to a
  method and cached a compiled program per simulation budget: **675ms → 0.53ms per call**.
- **JIT-compiled the training step**: 115ms → 6.2ms.
- **Batched self-play** (`--parallel-envs`): one compiled search serves N environments.
  Per-environment search cost drops ~3.8x at N=48. End-to-end gain is only ~20% because
  gradient updates now dominate and their count does not change with N. 16 is the default.

The simulator itself was never the bottleneck: 0.06ms per step.

### Correctness

- **Replay buffer was discarded on every `--resume`**, so each chunk cold-started with an
  empty buffer. Now saved to `replay.npz` and restored, with shape checks that reject a
  stale buffer rather than misreading it.
- **`MuZeroController` had a stale observation check** (`5 + max_asteroids * 8`) that would
  have rejected every current checkpoint. It now derives the layout from the checkpoint.
- **`_latest_checkpoint()` hardcoded `models/muzero/`** — the oldest, incompatible run —
  and ignored newer directories. Now searches all of them, newest first, skipping any whose
  observation layout does not fit the config.
- **Dense and dilated history produce identical observation sizes**, so auto-detection
  would have silently mislabeled one as the other and fed the model scrambled history with
  no error. Checkpoints now record `observation_layout` in `metadata.json`.

### Learning

- 5-step dynamics unroll (was 1) with MuZero's 0.5 gradient scaling on the recurrent path,
  masked so sequences never unroll across an episode boundary.
- n-step bootstrapped value targets (was full Monte-Carlo over ~110 steps).
- Scalar value transform, so value loss stops drowning out policy loss.
- `--shot-penalty`: firing was free, so nothing pushed the agent toward trigger discipline.

### Environment

- Weapon cooldown, bearing features, isotropic scaling (above).
- Per-asteroid history keyed by **asteroid id**, not stacked observations — slots are
  re-sorted by distance every step, so naive stacking compares different asteroids between
  frames.
- Two-tier memory: `--history-long-frames` / `--history-long-stride` add sparsely sampled
  older positions. 8 dense + 8 strided spans 4.73s for the same 16 slots that dense-16
  spends on 1.07s. **Validated** (see above); off by default.
- History deltas are now wrap-aware. They were not, and 8.6% of them were garbage.
- In-episode difficulty ramp for every knob (`spawn_interval`, `active_cap`, speeds,
  `amplitude_max`, `spawn_spread`, `wave_threshold`) via `<field>_start` values. Unset means
  constant, so old configs are untouched.
- Arcade wave spawning (`spawn_mode = "wave"`): a wave arrives only once the field is worn
  down to `wave_threshold`, then trickles in one asteroid at a time. Sizes follow the
  original game — 4, then +2 per wave, capped at 11.
- Square arena, 900x900 (was 1280x720). Cost ~8% survival; the skill *ratio* was unchanged.

### Tooling

- `./run.sh` — `play`, `showdown`, `compare`, `watch`, `train`, `status`, `baseline`,
  `test`. Finds the newest compatible checkpoint and infers its history layout, so commands
  cannot fail from a forgotten flag.
- `compare` — you, greedy, and a checkpoint over identical seeds, scored through the same
  env. Human input is sampled once per *decision*, matching the agents' control rate rather
  than giving a human 4x the reactions.
- `showdown` — all three in one shared arena, with a scoreboard that freezes each ship's
  time as it dies. Note this is a live side-by-side, not a controlled measurement: all
  three share one asteroid field, so kills help everyone.
- Block-averaged training logs and held-out `evaluation.jsonl`.
- `COMMANDS.md`, README sections, 67 tests (there were 3 covering the RL code, and none
  covering the agent, at the start).

---

## Files

| path | what |
|---|---|
| `run.sh` | task launcher, start here |
| `COMMANDS.md` | every command, copy-pasteable |
| `configs/rl-curriculum.toml` | source of truth for all 28 stages and reward values |
| `configs/rl-nonlinear.toml` | extension defining slowly scaled rounds 29–48 |
| `configs/solo.toml` | just you |
| `configs/showdown.toml` | you + greedy + muzero, one arena |
| `configs/showdown-ppo.toml` | you + greedy + PPO/LSTM-PPO, one arena |
| `configs/showdown-ppo-nonlinear.toml` | same PPO lineup against strong mixed curves |
| `configs/rl.toml` | constant hard difficulty, the benchmark |
| `configs/rl-ramp.toml` | legacy ramped-difficulty preset |
| `configs/rl-wave.toml` | legacy arcade-wave experiment preset |
| `models/*/evaluation.jsonl` | held-out scores — the real progress signal |
| `models/*/training.jsonl` | per-episode records |
| `models/*/curriculum_state.json` | learner's current zero-based stage and promotion history |
| `models/*/champion_state.json` | protected champion episode, score, and recovery counters |
