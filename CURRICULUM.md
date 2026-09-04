# Survival round reference

This documents **`configs/rl-survival-v2.toml`**. For the newer ladder, whose rounds 28-96
grow the pattern instead of the speed, see "Survival v3" at the bottom. Do not confuse it with
`configs/rl-endless.toml`, documented under "Endless survival ladder" in `README.md`:
that one is also 96 rounds but flies straight until round 39, whereas this one introduces
sine at round 3.

All 96 rounds last 30 seconds. The authoritative per-round table is generated from the same
config the trainer uses:

```bash
./run.sh rounds survival-v2
```

There are no hidden hand-maintained wave definitions. Survival rounds use continuous side
spawning, so the relevant “wave” is the current round's opening count plus spawn interval.

Straight is a trajectory like any other: each phase sets its probability to
`1/(len(patterns) + 1)`, so a rock is exactly as likely to fly straight as to fly any one
curve. The share falls only because the pool grows.

| Rounds | Movement pool | Straight probability | Spawn size |
|---|---|---:|---|
| 1–2 | linear foundation | 100% | small |
| 3–6 | sine | 50% | small |
| 7–10 | sine, arc, S-curve | 25% | small |
| 11–12 | smooth three | 25% | 75% small / 25% medium |
| 13–14 | smooth three | 25% | 50% small / 50% medium |
| 15–16 | smooth three | 25% | 25% small / 75% medium |
| 17–22 | smooth three plus zigzag, sawtooth, lane-change | 14.3% | medium |
| 23–25 | all eleven nonlinear patterns | 8.3% | 75% medium / 25% large |
| 26–28 | all eleven nonlinear patterns | 8.3% | 50% medium / 50% large |
| 29–52 | all eleven nonlinear patterns | 8.3% | large |
| 53–82 | all eleven nonlinear patterns | 8.3% | 75% large / 25% medium |
| 83–96 | all eleven nonlinear patterns | 8.3% | random mixed size |

Every numeric control changes linearly each round from round 1 to round 96. (Survival v3
breaks this into two segments; see below.)

| Control | Round 1 | Round 96 |
|---|---:|---:|
| Asteroid base speed | 26–38 | 126–196 |
| Pattern amplitude | 0–0 | 12–160 |
| Pattern wavelength/period control | 4.8–6.0 | 1.4–2.4 |
| Spawn interval | 2.00 s | 0.75 s |
| Spawn spread | 170° | 24° |
| Opening asteroids | 8 | 20 |

Mastery requires mean survival fraction of at least 90%, full-round clear rate of at least
80%, and accuracy of at least 5%, twice within the latest four evaluations. Retention uses a
75% pooled survival target and a 50% per-round floor.

The reward baseline is `0.10` per complete decision survived (time-prorated on a partial
fatal decision), `+10` for reaching 30 seconds alive, `-5` on death, split-neutral asteroid
rewards of `0.60/0.30/0.15`, and `-0.02` per miss. The death-10 and safety-potential TOMLs are
ablation configs, not silent baseline changes.

## Rounds 97-100: the overfitting check

`configs/rl-survival-v2-varied.toml` extends the ladder with four more rounds. It is a
separate file on purpose -- appending rounds to `rl-survival-v2.toml` would change its task
hash and strand every checkpoint trained against it. Rounds 1-96 are inherited unchanged and
stay in the retention set.

Everything below round 97 moves one way: by round 96 every rock is fast, swings wide, and
oscillates quickly. A policy can pass that by *assuming* those properties instead of reading
each rock -- lead by a fixed amount, expect a turn, never wait. These rounds keep round 96's
difficulty as the default draw and mix a minority of slow rocks back in.

| Round | Varied share | Everything else |
|---:|---:|---|
| 97 | 25% | round 96's envelope, still tightening each round |
| 98 | 30% | |
| 99 | 35% | |
| 100 | 40% | |

`variety_probability` scales one rock's speed, amplitude, and period **together**, so a
varied rock is coherently sluggish rather than an incoherent blend of fast travel and a lazy
wobble. The scale is drawn from 0.25 to 1.0.

Measured over 1,000 spawns, round 97 against round 96:

| | round 96 | round 97 |
|---|---:|---:|
| median speed | 171.7 | 150.1 |
| 10th-percentile speed | 137.1 | 67.5 |
| slowest rock | 126.0 | 37.1 |
| rocks below the round's speed floor | 0% | 28% |
| longest period | 2.40s | 9.15s |

Play or score them with the `varied` mode:

```bash
./run.sh play varied 97
./run.sh watch varied 100 20
```

A policy that has genuinely learned to read asteroids should lose only a little here. One
that has memorised the late-game distribution will fall off a cliff -- which is the number
this check exists to produce.


---

## Survival v3: expressive patterns from round 28

`configs/rl-survival-v3.toml`, played and scored as the `survival-v3` mode. Rounds 1-27 are
`rl-survival-v2-detfrag.toml`'s, unchanged and inherited via `extends`; rounds 28-96 are a
second linear segment whose `_start` values are v2's round-28 values to five decimals, so
there is no step at the seam.

**Why it exists.** v2 raises late difficulty mostly by making rocks faster and making them
oscillate faster. Measured on the simulator, that does not read as shaped motion: by round 96
a rock's sideways swing speed is 1.77x its forward speed on a 1.9-second period, so it
vibrates across the arena rather than tracing a path. At round 27, where training actually
sits, the mean swing is 24px, about one medium asteroid across.

**What changes.** Three slopes, and nothing else. Spawn interval, spread and opening count
keep v2's slopes and land on v2's round-96 values.

| Control | v2 round 28 -> 96 | v3 round 28 -> 96 |
|---|---|---|
| Asteroid base speed | 54-83 -> 126-196 | 54-83 -> **90-140** (half the growth) |
| Pattern amplitude | 3-45 -> 12-160 | 3-45 -> **56-280** (about twice) |
| Pattern period | 3.83-4.98s -> 1.4-2.4s | 3.83-4.98s -> **5.40-6.60s** (grows, not shrinks) |

**Measured, ten episodes per round.** "0.5s error" is how far a rock lands from where its
current velocity says it will be half a second later, which is the horizon the agent acts on.
Reproduce with `python scripts/pattern_expression.py configs/rl-survival-v3.toml 28,52,96`.

| round | mean swing | 0.5s error, p90 |
|---:|---:|---:|
| 28 (both ladders) | 25 px | 11 px |
| 52 v2 / v3 | 47 / **77** px | 28 / **24** px |
| 96 v2 / v3 | 84 / **168** px | 144 / **33** px |

**The trade, stated plainly.** v3's rocks trace roughly twice the shape, and its top rounds
are markedly *more* predictable than v2's -- round 96 sits near v2's round 52 on the error
measure. Motion difficulty also flattens above about round 64, because amplitude and period
grow together and their ratio is what sets curvature. Above that point the added difficulty
is rock size, count and spawn rate, which v3 does not change. That was the deliberate choice:
a rock that traces one big arc beats a rock that buzzes.

**Measured ceiling.** The blind planning oracle -- lookahead over what is on the field, no
clairvoyance about spawns or fragments -- over 16 seeds a round:

| round | v2 ladder | v3 ladder |
|---:|---:|---:|
| 52 | 0.500 | 0.812 |
| 73 | -- | 0.812 |
| 96 | 0.188 | 0.750 |

v2's late rounds sit far below its own 0.75 promotion gate, so they were unpassable by any
reactive policy rather than merely hard. That, more than how the motion reads, is the reason
to prefer v3. Its round 96 at 0.750 is exactly on the gate, so the very top is still
marginal.

**Observation v10, and why `orbit` needed it.** Orbit cost 15-19 points of clear rate at
round 29 at every rate and reach tried, from 57 to 249 px/s of tangential speed, so its
difficulty is not a knob. Measured on round 29, a half-second straight-line prediction is off
by a median 58px for an orbiting rock against 0-7px for every other pattern -- more than a
large asteroid and the ship put together -- because the threat features extrapolate velocity
in a straight line while a circling rock's heading rotates continuously. Extrapolating along
an arc instead cuts that to 18px.

v10 appends that arc-aware threat block rather than replacing the straight-line one, because
an arc guess is *worse* on the patterns that turn in corners: zigzag's p90 goes 11.9 to 16.5px
and serpentine's 45.3 to 53.7. Both readings stay available, and the block's last input is the
gap between them, which is near zero for a rock flying straight and large for one circling.
Rocks turning slower than 0.2 rad/s take the closed form instead of an eight-step march, which
keeps the cost near 5%.

**Observation v9.** v3 runs past two clamps in the v5 difficulty block -- `amplitude_max/200`
pins from round 73 and `wavelength_max/6` from round 71 -- which would make its last
twenty-odd rounds read as one identical round. v9 appends rescaled copies (`/300` and `/8`)
rather than editing the v5 inputs, so existing weights keep their meaning and a v8 checkpoint
widens in at zero weight. `wavelength_min` is deliberately not duplicated: it tops out at
5.4s and never reaches its own clamp.


---

## What each pattern actually is

Plain-language labels, for reading rather than for anything the agent sees. Watch any of them
with `./run.sh patterns survival-v3 52` and the number keys.

| key | pattern | what it does |
|---|---|---|
| 1 | `sine` | an even side-to-side wave |
| 2 | `zigzag` | the same wave, but with sharp corners instead of curves |
| 3 | `sawtooth` | a long drift one way, then a quick run back, at uneven intervals |
| 4 | `s_curve` | a lazy S: sweeps across, then dwells at each side |
| 5 | `serpentine` | full-width sweeps with sharp corners, timed so they never repeat |
| 6 | `lane_change` | parks in one lane, then snaps across into another |
| 7 | `arc` | one wide continuous turn, a slow circle |
| 8 | `corkscrew` | tight repeated loops that curl back on themselves |
| 9 | `figure_eight` | a real figure eight, crossing its own path |
| 0 | `spiral` | loops that widen from nothing out to full size |
| - | `brownian` | an aimless wander, no shape and no rhythm |
| = | `tumble` | never repeats and never settles: always turning, never the same turn |
| \\ | `orbit` | circles a centre point that travels with it, looping as it crosses |
| L | linear | dead straight, no pattern at all |

`tumble` is new, **not yet in any curriculum, and deliberately outside the default pool**.
`pattern_pool` is part of the task hash and defaults to `PATTERN_NAMES`, so appending to that
tuple restates the task of every ladder that names no patterns of its own and strands its
checkpoints. It lives in `EXPERIMENTAL_PATTERNS` instead; a ladder opts in by naming it. It is the only pattern here that is not
built from sines. The other ten curved shapes are sums or warps of trigonometric terms, which
makes them smooth and quasi-periodic; `tumble` is value noise, a repeatable random number per
cell of time, smoothly interpolated and layered over four octaves whose rates sit at an
irrational ratio so they never line up. The per-cell values come from an integer hash rather
than from `sin` of a large argument, so a seed means bit-for-bit the same episode on the
training box and on a laptop.

The slowest octave carries most of the weight, and a gain with a `tanh` limit lifts the
typical swing, because a pattern that shivers in place is the easiest thing in the pool to
survive. Measured: it sits a median 68% of its amplitude off the centreline against a sine's
64%, reaches 96% of amplitude, and changes direction 0.90 times a second against a sine's
0.50. Correlating one ten-minute window against the next gives 0.07, where a sine gives 1.00,
so it genuinely does not repeat. It stays inside its amplitude (47.9 of a 51 bound), peaks at
1.44x `amplitude x frequency` against the 3.0 ceiling, and its closest resemblance to any
existing pattern is 0.437 against a 0.75 near-duplicate limit.
