# Survival v2 round reference

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

Every numeric control changes linearly each round from round 1 to round 96:

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
