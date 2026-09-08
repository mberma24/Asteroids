# Fragment-warning v21 — VM-only, 2026-09-08

## What the diagnostic found

Preserved v20 episode-31500 champion; round 29, fresh seeds 1000001200–1000001263.
No action masking, reward changes, or training during these evaluation episodes.
All projectile-to-parent joins use actual physics collision ordering, not predicted target IDs.

- 41/64 clears; 23 deaths; 22 deaths traced to self-created fragments.
- Legacy straight-ahead observation correctly targeted and flagged 10/22 fatal shots.
- Correcting flight-time drift and split-cap accounting alone flagged 11/22.
- Legacy prediction conditioned on the actual firing action flagged 13/22.
- Corrected, action-conditioned prediction flagged 14/22.
- Across 6,449 fired shots, the corresponding total warnings were 89, 102, 100, 112.
  These totals are NOT false-positive rates: a warning can be followed by successful evasion.

The main observed gap is action conditioning, not merely a shorter warning horizon.
Eight fatal shots still lacked a correct danger flag. Future controls, target curvature,
other projectiles and spawns remain outside this approximation. Longer training or
more warnings alone are not evidence of improved performance.

## Implementation

Observation v11 appends 32 inputs, four for each firing action in Action enumeration order:
predicted hit, predicted split, signed minimum clearance /150 (clipped), and danger time
divided by projectile lifetime +1 second. Each prediction rolls turn/thrust through the
weapon cooldown, then assumes coasting. The fragment horizon remains one second.

Opt-in corrected physics includes ship drift during projectile flight, drag-before-movement,
and the slot freed when destroying a parent at the active cap. Legacy v8–v10 predictions
and the entire v10 observation prefix remain unchanged. Target intersection is still linear;
these features are predictions, not guaranteed counterfactual outcomes.

The separate curriculum only changes observation version. Stages, rewards, orbit schedule,
promotion rules and task hash are unchanged. All policy transfer input columns start at zero.
PPO settings are explicitly preserved (including n_steps=256 and gamma=.997).

## Verification

- 270 passed, 21 skipped, 1 deselected in the full available VM suite.
- The excluded test imports absent Flax through the existing live-play/MuZero controller.
- 16 paired complete episodes / 5,074 decisions: identical actions and outcomes before learning.
- Maximum action-probability difference: 0.
- Single-worker sequential rollout benchmark: 16.93 seconds old, 20.68 seconds new,
  approximately 18% lower throughput. This is a small benchmark with training running concurrently.
- Eight-episode ordinary PPO smoke test and two-episode checkpoint-resume test.
  These are wiring/update checks, not held-out evidence of better clear rates.
- Launch gate checks finite parameters and nonzero learned weights on both new input layers.

## Operations and rollback

VM worktree: /home/ubuntu/Asteroids-v21
Ordinary run: models/oracle-survival-v3-v21-action-fire
Service: asteroids.service (normal restart/reboot behavior, no experiment auto-pause)
Log: /home/ubuntu/Asteroids-v21/cloud-train.log
Evidence and preserved checkpoints: /home/ubuntu/asteroids-experiments/fragment-warning-v21

The new run starts from the behavior-preserving widened champion, not the smoke learner.
Its protected incumbent is primed with the prior independent 256-seed current-round
clear rate of 67.1875%, after verifying the tested source checkpoint checksum matches.
The entire v20 run remains intact and an extra latest-complete-learner backup is retained.

Explicit rollback, if needed:
sudo systemctl stop asteroids.service
sudo install -m 644 /home/ubuntu/asteroids-experiments/fragment-warning-v21/v20-bridge.conf /etc/systemd/system/asteroids.service.d/bridge.conf
sudo systemctl daemon-reload
sudo systemctl start asteroids.service

Next assessment should compare the source and learned champion on matched fresh current-round
and full-orbit seeds, with earlier-round retention. Training clear rates or these diagnostic
seeds are not an independent success test. No claim of improved gameplay has yet been established.
