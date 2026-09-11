"""Prime the ordinary v21 run with the measured, behavior-preserving incumbent."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, "src")
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.ppo_support import PPOChampionTracker

ROOT = Path("/home/ubuntu/Asteroids/experiments/fragment-warning-v21")
OLD = Path("/home/ubuntu/Asteroids/models/oracle-survival-v3-v20-conservative")
NIGHT = Path("/home/ubuntu/Asteroids/experiments/night-2026-09-08")
RUN = Path("/home/ubuntu/Asteroids/models/oracle-survival-v3-v21-action-fire")


def main():
    assert not RUN.exists(), "do not overwrite an existing run"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(ROOT / "source/model.zip") == sha(NIGHT / "morning-champion/model.zip")
    transfer = json.loads((ROOT / "transfer-check.json").read_text())
    assert transfer["identical_episode_trajectories"] == 16
    assert transfer["max_action_probability_difference"] < 1e-5
    assert transfer["throughput_ratio"] > 0.7
    source_meta = json.loads((ROOT / "source/metadata.json").read_text())
    record = next(json.loads(line) for line in (OLD / "evaluation.jsonl").open()
                  if json.loads(line)["episode"] == source_meta["episodes"])
    record["episode"] = 0
    baseline = json.loads((NIGHT / "morning-champion.json").read_text())
    record["stages"][28] = baseline["benchmarks"]["current"]["aggregate"]
    RUN.mkdir(parents=True)
    spec = load_curriculum("configs/rl-survival-v3-action-fire.toml")
    tracker = PPOChampionTracker(
        RUN, spec.retention_completion, patience=4, retention_floor=spec.retention_floor,
        learning_rate=5e-5/3, minimum_learning_rate=5e-5/9,
        promotion_completion=spec.promotion_completion, clear_target=spec.promotion_clear_rate,
        accuracy_targets=tuple(spec.promotion_accuracy if s.promotion_accuracy is None
                               else s.promotion_accuracy for s in spec.stages))
    tracker.consider(record, ROOT / "widened-source", allow_recovery=False)
    shutil.copy2("/etc/systemd/system/asteroids.service.d/bridge.conf", ROOT / "v20-bridge.conf")
    (ROOT / "starting-baseline.json").write_text(json.dumps(baseline, indent=2)+"\n")
    print(json.dumps({"run": str(RUN), "champion_primed": True,
                      "clear_rate": record["stages"][28]["clear_rate"]}), flush=True)


if __name__ == "__main__":
    main()
