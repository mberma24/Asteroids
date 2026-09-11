"""Start conservative ordinary PPO and review it after eight hours without stopping it."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.orbit_experiment import atomic_json, paired_difference, read_json, retention_ok
from asteroid_survival.rl.ppo_support import PPOChampionTracker

REPO = Path("/home/ubuntu/Asteroids")
TOOLS = Path("/home/ubuntu/Asteroids/archives/orbit-practice-v18")
ROOT = Path("/home/ubuntu/Asteroids/experiments/night-2026-09-08")
RUN = REPO / "models/oracle-survival-v3-v20-conservative"
LR = 5e-5 / 3
EVALUATOR = [str(REPO / ".venv/bin/python"), str(TOOLS / "scripts/orbit_experiment.py"),
             "evaluate", "--root", "/home/ubuntu/Asteroids/experiments/v18"]


def evaluate(checkpoint, name, *, seed=1_000_000_832, count=256, names="current,full"):
    destination = ROOT / f"{name}.json"
    if not destination.exists():
        subprocess.run([*EVALUATOR, "--checkpoint", str(checkpoint), "--names", names,
                        "--seed", str(seed), "--count", str(count),
                        "--output", str(destination)], cwd=TOOLS, check=True)
    return read_json(destination)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose():
    reference = read_json(ROOT / "reference-check.json")
    candidate = read_json(ROOT / "v19-check.json")
    if not reference or not candidate:
        raise RuntimeError("both 256-seed checkpoint comparisons must finish first")
    differences = {name: paired_difference(candidate["benchmarks"][name]["episodes"],
                                           reference["benchmarks"][name]["episodes"])
                   for name in ("current", "full")}
    selected, baseline = "control-reference", reference
    final = read_json(Path("/home/ubuntu/Asteroids/experiments/v18/report.json"))
    previous = Path(final["state"]["candidates"]["control"])
    if sha(ROOT / "control-reference/model.zip") != sha(previous / "model.zip"):
        raise RuntimeError("reference differs from the independently tested control candidate")
    retention = {"benchmarks": final["final"]["control"]}
    if not retention_ok(retention["benchmarks"]):
        raise RuntimeError("reference failed retention")
    # Favor the existing incumbent when its measured current-round score is higher
    # without losing on full orbit. This selects a starting point, not a significance claim.
    if (differences["current"]["difference"] >= 0
            and differences["full"]["difference"] >= 0):
        candidate_retention = evaluate(
            ROOT / "v19-champion", "v19-retention",
            seed=1_000_000_576, count=64,
            names="retention23,retention24,retention25,retention26,retention27,retention28")
        if retention_ok(candidate_retention["benchmarks"]):
            selected, baseline, retention = "v19-champion", candidate, candidate_retention
    return selected, baseline, retention, differences


def launch():
    selected, baseline, retention, differences = choose()
    if RUN.exists() or (ROOT / "launch.json").exists():
        raise RuntimeError("night run already prepared; inspect before reusing its directory")
    environment = subprocess.check_output(
        ["systemctl", "show", "asteroids", "-p", "Environment", "--value"], text=True)
    if "RUN=models/oracle-survival-v3-v19 " not in environment:
        raise RuntimeError("active service is no longer v19")
    shutil.copytree(ROOT / selected, ROOT / "selected-source")
    RUN.mkdir(parents=True)
    spec = load_curriculum(REPO / "configs/rl-survival-v3.toml")
    stages = [{"episodes": 0} for _ in range(29)]
    for name, value in retention["benchmarks"].items():
        if name.startswith("retention"):
            stages[int(name[9:]) - 1] = value["aggregate"]
    stages[28] = baseline["benchmarks"]["current"]["aggregate"]
    tracker = PPOChampionTracker(
        RUN, spec.retention_completion, patience=4, retention_floor=spec.retention_floor,
        learning_rate=LR, minimum_learning_rate=LR / 3,
        promotion_completion=spec.promotion_completion, clear_target=spec.promotion_clear_rate,
        accuracy_targets=tuple(spec.promotion_accuracy if s.promotion_accuracy is None
                               else s.promotion_accuracy for s in spec.stages))
    # Install the measured source before any training, rather than crowning the first panel.
    tracker.consider({"episode": 0, "training_stage": 28, "stages": stages},
                     ROOT / "selected-source", allow_recovery=False)
    atomic_json(ROOT / "starting-baseline.json", baseline)
    now = datetime.datetime.now(datetime.timezone.utc)
    review_at = now + datetime.timedelta(hours=8)
    atomic_json(ROOT / "launch.json", {
        "run": str(RUN), "selected_source": selected,
        "starting_metrics": baseline["benchmarks"]["current"]["aggregate"],
        "checkpoint_comparison": differences, "learning_rate": LR,
        "started_at": now.isoformat(), "review_at": review_at.isoformat(),
        "mode": "ordinary training; review does not stop or restart training",
    })
    dropin = Path("/etc/systemd/system/asteroids.service.d/bridge.conf")
    shutil.copy2(dropin, ROOT / "v19-bridge.conf")
    (ROOT / "review.service").write_text(
        "[Unit]\nDescription=Review ordinary Asteroids training after eight hours\n"
        "[Service]\nType=oneshot\nUser=ubuntu\nWorkingDirectory=/home/ubuntu/Asteroids\n"
        "Environment=PYTHONPATH=/home/ubuntu/Asteroids/archives/orbit-practice-v18/src\n"
        "Environment=OMP_NUM_THREADS=1\nEnvironment=MKL_NUM_THREADS=1\nNice=10\n"
        "TimeoutStartSec=3600\n"
        "ExecStart=/home/ubuntu/Asteroids/.venv/bin/python scripts/overnight_conservative.py review\n"
        "StandardOutput=append:/home/ubuntu/Asteroids/experiments/night-2026-09-08/review.log\n"
        "StandardError=append:/home/ubuntu/Asteroids/experiments/night-2026-09-08/review.log\n")
    (ROOT / "review.timer").write_text(
        "[Unit]\nDescription=Asteroids morning review\n[Timer]\n"
        f"OnCalendar={review_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        "Persistent=true\nUnit=asteroids-morning-review.service\n"
        "[Install]\nWantedBy=timers.target\n")
    subprocess.run(["sudo", "systemctl", "stop", "asteroids"], check=True)
    try:
        # Original v19 files remain in place, including its most recent checkpoint.
        subprocess.run(["sudo", "install", "-m", "644",
                        str(REPO / "cloud/asteroids-v20.conf"), str(dropin)], check=True)
        for suffix in ("service", "timer"):
            subprocess.run(["sudo", "install", "-m", "644", str(ROOT / f"review.{suffix}"),
                            f"/etc/systemd/system/asteroids-morning-review.{suffix}"], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "--now", "asteroids"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "--now",
                        "asteroids-morning-review.timer"], check=True)
    except Exception:
        subprocess.run(["sudo", "systemctl", "stop", "asteroids"], check=False)
        subprocess.run(["sudo", "cp", str(ROOT / "v19-bridge.conf"), str(dropin)], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "start", "asteroids"], check=True)
        raise
    print(json.dumps(read_json(ROOT / "launch.json"), indent=2))


def review():
    # This is read-only with respect to the live trainer: scoring uses a copied champion.
    snapshot = ROOT / "morning-champion"
    if not snapshot.exists():
        pending = snapshot.with_name(snapshot.name + ".incomplete")
        if pending.exists():
            shutil.rmtree(pending)
        shutil.copytree(RUN / "champion", pending)
        # A failed copy never masquerades as a reusable complete snapshot.
        json.loads((pending / "metadata.json").read_text())
        import zipfile
        with zipfile.ZipFile(pending / "model.zip") as archive:
            if archive.testzip() is not None:
                raise RuntimeError("invalid champion snapshot")
        pending.rename(snapshot)
    source = evaluate(ROOT / "selected-source", "morning-source")
    champion = evaluate(snapshot, "morning-champion")
    differences = {
        name: paired_difference(champion["benchmarks"][name]["episodes"],
                                source["benchmarks"][name]["episodes"])
        for name in ("current", "full")}
    logs = [json.loads(line) for line in (RUN / "evaluation.jsonl").read_text().splitlines()]
    latest = logs[-1] if logs else {}
    current = differences["current"]
    verdict = ("measured improvement" if current["difference"] >= .05 and current["ci95"][0] > 0
               else "measured regression" if current["ci95"][1] < 0 else "inconclusive")
    report = {
        "verdict": verdict, "paired_differences": differences, "source": source,
        "champion": champion, "champion_state": read_json(RUN / "champion_state.json"),
        "latest_training_evaluation": latest,
        "note": "Training continues. Evaluation intervals do not capture variation across training seeds.",
    }
    atomic_json(ROOT / "morning-report.json", report)
    lines = ["# Overnight ordinary-training review", "", f"Result: {verdict}.", "",
             "| Checkpoint | Round 29 clear | Full-orbit clear |",
             "|---|---:|---:|"]
    for name, value in (("Starting policy", source), ("Protected champion", champion)):
        values = [value["benchmarks"][n]["aggregate"]["clear_rate"] for n in ("current", "full")]
        lines.append(f"| {name} | {values[0]:.1%} | {values[1]:.1%} |")
    lines += ["", "Training remains running. This review does not change weights or service state.",
              "Per-seed results, paired 95% bootstrap intervals, and the latest learner evaluation "
              "are in morning-report.json."]
    (ROOT / "MORNING_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "differences": differences}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "review", "choose"))
    args = parser.parse_args()
    if args.command == "launch":
        launch()
    elif args.command == "review":
        review()
    else:
        selected, baseline, retention, differences = choose()
        print(json.dumps({"selected": selected, "differences": differences,
                          "baseline": {n: v["aggregate"] for n, v in baseline["benchmarks"].items()}},
                         indent=2))
