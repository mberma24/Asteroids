"""Unattended: clone the blind oracle, correct it with DAgger, then hand PPO the best clone.

Every step writes its output under ROOT and is skipped when that output already exists, so
a crash or reboot resumes where it stopped. Any failure -- and the fallback branch of the
final decision -- ends the same way: `asteroids.service` is started with whatever drop-in is
installed, which is the previous run's until the decision step replaces it. A watchdog timer
(`watchdog` subcommand, hourly) starts the service if neither it nor this pipeline is alive.

    run       the whole pipeline (default)
    watchdog  start asteroids.service if nothing is training
    --local   no systemd: finish with a short train-ppo instead of installing the service
    --smoke   tiny budgets everywhere; exercises every step in minutes
    --fail-at STEP  raise inside STEP, to prove the fallback path

Decision rule, fixed before the data existed: the clone with the best held-out round-29
clear launches PPO fine-tuning if that clear is at least THRESHOLD; otherwise the previous
run resumes. The 2026-08-26 clone scored 0.078, so a bad clone is a live possibility and
the box must not sit on it for days.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CURRICULUM = "configs/rl-survival-v3-action-fire.toml"
STAGE = 28                       # zero-based: survival-v3-round-29
RUN = "models/oracle-survival-v3-v22-oracle-clone"
DROPIN = Path("/etc/systemd/system/asteroids.service.d/bridge.conf")
RETENTION = [f"retention{r}" for r in range(23, 29)]
# Seeds that must never be recorded: 10000-10255 is the held-out panel, 1,000,000,000+ the
# benchmark, diagnostic and oracle-ceiling ranges. Each recording round gets its own block.
RECORD_SEEDS = {"r0": 5_000_000, "r1": 6_000_000, "r2": 7_000_000}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root).resolve()
        self.python = str(REPO / ".venv/bin/python")
        if not Path(self.python).exists():
            self.python = sys.executable
        self.env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1"}
        smoke = args.smoke
        self.settings = {
            "workers": 2 if smoke else 4,
            "record": {"candidates": 4 if smoke else 16, "horizon": 5 if smoke else 30,
                       "episodes": 2 if smoke else 100_000},
            "hours": {"r0": None if smoke else args.hours_r0,
                      "r1": None if smoke else args.hours_r1,
                      "r2": None if smoke else args.hours_r2},
            "clone_epochs": 1 if smoke else 30,
            "bench_count": 4 if smoke else 256,
            "retention_count": 2 if smoke else 64,
            "threshold": args.threshold,
        }

    # -- infrastructure ------------------------------------------------------------------

    def step_done(self, *outputs: Path) -> bool:
        return all(p.exists() for p in outputs)

    def check_fail(self, step: str) -> None:
        if self.args.fail_at == step:
            raise RuntimeError(f"injected failure at {step}")

    def run(self, command: list[str], *, hours: float | None = None, cwd: Path = REPO) -> None:
        log("$ " + " ".join(str(c) for c in command)
            + (f"   [budget {hours}h]" if hours else ""))
        process = subprocess.Popen([str(c) for c in command], cwd=cwd, env=self.env,
                                   start_new_session=True)
        try:
            process.wait(timeout=hours * 3600 if hours else None)
        except subprocess.TimeoutExpired:
            log(f"budget reached, stopping pid {process.pid}")
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return
        if process.returncode != 0:
            raise RuntimeError(f"command failed ({process.returncode}): {command[:3]}")

    def systemctl(self, *words: str) -> None:
        if self.args.local:
            log(f"(local) would run: systemctl {' '.join(words)}")
            return
        subprocess.run(["sudo", "systemctl", *words], check=True)

    # -- steps ---------------------------------------------------------------------------

    def snapshot_source(self) -> Path:
        destination = self.root / "source"
        if not self.step_done(destination / "model.zip"):
            self.check_fail("source")
            source = Path(self.args.source)
            log(f"snapshotting {source} -> {destination}")
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(source, destination.with_suffix(".partial"))
            destination.with_suffix(".partial").rename(destination)
        return destination

    def record(self, name: str, *, driver: Path | None) -> Path:
        trace = self.root / f"{name}.npz"
        summary = self.root / f"{name}.json"
        if self.step_done(trace, summary):
            return trace
        self.check_fail(name)
        settings = self.settings["record"]
        command = [self.python, "scripts/planning_oracle.py", "--stages", STAGE,
                   "--seeds", settings["episodes"], "--seed-start", RECORD_SEEDS[name],
                   "--candidates", settings["candidates"], "--horizon", settings["horizon"],
                   "--workers", self.settings["workers"], "--blind",
                   "--curriculum", CURRICULUM, "--layout-from", self.root / "source",
                   "--record", trace, "--output", self.root / f"{name}-oracle.json"]
        if driver is not None:
            command += ["--driver", driver]
        started = time.time()
        self.run(command, hours=self.settings["hours"][name])
        if not trace.exists():
            raise RuntimeError(f"{name}: no trace written -- fewer than 20 episodes finished")
        with np.load(trace) as data:
            pairs, episodes = int(len(data["actions"])), int(data["episodes"].max()) + 1
        write_json(summary, {"pairs": pairs, "episodes": episodes,
                             "hours": (time.time() - started) / 3600,
                             "driver": str(driver) if driver else None})
        log(f"{name}: {episodes} episodes, {pairs:,} pairs")
        return trace

    def clone(self, name: str, traces: list[Path]) -> Path:
        destination = self.root / name
        if self.step_done(destination / "model.zip", destination / "clone-report.json"):
            return destination
        self.check_fail(name)
        shutil.rmtree(destination, ignore_errors=True)
        self.run([self.python, "scripts/clone_oracle.py", "--source", self.root / "source",
                  "--output", destination, "--traces", *traces,
                  "--epochs", self.settings["clone_epochs"],
                  "--threads", self.settings["workers"]])
        return destination

    def benchmark(self, checkpoint: Path) -> dict:
        rounds = checkpoint.with_name(checkpoint.name + "-bench.json")
        retention = checkpoint.with_name(checkpoint.name + "-retention.json")
        common = [self.python, "scripts/benchmark_checkpoint.py", checkpoint,
                  "--workers", self.settings["workers"]]
        if self.args.snapshots:
            common += ["--snapshots", self.args.snapshots]
        if not rounds.exists():
            self.check_fail(f"{checkpoint.name}-bench")
            self.run(common + [rounds, "--names", "current",
                               "--count", self.settings["bench_count"]])
        if not retention.exists():
            self.run(common + [retention, "--names", ",".join(RETENTION),
                               "--seed", 1_000_000_576,
                               "--count", self.settings["retention_count"]])
        result = {"current": read_json(rounds)["benchmarks"]["current"]["aggregate"],
                  "retention": {k: v["aggregate"]
                                for k, v in read_json(retention)["benchmarks"].items()}}
        log(f"{checkpoint.name}: round 29 clear {result['current']['clear_rate']:.3f}, "
            "retention " + " ".join(f"{v['completion_rate']:.2f}"
                                    for v in result["retention"].values()))
        return result

    def decide(self, scores: dict[str, dict]) -> Path | None:
        decision = self.root / "decision.json"
        if decision.exists():
            chosen = read_json(decision)["chosen"]
            return Path(chosen) if chosen else None
        self.check_fail("decide")
        best = max(scores, key=lambda k: scores[k]["current"]["clear_rate"])
        clear = scores[best]["current"]["clear_rate"]
        chosen = None
        if clear >= self.settings["threshold"]:
            chosen = self.root / "best-clone"
            shutil.rmtree(chosen, ignore_errors=True)
            shutil.copytree(self.root / best, chosen)
        write_json(decision, {"scores": scores, "best": best, "clear": clear,
                              "threshold": self.settings["threshold"],
                              "chosen": str(chosen) if chosen else None,
                              "outcome": ("fine-tune from the clone" if chosen else
                                          "resume the previous run")})
        log(f"decision: best clone {best} at {clear:.3f} against {self.settings['threshold']}"
            f" -> {'fine-tune' if chosen else 'resume previous run'}")
        return chosen

    def prime_run(self, clone: Path, score: dict) -> Path:
        """Install the clone as the new run's protected champion with its measured score,
        so the first lucky 64-episode panel cannot crown itself (prime_fragment_warning.py)."""
        from asteroid_survival.rl.curriculum import load_curriculum
        from asteroid_survival.rl.ppo_support import PPOChampionTracker
        run = (self.root / "smoke-run") if self.args.local else (REPO / RUN)
        if (run / "champion_state.json").exists():
            return run
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        spec = load_curriculum(REPO / CURRICULUM)
        stages = [{"episodes": 0} for _ in range(STAGE + 1)]
        for name, value in score["retention"].items():
            stages[int(name.removeprefix("retention")) - 1] = value
        stages[STAGE] = score["current"]
        rate = self.args.learning_rate
        tracker = PPOChampionTracker(
            run, spec.retention_completion, patience=4, retention_floor=spec.retention_floor,
            learning_rate=rate, minimum_learning_rate=rate / 3,
            promotion_completion=spec.promotion_completion,
            clear_target=spec.promotion_clear_rate,
            accuracy_targets=tuple(spec.promotion_accuracy if s.promotion_accuracy is None
                                   else s.promotion_accuracy for s in spec.stages))
        tracker.consider({"episode": 0, "training_stage": STAGE, "stages": stages},
                         clone, allow_recovery=False)
        return run

    def launch(self, clone: Path, score: dict) -> None:
        run = self.prime_run(clone, score)
        conf = self.root / "asteroids-v22-clone.conf"
        conf.write_text(
            "# Ordinary PPO fine-tuning from the best oracle clone (holiday pipeline).\n"
            "[Service]\n"
            f"Environment=RUN={run.relative_to(REPO) if not self.args.local else run}\n"
            f"Environment=CURRICULUM={CURRICULUM}\n"
            f"Environment=INITIALIZE_FROM={clone}\n"
            f"Environment=START_STAGE={STAGE + 1}\n"
            "Environment=SEED=2209\n"
            f"Environment=LEARNING_RATE={self.args.learning_rate}\n"
            "Environment=PPO_ENT_COEF=0.0025\n"
            "Environment=PPO_TARGET_KL=0.02\n"
            "Environment=EVAL_EVERY=500\n", encoding="utf-8")
        if self.args.local:
            env = {**self.env, "OUTPUT": str(run), "CURRICULUM": CURRICULUM,
                   "INITIALIZE_FROM": str(clone), "START_STAGE": str(STAGE + 1),
                   "EVAL_EVERY": "4", "PARALLEL_ENVS": "2", "PPO_EVAL_WORKERS": "2",
                   "LEARNING_RATE": str(self.args.learning_rate), "PPO_ENT_COEF": "0.0025",
                   "PPO_TARGET_KL": "0.02", "PPO_DEVICE": "cpu"}
            subprocess.run([str(REPO / "run.sh"), "train-ppo", "8"], cwd=REPO, env=env,
                           check=True)
            return
        shutil.copy2(DROPIN, self.root / "previous-bridge.conf")
        subprocess.run(["sudo", "install", "-m", "644", str(conf), str(DROPIN)], check=True)
        self.systemctl("daemon-reload")
        self.systemctl("start", "asteroids")

    # -- driver --------------------------------------------------------------------------

    def main(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "settings.json", {**self.settings, "args": vars(self.args)})
        self.systemctl("stop", "asteroids")
        source = self.snapshot_source()
        scores: dict[str, dict] = {}
        traces: list[Path] = []
        driver: Path | None = None
        for index in range(3):
            traces.append(self.record(f"r{index}", driver=driver))
            clone = self.clone(f"c{index}", traces)
            scores[f"c{index}"] = self.benchmark(clone)
            write_json(self.root / "scores.json", scores)
            driver = clone
        chosen = self.decide(scores)
        if chosen is None:
            self.systemctl("start", "asteroids")
            return
        best = read_json(self.root / "decision.json")["best"]
        self.launch(chosen, scores[best])


def watchdog() -> None:
    active = lambda unit: subprocess.run(["systemctl", "is-active", "--quiet", unit]).returncode == 0
    if active("asteroids-holiday.service") or active("asteroids.service"):
        return
    log("watchdog: nothing training, starting asteroids.service")
    subprocess.run(["sudo", "systemctl", "start", "asteroids"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=("run", "watchdog"))
    parser.add_argument("--root", default=str(REPO / "experiments/holiday-2026-09-11"))
    parser.add_argument("--source",
                        default=str(REPO / "models/oracle-survival-v3-v21-action-fire/champion"))
    parser.add_argument("--snapshots", default=None,
                        help="benchmark config snapshots (default: the VM path)")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--learning-rate", type=float, default=5e-5 / 3)
    parser.add_argument("--hours-r0", type=float, default=10.0)
    parser.add_argument("--hours-r1", type=float, default=5.0)
    parser.add_argument("--hours-r2", type=float, default=5.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--fail-at", default=None)
    args = parser.parse_args()
    if args.command == "watchdog":
        watchdog()
        return
    pipeline = Pipeline(args)
    try:
        pipeline.main()
    except BaseException as exc:      # a SIGTERM from systemd must fall back too
        write_json(pipeline.root / "error.json",
                   {"error": repr(exc), "traceback": traceback.format_exc(),
                    "time": time.time()})
        log(f"FAILED: {exc!r} -- starting the previous run")
        try:
            pipeline.systemctl("start", "asteroids")
        finally:
            raise


if __name__ == "__main__":
    main()
