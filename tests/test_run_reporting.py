from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def _make_run(tmp_path: Path) -> tuple[Path, dict]:
    run = tmp_path / "reporting-run"
    run.mkdir()
    stage = {
        "completion_rate": 0.91,
        "clear_rate": 0.8125,
        "mean_wave": 0.0,
        "mean_accuracy": 0.63,
        "mean_survival_time": 27.3,
    }
    record = {
        "episode": 500,
        "training_stage": 0,
        "promotion_completion_target": 0.90,
        "promotion_clear_rate_target": 0.80,
        "stages": [stage],
    }
    (run / "evaluation.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return run, record


def _environment(tmp_path: Path, run: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ps = bin_dir / "ps"
    ps.write_text(
        "#!/bin/sh\n"
        f"echo '4242 python -m asteroid_survival train-ppo --output {run}'\n",
        encoding="utf-8")
    ps.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PY": os.environ.get("PYTHON", "python3"), "FOLLOW_EVERY": "0.02"}


def _run_with_evaluation(tmp_path: Path, command: str) -> str:
    run, _ = _make_run(tmp_path)
    env = _environment(tmp_path, run)
    result = subprocess.run(
        [str(ROOT / "run.sh"), command, str(run)], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=5, check=True)
    return result.stdout


def test_status_reports_the_clear_rate(tmp_path):
    output = _run_with_evaluation(tmp_path, "status")
    assert "completion  91.0%" in output
    assert "clear  81.2%" in output


def test_status_survives_an_oracle_with_no_log_here(tmp_path):
    # Any process matching `planning_oracle.py` -- including an ssh command that launches one
    # on the VM -- used to make `status` print its RUNNING line and exit 1 silently, because
    # no `oracle-*.log` existed in this directory. The two tests above failed whenever one was
    # running, and passed otherwise.
    run, _ = _make_run(tmp_path)
    env = _environment(tmp_path, run)
    pgrep = tmp_path / "bin" / "pgrep"
    pgrep.write_text("#!/bin/sh\necho 777\n", encoding="utf-8")
    pgrep.chmod(0o755)
    result = subprocess.run(
        [str(ROOT / "run.sh"), "status", str(run)], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=5, check=True)
    assert "planning oracle: RUNNING (pid 777)" in result.stdout
    assert "clear  81.2%" in result.stdout


def test_follow_reports_clear_rate_and_target_for_new_records(tmp_path):
    # The initial status block and the streaming formatter deliberately use different
    # layouts; pin both so a future display edit cannot hide a promotion gate again.
    run, record = _make_run(tmp_path)
    env = _environment(tmp_path, run)
    follower = subprocess.Popen(
        [str(ROOT / "run.sh"), "follow", str(run)], cwd=ROOT, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        time.sleep(1.0)
        record["episode"] = 1000
        with (run / "evaluation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        time.sleep(0.2)
    finally:
        if follower.poll() is None:
            os.killpg(follower.pid, signal.SIGTERM)
        stdout, stderr = follower.communicate(timeout=5)
    assert follower.returncode == -15, stderr
    assert "ep   1000" in stdout
    assert "clear  81.2% of 80%" in stdout
