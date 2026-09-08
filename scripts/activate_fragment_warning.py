"""Activate v21 only after checking its trained smoke checkpoint; retain rollback."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from stable_baselines3 import PPO

ROOT = Path("/home/ubuntu/asteroids-experiments/fragment-warning-v21")
REPO = Path("/home/ubuntu/Asteroids-v21")
DROPIN = Path("/etc/systemd/system/asteroids.service.d/bridge.conf")


def command(*args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def main():
    torch.set_num_threads(1)
    model = PPO.load(ROOT / "verified-smoke/checkpoint_000008/model.zip", device="cpu")
    assert model.n_steps == 256 and model.gamma == 0.997 and model.target_kl == 0.02
    assert model.observation_space.shape == (1317,)
    learned = {}
    for name, parameter in model.policy.state_dict().items():
        assert torch.isfinite(parameter).all()
        if parameter.ndim == 2 and parameter.shape[1] == 1317:
            learned[name] = int(torch.count_nonzero(parameter[:, -32:]))
    assert len(learned) == 2 and all(learned.values()), "new input weights have not learned"
    environment = command("systemctl", "show", "asteroids", "-p", "Environment", "--value")
    assert "RUN=models/oracle-survival-v3-v20-conservative " in environment
    assert DROPIN.read_bytes() == (ROOT / "v20-bridge.conf").read_bytes()
    assert (REPO / "models/oracle-survival-v3-v21-action-fire/champion/model.zip").is_file()
    # The old run directory is retained in full. Stopping prevents checkpoint pruning
    # while selecting and copying its newest complete learner for an extra backup.
    command("sudo", "systemctl", "stop", "asteroids.service")
    try:
        old_run = Path("/home/ubuntu/Asteroids/models/oracle-survival-v3-v20-conservative")
        complete = sorted(p for p in old_run.glob("checkpoint_*")
                          if (p / "model.zip").is_file() and (p / "metadata.json").is_file())
        assert complete
        shutil.copytree(complete[-1], ROOT / "v20-last-complete")
        command("sudo", "install", "-m", "644", str(REPO / "cloud/asteroids-v21.conf"), str(DROPIN))
        command("sudo", "systemctl", "daemon-reload")
        command("sudo", "systemctl", "enable", "--now", "asteroids.service")
        command("systemctl", "is-active", "asteroids.service")
    except BaseException:
        command("sudo", "install", "-m", "644", str(ROOT / "v20-bridge.conf"), str(DROPIN))
        command("sudo", "systemctl", "daemon-reload")
        command("sudo", "systemctl", "start", "asteroids.service")
        raise
    report = {"activated": True, "old_checkpoint": str(complete[-1]),
              "new_feature_nonzero_weights": learned,
              "service_environment": command("systemctl", "show", "asteroids", "-p", "Environment", "--value")}
    (ROOT / "activation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
