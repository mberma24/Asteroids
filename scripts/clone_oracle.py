"""Clone recorded oracle decisions into an existing PPO policy, in place.

The policy network is trained inside the loaded stable-baselines3 model rather than in a
separate network of the same shape, so there is nothing to transplant afterwards: the result
is an ordinary checkpoint directory that `INITIALIZE_FROM` accepts, with the source's critic
and observation layout untouched.

Two labels per pair, both from `planning_oracle.py --record`:
  actions   the plan the search chose -- cross-entropy target
  safe      bitmask of first actions whose plan survived the horizon -- the set loss
            `-log sum_{a in safe} pi(a)` says "any of these", which is closer to what the
            search actually knows than the single argmax it happened to return
Validation holds out whole episodes: consecutive frames are near-duplicates and a random
split would leak. The 2026-08-26 clone reported train 99% / val 19% for that reason.

Usage: PYTHONPATH=src python scripts/clone_oracle.py --source CHAMPION --output DIR
           --traces a.npz [b.npz ...] [--epochs 30] [--set-weight 0.5]
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO


def load_traces(paths: list[str]) -> dict[str, np.ndarray]:
    parts = {"observations": [], "actions": [], "safe": [], "episodes": []}
    offset = 0
    for path in paths:
        with np.load(path) as data:
            for key in parts:
                value = data[key]
                if key == "episodes":
                    value = value + offset
                parts[key].append(value)
            offset += int(data["episodes"].max()) + 1 if len(data["episodes"]) else 0
    return {key: np.concatenate(value) for key, value in parts.items()}


def episode_split(episodes: np.ndarray, fraction: float) -> np.ndarray:
    """True for training pairs; the last `fraction` of episodes are held out."""
    ids = np.unique(episodes)
    cut = ids[int(len(ids) * (1 - fraction))] if len(ids) > 1 else ids[-1] + 1
    return episodes < cut


def logits_of(model, observations: torch.Tensor) -> torch.Tensor:
    return model.policy.get_distribution(observations).distribution.logits


def losses(logits: torch.Tensor, actions: torch.Tensor, safe: torch.Tensor,
           set_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    log_probs = torch.log_softmax(logits, dim=-1)
    argmax = torch.nn.functional.nll_loss(log_probs, actions)
    bits = (safe.unsqueeze(1) >> torch.arange(logits.shape[1])) & 1
    mask = bits.bool()
    has_set = mask.any(dim=1)
    set_loss = torch.zeros((), dtype=logits.dtype)
    if has_set.any():
        masked = log_probs.masked_fill(~mask, -1e9)
        set_loss = -torch.logsumexp(masked[has_set], dim=-1).mean()
    total = argmax + set_weight * set_loss
    return total, {"argmax": float(argmax.detach()), "set": float(set_loss.detach())}


@torch.no_grad()
def agreement(model, observations: torch.Tensor, actions: torch.Tensor,
              safe: torch.Tensor, batch: int = 4096) -> dict[str, float]:
    top1 = in_set = 0
    for start in range(0, len(actions), batch):
        logits = logits_of(model, observations[start:start + batch])
        chosen = logits.argmax(dim=-1)
        top1 += int((chosen == actions[start:start + batch]).sum())
        in_set += int((((safe[start:start + batch] >> chosen) & 1) == 1).sum())
    return {"top1": top1 / max(1, len(actions)), "in_safe_set": in_set / max(1, len(actions))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="checkpoint dir to clone into")
    parser.add_argument("--output", required=True, help="checkpoint dir to write")
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--set-weight", type=float, default=0.5)
    parser.add_argument("--validation", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.time()
    source = Path(args.source)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    data = load_traces(args.traces)
    if data["observations"].shape[1] != metadata["observation_size"]:
        raise SystemExit(f"traces are {data['observations'].shape[1]} wide, the source "
                         f"expects {metadata['observation_size']}: record with "
                         f"--layout-from {source}")
    train = episode_split(data["episodes"], args.validation)
    if not (~train).any() or not train.any():
        raise SystemExit(f"validation split {args.validation} leaves one side empty over "
                         f"{int(data['episodes'].max()) + 1} episodes")
    tensors = {k: torch.as_tensor(v) for k, v in data.items() if k != "episodes"}
    tensors["safe"] = tensors["safe"].to(torch.int64)
    split = {name: {k: v[torch.as_tensor(sel)] for k, v in tensors.items()}
             for name, sel in (("train", train), ("val", ~train))}
    counts = np.bincount(data["actions"], minlength=metadata["num_actions"])
    majority = int(counts.argmax())
    report: dict = {
        "source": str(source), "traces": args.traces,
        "pairs": int(len(data["actions"])), "episodes": int(data["episodes"].max()) + 1,
        "train_pairs": int(train.sum()), "val_pairs": int((~train).sum()),
        "majority_action": majority,
        "majority_baseline_val": float((split["val"]["actions"] == majority).float().mean()),
        "safe_set_mean_size": float(np.mean([bin(int(s)).count("1") for s in data["safe"]])),
        "epochs": [],
    }

    model = PPO.load(source / "model.zip", device="cpu")
    report["source_agreement_val"] = agreement(model, **split["val"])
    policy = model.policy
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(
        policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    best_state, best_val, stale = copy.deepcopy(policy.state_dict()), float("inf"), 0
    generator = torch.Generator().manual_seed(args.seed)
    n = len(split["train"]["actions"])
    for epoch in range(args.epochs):
        policy.set_training_mode(True)
        order = torch.randperm(n, generator=generator)
        total = {"argmax": 0.0, "set": 0.0}
        for start in range(0, n, args.batch):
            index = order[start:start + args.batch]
            batch = {k: v[index] for k, v in split["train"].items()}
            loss, parts = losses(logits_of(model, batch["observations"]), batch["actions"],
                                 batch["safe"], args.set_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            for key in total:
                total[key] += parts[key] * len(index)
        policy.set_training_mode(False)
        with torch.no_grad():
            val_loss, val_parts = losses(logits_of(model, split["val"]["observations"]),
                                         split["val"]["actions"], split["val"]["safe"],
                                         args.set_weight)
        row = {"epoch": epoch, "train": {k: v / n for k, v in total.items()},
               "val_loss": float(val_loss), "val": val_parts,
               "train_agreement": agreement(model, **split["train"]),
               "val_agreement": agreement(model, **split["val"])}
        report["epochs"].append(row)
        print(f"epoch {epoch:3d}  train argmax {row['train']['argmax']:.3f} set "
              f"{row['train']['set']:.3f}  val loss {row['val_loss']:.3f}  "
              f"top1 train {row['train_agreement']['top1']:.3f} val "
              f"{row['val_agreement']['top1']:.3f}  in-set val "
              f"{row['val_agreement']['in_safe_set']:.3f}", flush=True)
        if float(val_loss) < best_val:
            best_val, best_state, stale = float(val_loss), copy.deepcopy(policy.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    policy.load_state_dict(best_state)
    report["best_val_loss"] = best_val
    report["final_agreement_val"] = agreement(model, **split["val"])
    report["seconds"] = time.time() - started

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    model.save(destination / "model.zip")
    metadata = copy.deepcopy(metadata)
    metadata.update(episodes=0, environment_steps=0, parent_checkpoint=str(source),
                    cloned_from={"traces": args.traces, "pairs": report["pairs"],
                                 "val_agreement": report["final_agreement_val"]})
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                               encoding="utf-8")
    (destination / "clone-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pairs", "episodes", "majority_baseline_val",
                                              "source_agreement_val", "final_agreement_val",
                                              "best_val_loss", "seconds")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
