"""How does the champion die on round 26? Age and origin of the killing rock.

Usage: python scripts/death_causes.py CHECKPOINT SEEDS WORKERS [CURRICULUM] [OUTPUT]
"""
import json, math, os, sys, statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "src")
from asteroid_survival.rl.curriculum import load_curriculum
from asteroid_survival.rl.ppo import _stage_env, PPOController
from asteroid_survival.math2d import Vec2, wrapped_delta

CKPT = sys.argv[1]
_META = json.load(open(os.path.join(CKPT, "metadata.json")))
LAYOUT = dict(_META.get("observation_layout") or {})
for _key, _default in (("history_frames", 8), ("history_long_frames", 8),
                       ("history_long_stride", 8), ("max_projectiles", 8), ("version", 7)):
    LAYOUT.setdefault(_key, _default)
STAGE = 25
CURRICULUM = sys.argv[4] if len(sys.argv) > 4 else "configs/rl-survival-v2.toml"

def run(seed):
    spec = load_curriculum(CURRICULUM)
    env = _stage_env(spec, STAGE, LAYOUT)
    ctl = PPOController(CKPT)
    obs, _ = env.reset(seed)
    events = []
    real_step = env.simulation.step
    def wrapped(actions):
        r = real_step(actions)
        events.extend((env.simulation.step_count, e) for e in r.events)
        return r
    env.simulation.step = wrapped
    snaps = []   # per decision: (elapsed, ship pos, {id: (x,y,radius)})
    done = False
    while not done:
        st = env.state
        ship = next(s for s in st.ships if s.id == env.agent_id)
        snaps.append((st.elapsed, (ship.x, ship.y), {a.id: (a.x, a.y, a.radius, a.vx, a.vy) for a in st.asteroids}))
        obs, _, term, trunc, info = env.step(ctl(obs))
        done = term or trunc
    m = info["episode_metrics"]
    out = {"seed": seed, "cleared": m["completed_stage"], "t": m["survival_time"]}
    if m["completed_stage"]:
        return out
    kill = [e for _, e in events if e.kind == "ship_destroyed" and e.entity_id == env.agent_id]
    if not kill:
        out["cause"] = m["terminal_reason"]; return out
    kid = int(kill[0].detail)
    born = {int(e.entity_id): (step, e.kind, e.detail) for step, e in events
            if e.kind in ("asteroid_spawned", "asteroid_split") and e.entity_id.isdigit()}
    fps = 60.0
    death_step = [s for s, e in events if e is kill[0]][0]
    if kid in born:
        bstep, kind, detail = born[kid]
        out["killer_age"] = (death_step - bstep) / fps
        out["killer_origin"] = "fragment" if kind == "asteroid_split" else "spawn"
        if kind == "asteroid_split":
            shooter = [e.detail for s, e in events if e.kind == "asteroid_shot" and e.entity_id == detail]
            out["parent_shot_by_agent"] = bool(shooter) and shooter[0] == env.agent_id
    else:
        out["killer_age"] = m["survival_time"]; out["killer_origin"] = "initial"
    # was the killer visible & where was it 1s before death?
    W = H = 900.0
    for back in (15, 8, 4):
        if len(snaps) > back:
            el, (sx, sy), rocks = snaps[-back]
            if kid in rocks:
                x, y, r, vx, vy = rocks[kid]
                d = wrapped_delta(Vec2(sx, sy), Vec2(x, y), W, H)
                dist = d.length()
                rank = sorted(wrapped_delta(Vec2(sx, sy), Vec2(a[0], a[1]), W, H).length() for a in rocks.values()).index(dist)
                out[f"dist_{back}"] = dist - r
                out[f"rank_{back}"] = rank
                out[f"speed_{back}"] = math.hypot(vx, vy)
            else:
                out[f"dist_{back}"] = None
    out["rocks_at_death"] = len(snaps[-1][2])
    return out

if __name__ == "__main__":
    seeds = list(range(10000, 10000 + int(sys.argv[2])))
    with ProcessPoolExecutor(max_workers=int(sys.argv[3])) as pool:
        rows = list(pool.map(run, seeds))
    deaths = [r for r in rows if not r["cleared"]]
    print(f"clear {statistics.fmean(r['cleared'] for r in rows):.3f} over {len(rows)}; deaths {len(deaths)}")
    print("origin:", Counter(r.get("killer_origin") for r in deaths))
    print("parent shot by agent:", Counter(r.get("parent_shot_by_agent") for r in deaths if r.get("killer_origin") == "fragment"))
    ages = sorted(r["killer_age"] for r in deaths if "killer_age" in r)
    print("killer age (s): ", [round(a, 2) for a in ages])
    print("killer age <1s: %d  <2s: %d  of %d" % (sum(a < 1 for a in ages), sum(a < 2 for a in ages), len(ages)))
    for back in (15, 8, 4):
        vis = [r for r in deaths if r.get(f"dist_{back}") is not None]
        print(f"{back} decisions ({back/15:.2f}s) before death: killer existed in {len(vis)}/{len(deaths)}; "
              f"median gap {statistics.median(r[f'dist_{back}'] for r in vis) if vis else float('nan'):.0f}px, "
              f"median nearest-rank {statistics.median(r[f'rank_{back}'] for r in vis) if vis else float('nan')}")
    print("rocks at death median", statistics.median(r["rocks_at_death"] for r in deaths))
    json.dump(rows, open(sys.argv[5] if len(sys.argv) > 5 else "metrics/death-causes.json", "w"), indent=1)
