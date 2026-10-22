"""
Assemble results/reinforce_25418551.json (the shared cell 2.3 schema) from the per-run files
written by agroflight.training.train_reinforce. Training is NOT repeated: this only reads
results/reinforce/runs/*.json and re-runs the cheap hand-coded baselines.

    python scripts/assemble_reinforce_results.py

Conventions in the output:
  evaluation.levels[Ln]      mean over the training seeds of the final.pt evaluation
                             (extra key `n_seeds`; per-seed values are in evaluation.per_seed)
  evaluation.per_seed[Ln_seedS]  final.pt evaluation of that run
  evaluation.best_ckpt[Ln_seedS] best.pt (best by training return) evaluation of that run
  training.runs[Ln_seedS]    hparams-level info + learning curve + checkpoint_eval learning curve
  training.curve             L0 seed 0 curve (the shared schema's single main curve)
"""
import datetime
import glob
import json
import os
import subprocess
import sys

import gymnasium as gym
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("AGRO_DRONE_XML", os.path.join(REPO, "mujoco_menagerie", "skydio_x2", "x2.xml"))

import agroflight.envs  # noqa: E402,F401
from agroflight.baselines import altitude_hold_action, random_action  # noqa: E402
from agroflight.envs import ENV_ID  # noqa: E402
from agroflight.evaluation.evaluate_policy import evaluate  # noqa: E402

RUNS_DIR = os.path.join(REPO, "results", "reinforce", "runs")
OUT = os.path.join(REPO, "results", "reinforce_25418551.json")
STAT_KEYS = ["mean_reward", "std_reward", "crash_rate", "finish_rate", "mean_steps", "mean_abs_alt_error"]


def mean_stats(stats):
    out = {k: float(np.mean([s[k] for s in stats])) for k in STAT_KEYS}
    out["n_episodes"] = int(sum(s["n_episodes"] for s in stats))
    out["n_seeds"] = len(stats)
    return out


def main():
    runs = {}
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "L*_seed*.json"))):
        d = json.load(open(p))
        runs[f"L{d['level']}_seed{d['train_seed']}"] = d
    levels = sorted({d["level"] for d in runs.values()})
    eval_seeds = next(iter(runs.values()))["eval_seeds"]
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()

    per_seed = {n: {k: d["checkpoint_eval"]["final.pt"][k] for k in STAT_KEYS + ["n_episodes"]} for n, d in runs.items()}
    best_ckpt = {n: {k: d["checkpoint_eval"]["best.pt"][k] for k in STAT_KEYS + ["n_episodes"]} for n, d in runs.items()}
    level_stats = {f"L{lv}": mean_stats([per_seed[n] for n, d in runs.items() if d["level"] == lv]) for lv in levels}

    # Hand-coded baselines at every level (same eval seeds), so the comparison uses the current reward.
    baselines = {}
    rng = np.random.default_rng(0)
    for lv in levels:
        env = gym.make(ENV_ID, level=lv)
        baselines[f"zero_action@L{lv}"] = evaluate(lambda obs: np.zeros(4, dtype=np.float32), seeds=eval_seeds, env=env)
        baselines[f"altitude_hold@L{lv}"] = evaluate(altitude_hold_action, seeds=eval_seeds, env=env)
        baselines[f"random@L{lv}"] = evaluate(lambda obs: random_action(env.action_space, rng), seeds=eval_seeds, env=env)
        env.close()
    baselines["altitude_hold"] = baselines["altitude_hold@L0"]
    baselines["random"] = baselines["random@L0"]

    any_run = next(iter(runs.values()))
    main_run = runs.get("L0_seed0", any_run)
    results = dict(
        schema=1,
        algorithm="REINFORCE",
        student=dict(name="Juan David Rodriguez Arciniegas", id="25418551"),
        env_id=ENV_ID,
        git_commit=commit,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        hparams=any_run["hparams"],
        training=dict(
            train_seed=0,
            total_timesteps=main_run["training"]["total_timesteps"],
            wall_clock_s=main_run["training"]["wall_clock_s"],
            best_checkpoint="checkpoints/reinforce/L0_seed0/best.pt",
            curve=main_run["training"]["curve"],
            runs={n: dict(level=d["level"], train_seed=d["train_seed"], git_commit=d["git_commit"],
                          total_timesteps=d["training"]["total_timesteps"],
                          wall_clock_s=d["training"]["wall_clock_s"],
                          best_train_avg_return=d["training"]["best_train_avg_return"],
                          checkpoint_dir=f"checkpoints/reinforce/{n}",
                          curve=d["training"]["curve"], checkpoint_eval=d["checkpoint_eval"])
                  for n, d in runs.items()},
        ),
        evaluation=dict(eval_seeds=eval_seeds, baselines=baselines, levels=level_stats,
                        per_seed=per_seed, best_ckpt=best_ckpt),
    )
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", OUT, f"({os.path.getsize(OUT) / 1e6:.1f} MB)")
    for lv in levels:
        s = level_stats[f"L{lv}"]
        print(f"L{lv}: reward {s['mean_reward']:7.1f}  crash {s['crash_rate']:.0%}  finish {s['finish_rate']:.0%}  "
              f"alt {s['mean_abs_alt_error']:.3f}   | altitude_hold {baselines[f'altitude_hold@L{lv}']['mean_reward']:.1f}")


if __name__ == "__main__":
    main()
