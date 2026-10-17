"""
REINFORCE (Monte-Carlo policy gradient) for AgroDrone-v0, as a restartable batch script.

Same algorithm and hyperparameters as notebooks/reinforce_25418551.ipynb, but every run is
saved to disk as it finishes, so nothing is ever retrained by accident:

    checkpoints/reinforce/L{n}_seed{s}/step_0100k.pt ... final.pt, best.pt
    results/reinforce/runs/L{n}_seed{s}.json   (hparams, wall-clock, training curve,
                                                evaluation of every checkpoint)

A run is skipped when both its final.pt and its run JSON already exist.

Run (from the repo root, with the venv active):
    python -m agroflight.training.train_reinforce --levels 0 1 2 3 4 --seeds 0
    python -m agroflight.training.train_reinforce --levels 0 1 --seeds 1 2 --workers 4
"""
import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

import agroflight.envs  # noqa: F401  (registers AgroDrone-v0)
from agroflight.envs import ENV_ID

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ.setdefault("AGRO_DRONE_XML", os.path.join(REPO, "mujoco_menagerie", "skydio_x2", "x2.xml"))

CKPT_ROOT = os.path.join(REPO, "checkpoints", "reinforce")
RUNS_DIR = os.path.join(REPO, "results", "reinforce", "runs")
EVAL_SEEDS = list(range(1000, 1020))

HPARAMS = dict(
    learning_rate=3e-4,
    gamma=0.99,
    normalize_returns=False,
    ent_coef=0.0,
    hidden=(64, 64),
    init_log_std=-2.3,
    episodes_per_update=1,
)


class GaussianPolicy(nn.Module):
    """obs -> Gaussian over the action dimensions (mean network + learnable log-std)."""

    def __init__(self, obs_dim, act_dim, hidden=HPARAMS["hidden"]):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        layers.append(nn.Linear(last, act_dim))
        self.mean_net = nn.Sequential(*layers)
        self.mean_net[-1].weight.data *= 0.01
        self.mean_net[-1].bias.data.zero_()
        self.log_std = nn.Parameter(torch.full((act_dim,), float(HPARAMS["init_log_std"])))

    def forward(self, obs):
        return Normal(self.mean_net(obs), self.log_std.exp())


def compute_returns(rewards, gamma):
    returns, running = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return torch.tensor(returns, dtype=torch.float32)


def run_episode_train(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    log_probs, rewards = [], []
    while True:
        dist = policy(torch.as_tensor(obs, dtype=torch.float32))
        action = dist.sample()
        log_probs.append(dist.log_prob(action).sum())
        obs, reward, terminated, truncated, _ = env.step(action.numpy())
        rewards.append(reward)
        if terminated or truncated:
            return log_probs, rewards


def save_ckpt(path, policy, obs_dim, act_dim, level, seed, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(algorithm="REINFORCE", level=level, train_seed=seed, timesteps=step, hparams=HPARAMS,
                    obs_dim=obs_dim, act_dim=act_dim, state_dict=policy.state_dict()), path)


def evaluate_checkpoint(path, level):
    from agroflight.evaluation.evaluate_policy import evaluate
    ckpt = torch.load(path, map_location="cpu")
    model = GaussianPolicy(ckpt["obs_dim"], ckpt["act_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def policy_fn(obs):
        with torch.no_grad():
            return model(torch.as_tensor(obs, dtype=torch.float32)).mean.numpy()

    env = gym.make(ENV_ID, level=level)
    try:
        return evaluate(policy_fn, seeds=EVAL_SEEDS, env=env)
    finally:
        env.close()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def run_one(level, seed, total_steps, ckpt_every):
    """Train + evaluate one (level, seed). Returns the run name, or None if it was skipped."""
    torch.set_num_threads(1)
    name = f"L{level}_seed{seed}"
    ckpt_dir = os.path.join(CKPT_ROOT, name)
    run_json = os.path.join(RUNS_DIR, f"{name}.json")
    if os.path.isfile(os.path.join(ckpt_dir, "final.pt")) and os.path.isfile(run_json):
        print(f"[{name}] exists, skipped", flush=True)
        return None

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = gym.make(ENV_ID, level=level)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=HPARAMS["learning_rate"])

    curve, step, episode_idx = [], 0, 0
    best_avg, recent = -np.inf, []
    next_ckpt, saved = ckpt_every, []
    t0 = time.time()
    while step < total_steps:
        lp, rew = run_episode_train(env, policy, seed=seed * 100_000 + episode_idx)  # never 1000-1019
        episode_idx += 1
        step += len(rew)
        returns = compute_returns(rew, HPARAMS["gamma"])
        if HPARAMS["normalize_returns"]:   # ablation: standardise returns per episode (variance reduction)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -(torch.stack(lp) * returns).sum() - HPARAMS["ent_coef"] * policy.log_std.sum()
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        ret = float(sum(rew))
        curve.append({"timesteps": step, "mean_return": ret})
        recent = (recent + [ret])[-5:]
        if len(recent) == 5 and np.mean(recent) > best_avg:
            best_avg = float(np.mean(recent))
            save_ckpt(os.path.join(ckpt_dir, "best.pt"), policy, obs_dim, act_dim, level, seed, step)
        if step >= next_ckpt and step < total_steps:
            p = os.path.join(ckpt_dir, f"step_{next_ckpt // 1000:04d}k.pt")
            save_ckpt(p, policy, obs_dim, act_dim, level, seed, step)
            saved.append((next_ckpt, p))
            next_ckpt += ckpt_every
            print(f"[{name}] step {step:>8}  return(last5) {np.mean(recent):8.2f}", flush=True)
    env.close()
    train_s = time.time() - t0

    final_p = os.path.join(ckpt_dir, "final.pt")
    save_ckpt(final_p, policy, obs_dim, act_dim, level, seed, step)
    saved.append((step, final_p))

    # Evaluate every saved checkpoint (learning curve over training) + the best-by-training one.
    ckpt_eval = {os.path.basename(p): dict(timesteps=s, **evaluate_checkpoint(p, level)) for s, p in saved}
    best_p = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(best_p):
        ckpt_eval["best.pt"] = dict(timesteps=torch.load(best_p, map_location="cpu")["timesteps"],
                                    **evaluate_checkpoint(best_p, level))

    os.makedirs(RUNS_DIR, exist_ok=True)
    out = dict(
        schema=1, algorithm="REINFORCE", env_id=ENV_ID, level=level, train_seed=seed,
        git_commit=git_commit(), created=datetime.datetime.now().isoformat(timespec="seconds"),
        hparams=dict(HPARAMS, total_timesteps=total_steps, checkpoint_every=ckpt_every),
        training=dict(total_timesteps=step, wall_clock_s=train_s, best_train_avg_return=best_avg, curve=curve),
        eval_seeds=EVAL_SEEDS, checkpoint_eval=ckpt_eval,
    )
    with open(run_json, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    fin = ckpt_eval["final.pt"]
    print(f"[{name}] DONE {step} steps in {train_s:.0f}s | final: reward {fin['mean_reward']:.1f} "
          f"crash {fin['crash_rate']:.0%} finish {fin['finish_rate']:.0%} "
          f"|alt err| {fin['mean_abs_alt_error']:.3f}", flush=True)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--ckpt-every", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--normalize-returns", action="store_true",
                    help="ablation: standardise returns; saves under checkpoints|results/reinforce_normret")
    args = ap.parse_args()
    if args.normalize_returns:
        global CKPT_ROOT, RUNS_DIR
        HPARAMS["normalize_returns"] = True
        CKPT_ROOT = os.path.join(REPO, "checkpoints", "reinforce_normret")
        RUNS_DIR = os.path.join(REPO, "results", "reinforce_normret", "runs")

    jobs = [(lv, sd) for sd in args.seeds for lv in args.levels]
    print(f"{len(jobs)} runs, {args.steps:,} steps each, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futs = [pool.submit(run_one, lv, sd, args.steps, args.ckpt_every) for lv, sd in jobs]
        for f in as_completed(futs):
            f.result()


if __name__ == "__main__":
    main()
