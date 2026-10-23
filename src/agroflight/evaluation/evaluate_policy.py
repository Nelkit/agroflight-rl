"""
Shared evaluation harness: rolls out a policy on fixed seeds and reports
behaviour metrics (reward, crash rate, finish rate, steps, altitude error).
Every notebook and script evaluates through `evaluate()` so the three
algorithms and the baselines are measured the same way.

    from agroflight.evaluation.evaluate_policy import evaluate
    stats = evaluate(policy_fn, seeds=range(1000, 1020), env=env)
"""
import os

import gymnasium as gym
import numpy as np

import agroflight.envs  # noqa: F401  (registers AgroDrone-v0)
from agroflight.envs import ENV_ID

DEFAULT_N_EPISODES = 20   # used only when `seeds` is not given


def run_episode(env, policy_fn, seed):
    """Roll out one episode and return its summary metrics."""
    obs, info = env.reset(seed=seed)
    total_reward, steps = 0.0, 0
    crashed, finished_row = False, False
    alt_errors = []

    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if "alt_error" in info:
            alt_errors.append(abs(info["alt_error"]))
        if terminated or truncated:
            # BUGFIX: the prototype read `info["finished"]` in the viewer but
            # `info["finished_row"]` in the env, so the evaluation script's
            # `finished` metric silently read False forever. The env's key is
            # `finished_row`; that is now the single spelling everywhere.
            crashed = bool(info.get("crashed", False))
            finished_row = bool(info.get("finished_row", False))
            break

    return {
        "reward": total_reward,
        "steps": steps,
        "crashed": crashed,
        "finished_row": finished_row,
        "mean_abs_alt_error": float(np.mean(alt_errors)) if alt_errors else float("nan"),
    }


def evaluate(policy_fn, n_episodes=DEFAULT_N_EPISODES, seeds=None, xml_path=None, env=None):
    """Run `policy_fn` for n_episodes and return summary stats.

    `policy_fn` takes an observation and returns an action.
    Pass `env` to reuse an existing env (avoids reloading the MuJoCo model).
    """
    owns_env = env is None
    if owns_env:
        env = make_env(xml_path)
    try:
        episode_seeds = list(seeds) if seeds is not None else list(range(n_episodes))
        results = [run_episode(env, policy_fn, seed=s) for s in episode_seeds]
    finally:
        if owns_env:
            env.close()

    rewards = [r["reward"] for r in results]
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "crash_rate": float(np.mean([r["crashed"] for r in results])),
        "finish_rate": float(np.mean([r["finished_row"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_abs_alt_error": float(np.nanmean([r["mean_abs_alt_error"] for r in results])),
        "n_episodes": len(results),
    }


def make_env(xml_path=None):
    """Create the evaluation env, resolving the model path up front."""
    xml_path = xml_path or os.environ.get("AGRO_DRONE_XML")
    if xml_path is None:
        raise SystemExit(
            "Set AGRO_DRONE_XML (or pass xml_path) before evaluating, e.g.:\n"
            "  export AGRO_DRONE_XML=$(pwd)/mujoco_menagerie/skydio_x2/x2.xml"
        )
    return gym.make(ENV_ID, xml_path=xml_path)

