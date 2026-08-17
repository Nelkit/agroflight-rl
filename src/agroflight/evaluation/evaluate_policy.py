"""
Compares a TRAINED policy against the hand-coded baselines (altitude-hold,
random). This comparison -- not the raw training reward curve -- is the actual
evidence for the experimental-results section: does the learned policy beat a
naive baseline on behaviour metrics, not just "reward went up"?

Left as a skeleton on purpose: TODO markers show where you plug in your trained
checkpoint and where you decide the evaluation protocol (episode count, seeds,
success thresholds).

Run:
    python -m agroflight.evaluation.evaluate_policy
"""
import os

import gymnasium as gym
import numpy as np

import agroflight.envs  # noqa: F401  (registers AgroDrone-v0)
from agroflight.baselines import altitude_hold_action, random_action
from agroflight.envs import ENV_ID

# ---------------------------------------------------------------------------
# TODO (team): DECIDE YOUR EVALUATION PROTOCOL.
# How many episodes is enough to make a claim? Which seeds (and are they held
# out from training)? What counts as "success" -- a crash-rate threshold, a
# mean-reward threshold, a mean absolute altitude error in metres? Pick these
# BEFORE you look at the trained results, and justify them in the report.
# These are placeholders, not answers.
# ---------------------------------------------------------------------------
DEFAULT_N_EPISODES = 20   # TODO: justify this number
DEFAULT_SEEDS = None      # TODO: None -> range(n_episodes). Decide on a held-out seed set.


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


def evaluate(policy_fn, n_episodes=DEFAULT_N_EPISODES, seeds=DEFAULT_SEEDS, xml_path=None, env=None):
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


def _format(name, stats):
    return (
        f"  {name:<14} mean_reward={stats['mean_reward']:8.2f} "
        f"(±{stats['std_reward']:6.2f})  crash_rate={stats['crash_rate']:.0%}  "
        f"finish_rate={stats['finish_rate']:.0%}  mean|alt_err|={stats['mean_abs_alt_error']:.3f}m"
    )


def main():
    env = make_env()
    rng = np.random.default_rng(0)

    print(f"Baselines over {DEFAULT_N_EPISODES} episodes ({ENV_ID}):")
    print(_format("altitude-hold", evaluate(altitude_hold_action, env=env)))
    print(_format(
        "random",
        evaluate(lambda obs: random_action(env.action_space, rng), env=env),
    ))
    env.close()

    # -----------------------------------------------------------------------
    # TODO (team): load your trained checkpoint and evaluate it the same way.
    # The three-way comparison (learned vs altitude-hold vs random) is your
    # result. Example shape -- match whatever API your saved model exposes:
    #
    #   trained_agent = load_your_checkpoint("runs/AgroDrone-v0__ppo/agent.pt")
    #   def trained_policy(obs):
    #       return trained_agent.act(obs)
    #   print(_format("trained PPO", evaluate(trained_policy)))
    # -----------------------------------------------------------------------
    print()
    print("TODO: add your trained policy above once you have a checkpoint, then")
    print("      compare mean_reward / crash_rate / finish_rate against these baselines.")


if __name__ == "__main__":
    main()
