"""
Training entry point -- wires AgroDrone-v0 into CleanRL's existing PPO
implementation instead of reimplementing PPO from scratch. CleanRL
(https://github.com/vwxyzjn/cleanrl) is the reference implementation this
project builds on; this script deliberately does NOT contain PPO's update
logic -- writing/understanding that is the team's learning objective.

This is intentionally left UNTUNED. Running it as-is will train *something*,
but there is no guarantee it flies well. Tuning hyperparameters, iterating on
the reward function in agroflight/envs/agro_drone_env.py, and choosing a
training budget IS the project work.

--- Setup (one time) ---
    make setup && make menagerie
    export AGRO_DRONE_XML=$(pwd)/mujoco_menagerie/skydio_x2/x2.xml

--- Run ---
    python -m agroflight.training.train_ppo_cleanrl

--- What to expect / TODO for the team ---
  1. This calls CleanRL's ppo_continuous_action under the hood. Read that
     file (one file, ~300 lines, written to be read) to understand what PPO
     is actually doing -- that is the point of the assignment.
  2. Track training with TensorBoard (`make tensorboard`) to see the reward
     curve. If it is flat or diverging, that is a signal to revisit the
     reward weights in the env, not this script.
  3. Every hyperparameter below is a placeholder, not a tuned setting.
  4. Once you have a checkpoint, plug it into the evaluation module and
     compare it against the altitude-hold / random baselines -- that
     comparison is the result, not the training curve on its own.
"""
import os

# Importing the package registers "AgroDrone-v0" before CleanRL gym.make()s it.
import agroflight.envs  # noqa: F401


# ---------------------------------------------------------------------------
# TODO (team): EVERY VALUE HERE IS A STARTING GUESS, NOT A TUNED SETTING.
# Choosing these -- and justifying the choices in the report -- is the project
# work. Expect several rounds of trial and error.
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMS = dict(
    env_id="AgroDrone-v0",
    total_timesteps=200_000,   # TODO: placeholder to confirm the pipeline runs end-to-end.
                               #       Scale up once it does (tasks of this scale often need ~1M+).
    learning_rate=3e-4,        # TODO: tune
    num_envs=4,                # TODO: parallel rollout workers -- tune for your machine
    num_steps=500,             # TODO: keep in sync with max_episode_steps in agroflight/envs/__init__.py
    gamma=0.99,                # TODO: tune
    gae_lambda=0.95,           # TODO: tune
    ent_coef=0.0,              # TODO: consider > 0 to encourage exploration early on
    track=False,               # set True + wandb if you want experiment tracking
)


def build_args(**overrides):
    """Construct CleanRL's Args dataclass from DEFAULT_HYPERPARAMS.

    Unknown keys are dropped with a warning rather than raising: CleanRL's Args
    fields shift between releases, and a scaffold that hard-crashes on a
    renamed field is worse than one that tells you which field went away.
    """
    from cleanrl.ppo_continuous_action import Args

    params = {**DEFAULT_HYPERPARAMS, **overrides}
    valid = set(getattr(Args, "__dataclass_fields__", {}))
    if valid:
        unknown = sorted(set(params) - valid)
        for key in unknown:
            print(
                f"[warn] '{key}' is not a field of your installed CleanRL Args "
                f"-- dropping it. Check the CleanRL version's ppo_continuous_action.py."
            )
            params.pop(key)
    return Args(**params)


def main():
    if "AGRO_DRONE_XML" not in os.environ:
        raise SystemExit(
            "Set AGRO_DRONE_XML to your cloned skydio_x2/x2.xml (or the plantation "
            "scene) before running, e.g.:\n"
            "  export AGRO_DRONE_XML=$(pwd)/mujoco_menagerie/skydio_x2/x2.xml\n"
            "(Run `make menagerie` first if you have not cloned the model yet.)"
        )

    try:
        from cleanrl.ppo_continuous_action import train
    except ImportError as exc:
        raise SystemExit(
            f"Could not import CleanRL's PPO ({exc}).\n"
            'Install it with: pip install "cleanrl[mujoco]"\n'
            "If the import path does not match your installed CleanRL version, check\n"
            "https://github.com/vwxyzjn/cleanrl -- APIs shift between releases. Worst\n"
            "case, vendor ppo_continuous_action.py from their repo and set\n"
            "env_id='AgroDrone-v0'."
        )

    train(build_args())


if __name__ == "__main__":
    main()
