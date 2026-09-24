"""
Environment package. Importing `agroflight.envs` registers "AgroDrone-v0" with
Gymnasium, so any consumer (CleanRL, Stable-Baselines3, the evaluation module)
can just call `gymnasium.make("AgroDrone-v0")`.

Point AGRO_DRONE_XML at your local menagerie clone before creating the env,
or pass `xml_path=` explicitly to `gym.make`.
"""
from gymnasium.envs.registration import register, registry

from agroflight.envs.agro_drone_env import (
    ACTION_SCALE,
    OBS_DIM,
    OBS_IDX_VZ,
    OBS_LAYOUT,
    AgroDroneEnv,
)
from agroflight.envs.scene_builder import build_plantation_scene

ENV_ID = "AgroDrone-v0"

# The entry point is the packaged module path (was a bare `agro_drone_gym_env`
# module name in the prototype, which only resolved when the script's own
# folder happened to be on sys.path).
#
# Guarded so that re-importing the package -- which Gymnasium's vector envs and
# pytest collection both do -- does not raise on duplicate registration.
if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="agroflight.envs.agro_drone_env:AgroDroneEnv",
        # NOTE: AgroDroneEnv already truncates itself at `max_steps`. This outer
        # TimeLimit is the value CleanRL's rollout length is matched against.
        # TODO (team): keep this in sync with `num_steps` in the training config
        # if you change the episode length.
        max_episode_steps=1000,
    )

__all__ = [
    "AgroDroneEnv",
    "build_plantation_scene",
    "ENV_ID",
    "ACTION_SCALE",
    "OBS_DIM",
    "OBS_IDX_VZ",
    "OBS_LAYOUT",
]
