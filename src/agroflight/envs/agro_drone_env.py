"""
AgroDroneEnv -- Gymnasium environment for the crop-row flight-control task.

Physics: MuJoCo Skydio X2 quadrotor (Google DeepMind mujoco_menagerie,
Apache-2.0, https://github.com/google-deepmind/mujoco_menagerie).

Task: the agent flies the drone at a target altitude directly above a crop row
(the world x-axis), while a slowly-varying random wind gust pushes it sideways,
and makes progress down the row.

Observation: 14-dim. Action: 4 normalised high-level commands in [-1, 1]
(forward velocity, lateral velocity, climb rate, yaw rate). A fixed inner
controller (inner_controller.py) turns them into rotor thrusts, so the policy
never has to learn to hover from raw motor commands (Part B design).

The reward is still the ILLUSTRATIVE placeholder from the exploration
prototype and has NOT been validated for training. See the TODO block in step().
"""
import os
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from agroflight.envs.inner_controller import InnerController

# Body name of the drone's free-floating root in the Skydio X2 model. Used to
# resolve the body id for wind forces instead of assuming an index.
DRONE_BODY_NAME = "x2"

# Observation layout, exposed so tests/consumers don't hardcode magic indices.
OBS_LAYOUT = (
    ("alt_error", 1),
    ("lateral_offset", 1),
    ("quat", 4),
    ("linvel", 3),
    ("angvel", 3),
    ("wind", 2),
)
OBS_DIM = sum(size for _, size in OBS_LAYOUT)  # 14

# Index of the vertical linear-velocity component inside the observation
# vector (alt_error, lateral_offset, quat[4] -> linvel starts at 6, vz at 8).
OBS_IDX_VZ = 8

# Normalised action in [-1, 1] -> physical command for the inner controller:
# forward velocity (m/s), lateral velocity (m/s), climb rate (m/s), yaw rate (rad/s).
ACTION_SCALE = np.array([2.0, 1.0, 1.0, 1.0])


@dataclass
class RewardConfig:
    """Reward v1 weights, all configurable for sensitivity analysis (Part B).

    reward = -(w_alt*alt_err^2 + w_lat*lateral_err^2 + w_tilt*tilt^2 + w_action*|action|^2)
             + w_progress*(metres gained along the row) + survival_bonus
             - crash_penalty (on crash) + completion_bonus (on finishing the row)
    """
    w_alt: float = 2.0
    w_lat: float = 2.0
    w_tilt: float = 0.5
    w_action: float = 0.01
    w_progress: float = 10.0
    survival_bonus: float = 0.02
    crash_penalty: float = 50.0
    completion_bonus: float = 20.0


# Difficulty levels, fixed in advance (Part B) and never adjusted per algorithm.
#   wind:        "none", "steady" (constant sideways wind per episode) or "gusting" (random walk)
#   spawn_offset: max random sideways start offset in metres
#   row_task:    progress reward and row-finish condition on (off = altitude hold only)
#   yaw_free:    the policy may command yaw rate
LEVELS = {
    0: dict(wind="none", spawn_offset=0.0, row_task=False, yaw_free=False),
    1: dict(wind="none", spawn_offset=0.15, row_task=True, yaw_free=False),
    2: dict(wind="steady", spawn_offset=0.15, row_task=True, yaw_free=False),
    3: dict(wind="gusting", spawn_offset=0.15, row_task=True, yaw_free=False),
    4: dict(wind="gusting", spawn_offset=0.15, row_task=True, yaw_free=True),
}
DEFAULT_LEVEL = 3


def _resolve_xml_path(xml_path):
    """Resolve the scene XML path, falling back to the AGRO_DRONE_XML env var."""
    if xml_path is not None:
        return str(xml_path)
    from_env = os.environ.get("AGRO_DRONE_XML")
    if from_env is None:
        raise ValueError(
            "No xml_path given and AGRO_DRONE_XML env var not set. "
            "Point it at your cloned mujoco_menagerie/skydio_x2/x2.xml, or at "
            "the plantation scene produced by agroflight.envs.scene_builder. "
            "Run `make menagerie` to fetch the model."
        )
    return from_env


class AgroDroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, xml_path=None, target_altitude=1.2, field_length=8.0,
                 wind_strength=0.6, max_steps=1000, seed=None,
                 frame_skip=1, yaw_free=None, level=DEFAULT_LEVEL, reward_config=None,
                 max_alt_error=1.0, max_lateral_offset=1.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(_resolve_xml_path(xml_path))
        self.data = mujoco.MjData(self.model)

        self.target_altitude = target_altitude
        self.field_length = field_length
        self.wind_strength = wind_strength
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        # Flight envelope around the row (metres). Leaving it ends the episode as a
        # crash. Without it nothing stops the drone drifting away: random horizontal
        # commands make it climb ~10 m in 1000 steps, and the quadratic alt/lateral
        # penalties then grow without bound (returns of order -50k per episode).
        # None disables a bound.
        self.max_alt_error = max_alt_error
        self.max_lateral_offset = max_lateral_offset

        if level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
        spec = LEVELS[level]
        self.level = level
        self._wind_mode = spec["wind"]
        self._spawn_offset = spec["spawn_offset"]
        self._row_task = spec["row_task"]
        self.yaw_free = spec["yaw_free"] if yaw_free is None else yaw_free

        if reward_config is None:
            reward_config = RewardConfig()
        elif isinstance(reward_config, dict):
            reward_config = RewardConfig(**reward_config)
        self.reward_config = reward_config

        # BUGFIX (was `xfrc_applied[1]`): the prototype hardcoded body index 1
        # as the drone. That only holds when the drone body happens to be the
        # first body after the world body -- it silently applies the wind force
        # to the WRONG body as soon as the XML gains bodies before it. Resolve
        # the id by name instead, and fall back to 1 only if the name is absent
        # (e.g. a stripped-down test model).
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, DRONE_BODY_NAME)
        self._drone_body_id = body_id if body_id != -1 else 1
        self.controller = InnerController(self.model, self._drone_body_id)

        # hover thrust per rotor, read straight from the model's own "hover" keyframe
        self.hover_ctrl = self.model.key_ctrl[0].copy()
        self.ctrl_range = self.model.actuator_ctrlrange.copy()  # shape (4, 2)

        # action = normalised [forward vel, lateral vel, climb rate, yaw rate]; 0 -> hold position
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # observation: [alt_error, lateral_offset, quat(4), linvel(3), angvel(3), wind(2)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._wind = np.zeros(2)

    # ------------------------------------------------------------------
    def _get_obs(self):
        pos = self.data.qpos[0:3]
        quat = self.data.qpos[3:7]
        linvel = self.data.qvel[0:3]
        angvel = self.data.qvel[3:6]
        alt_error = pos[2] - self.target_altitude
        lateral_offset = pos[1]  # crop row centerline is y=0
        return np.concatenate(
            [[alt_error, lateral_offset], quat, linvel, angvel, self._wind]
        ).astype(np.float32)

    def _apply_wind(self):
        # gusting: slowly-varying random lateral/vertical wind (random walk).
        # steady / none: the wind was fixed once in reset(). Applied as a
        # translational force on the drone body -- the disturbance the policy must reject.
        if self._wind_mode == "gusting":
            self._wind += self._rng.normal(0, 0.05, size=2)
            self._wind = np.clip(self._wind, -self.wind_strength, self.wind_strength)
        self.data.xfrc_applied[self._drone_body_id, 0] = 0.0
        self.data.xfrc_applied[self._drone_body_id, 1] = self._wind[0]        # lateral (y)
        self.data.xfrc_applied[self._drone_body_id, 2] = self._wind[1] * 0.3  # small vertical gust

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # BUGFIX: the prototype called `super().reset(seed=seed)` (which seeds
        # Gymnasium's `self.np_random`) but then drew the start offset and the
        # wind from `self._rng`, created once in __init__ and never reseeded.
        # So `env.reset(seed=42)` did NOT give a reproducible episode -- which
        # silently breaks the per-seed evaluation loop in the evaluation module.
        # Reseed the env's own RNG on an explicit seed.
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # Clear any wind force left over from the previous episode; MuJoCo does
        # not reset xfrc_applied on a keyframe reset, so a terminal gust would
        # otherwise leak into the first step of the next episode.
        self.data.xfrc_applied[:] = 0.0

        # small random start offset so every episode is slightly different
        self.data.qpos[1] += self._rng.uniform(-self._spawn_offset, self._spawn_offset)
        self.data.qpos[2] = self.target_altitude + self._rng.uniform(-0.1, 0.1)
        self._steps = 0
        self._wind = np.zeros(2)
        if self._wind_mode == "steady":
            sign = self._rng.choice([-1.0, 1.0])
            self._wind[0] = sign * self._rng.uniform(0.5, 1.0) * self.wind_strength
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        cmd = action * ACTION_SCALE

        prev_x = float(self.data.qpos[0])
        self._apply_wind()
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.controller.thrusts(
                self.data.qpos, self.data.qvel, cmd, self.yaw_free)
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        pos = self.data.qpos[0:3]
        quat = self.data.qpos[3:7]
        alt_error = pos[2] - self.target_altitude
        lateral_offset = pos[1]

        # tilt angle from quaternion (0 = level, pi/2 = sideways)
        up_z = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
        tilt = np.arccos(np.clip(up_z, -1.0, 1.0))

        # Reward v1 (Part B). The weights in RewardConfig are first-pass defaults,
        # NOT tuned: iterate on them and watch actual flight in the viewer for
        # reward hacking, not only the reward curve.
        cfg = self.reward_config
        reward = (
            -cfg.w_alt * alt_error ** 2
            - cfg.w_lat * lateral_offset ** 2
            - cfg.w_tilt * tilt ** 2
            - cfg.w_action * np.sum(np.square(action))
            + cfg.survival_bonus
        )
        if self._row_task:
            reward += cfg.w_progress * (float(pos[0]) - prev_x)

        out_of_bounds = bool(
            (self.max_alt_error is not None and abs(alt_error) > self.max_alt_error)
            or (self.max_lateral_offset is not None and abs(lateral_offset) > self.max_lateral_offset)
        )
        # hit ground, flipped over, or left the flight envelope (counted as a crash)
        crashed = bool(pos[2] < 0.08 or tilt > 1.2 or out_of_bounds)
        finished_row = bool(self._row_task and pos[0] >= self.field_length)
        truncated = self._steps >= self.max_steps
        terminated = crashed or finished_row

        if crashed:
            reward -= cfg.crash_penalty
        if finished_row:
            reward += cfg.completion_bonus

        info = {
            "crashed": crashed,
            "out_of_bounds": out_of_bounds,
            "finished_row": finished_row,
            "alt_error": float(alt_error),
            "lateral_offset": float(lateral_offset),
            "x": float(pos[0]),
        }
        return self._get_obs(), float(reward), terminated, truncated, info
