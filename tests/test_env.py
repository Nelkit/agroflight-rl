"""
SANITY tests for AgroDroneEnv -- API compliance and numerical health only.

Deliberately out of scope: anything that asserts a policy flies *well*. There
is no trained policy yet, and the reward function is untuned by design. Tests
that encode "good behaviour" thresholds would be asserting the answer to the
assignment. Performance claims belong in the evaluation module and the report.

These tests need the Skydio X2 model. Run `make menagerie` first, or set
AGRO_DRONE_XML. Without it the whole module skips rather than fails.
"""
import os

import gymnasium as gym
import numpy as np
import pytest

import agroflight.envs  # noqa: F401  (registers AgroDrone-v0)
from agroflight.baselines import altitude_hold_action, random_action
from agroflight.envs import ENV_ID, OBS_DIM, AgroDroneEnv

ACTION_DIM = 4


def _find_model_xml():
    """Locate x2.xml via AGRO_DRONE_XML or the conventional clone location."""
    from_env = os.environ.get("AGRO_DRONE_XML")
    if from_env and os.path.isfile(from_env):
        return from_env
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, "mujoco_menagerie", "skydio_x2", "x2.xml")
    return candidate if os.path.isfile(candidate) else None


MODEL_XML = _find_model_xml()

pytestmark = pytest.mark.skipif(
    MODEL_XML is None,
    reason="Skydio X2 model not found. Run `make menagerie` or set AGRO_DRONE_XML.",
)


@pytest.fixture
def env():
    e = AgroDroneEnv(xml_path=MODEL_XML, seed=0)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Registration / construction
# ---------------------------------------------------------------------------
def test_env_is_registered():
    from gymnasium.envs.registration import registry

    assert ENV_ID in registry


def test_gym_make_produces_working_env():
    e = gym.make(ENV_ID, xml_path=MODEL_XML)
    try:
        obs, info = e.reset(seed=0)
        assert e.observation_space.contains(obs)
        assert isinstance(info, dict)
    finally:
        e.close()


def test_missing_xml_path_raises_informative_error(monkeypatch):
    monkeypatch.delenv("AGRO_DRONE_XML", raising=False)
    with pytest.raises(ValueError, match="AGRO_DRONE_XML"):
        AgroDroneEnv(xml_path=None)


# ---------------------------------------------------------------------------
# Space / shape contracts
# ---------------------------------------------------------------------------
def test_space_shapes(env):
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACTION_DIM,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.dtype == np.float32


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_five_tuple_with_correct_types(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    for key in ("crashed", "finished_row", "alt_error", "lateral_offset"):
        assert key in info, f"info is missing '{key}'"


# ---------------------------------------------------------------------------
# Numerical health
# ---------------------------------------------------------------------------
def test_no_nan_or_inf_in_observations_over_rollout(env):
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs)), "reset() produced a non-finite observation"

    for step in range(200):
        obs, reward, terminated, truncated, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert np.all(np.isfinite(obs)), f"non-finite observation at step {step}"
        assert np.isfinite(reward), f"non-finite reward at step {step}"
        if terminated or truncated:
            obs, _ = env.reset()


@pytest.mark.parametrize("policy_name", ["altitude-hold", "random"])
def test_baselines_produce_finite_actions_and_observations(env, policy_name):
    """Baselines must not blow up numerically -- says nothing about flying well."""
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)

    for _ in range(100):
        if policy_name == "altitude-hold":
            action = altitude_hold_action(obs)
        else:
            action = random_action(env.action_space, rng)
        assert np.all(np.isfinite(action))
        assert action.shape == (ACTION_DIM,)

        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated or truncated:
            obs, _ = env.reset()


# ---------------------------------------------------------------------------
# Action clipping
# ---------------------------------------------------------------------------
def test_out_of_range_action_is_clipped_not_rejected(env):
    """Actions outside the box must be clipped, and must not produce NaNs."""
    env.reset(seed=0)
    huge = np.full(ACTION_DIM, 1e6, dtype=np.float32)
    obs, reward, _, _, _ = env.step(huge)
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)


def test_control_stays_within_actuator_range(env):
    """Whatever the action, the ctrl written to MuJoCo must respect ctrlrange."""
    env.reset(seed=0)
    for action in (
        np.full(ACTION_DIM, 1e6, dtype=np.float32),
        np.full(ACTION_DIM, -1e6, dtype=np.float32),
        np.array([1e6, -1e6, 0.0, 5.0], dtype=np.float32),
    ):
        env.step(action)
        low, high = env.ctrl_range[:, 0], env.ctrl_range[:, 1]
        assert np.all(env.data.ctrl >= low - 1e-6), f"ctrl below range: {env.data.ctrl}"
        assert np.all(env.data.ctrl <= high + 1e-6), f"ctrl above range: {env.data.ctrl}"


# ---------------------------------------------------------------------------
# Hover keyframe sanity
# ---------------------------------------------------------------------------
def test_zero_action_means_hover_thrust(env):
    """action=0 must map to exactly the model's own hover keyframe ctrl."""
    env.reset(seed=0)
    env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    np.testing.assert_allclose(env.data.ctrl, env.hover_ctrl, rtol=1e-6)


def test_hover_holds_altitude_for_short_rollout():
    """With zero action delta and no wind, the drone should not fall out of
    the sky over a short rollout.

    This is a PHYSICS/WIRING check (is the hover keyframe actually wired to the
    right actuators?), not a performance check. The tolerance is deliberately
    loose -- it catches "thrust is not being applied at all", not "the
    controller is imprecise". Wind is disabled so the test is deterministic.
    """
    env = AgroDroneEnv(xml_path=MODEL_XML, wind_strength=0.0, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        start_altitude = env.data.qpos[2]

        for step in range(100):
            obs, _, terminated, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            assert not terminated, f"hover terminated (crashed) at step {step}"

        drift = abs(env.data.qpos[2] - start_altitude)
        assert drift < 0.5, (
            f"hover thrust drifted {drift:.3f}m over 100 steps -- the hover "
            "keyframe is probably not wired to the actuators correctly"
        )
    finally:
        env.close()


def test_forward_command_is_tracked():
    """Wiring check for the inner controller: a forward command produces forward speed."""
    env = AgroDroneEnv(xml_path=MODEL_XML, wind_strength=0.0, seed=0)
    try:
        env.reset(seed=0)
        action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # full forward = 2 m/s
        for _ in range(200):
            obs, _, terminated, _, _ = env.step(action)
            assert not terminated
        assert abs(obs[6] - 2.0) < 0.3, f"forward speed {obs[6]:.2f} m/s, expected ~2.0"
    finally:
        env.close()


def test_climb_command_raises_altitude():
    env = AgroDroneEnv(xml_path=MODEL_XML, wind_strength=0.0, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        start = obs[0]
        for _ in range(100):
            obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float32))
        assert obs[0] - start > 0.2
    finally:
        env.close()


def _fly_until_done(env, action, max_steps=1000):
    for _ in range(max_steps):
        _, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            return terminated, info
    return False, info


def test_climbing_out_of_the_envelope_ends_as_crash():
    env = AgroDroneEnv(xml_path=MODEL_XML, wind_strength=0.0, level=0, seed=0)
    try:
        env.reset(seed=0)
        terminated, info = _fly_until_done(env, [0.0, 0.0, 1.0, 0.0])
        assert terminated and info["out_of_bounds"] and info["crashed"]
        assert abs(info["alt_error"]) > env.max_alt_error
    finally:
        env.close()


def test_drifting_sideways_out_of_the_envelope_ends_as_crash():
    env = AgroDroneEnv(xml_path=MODEL_XML, wind_strength=0.0, level=0, seed=0)
    try:
        env.reset(seed=0)
        terminated, info = _fly_until_done(env, [0.0, 1.0, 0.0, 0.0])
        assert terminated and info["out_of_bounds"] and info["crashed"]
        assert abs(info["lateral_offset"]) > env.max_lateral_offset
    finally:
        env.close()


def test_envelope_can_be_disabled():
    env = AgroDroneEnv(xml_path=MODEL_XML, wind_strength=0.0, level=0, seed=0,
                       max_alt_error=None, max_lateral_offset=None)
    try:
        env.reset(seed=0)
        for _ in range(300):   # a sustained climb of ~3 m, well past the default bound
            _, _, terminated, _, info = env.step(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
            assert not terminated
        assert not info["out_of_bounds"]
    finally:
        env.close()


def test_yaw_is_held_unless_yaw_free():
    def final_yaw_rate(yaw_free):
        e = AgroDroneEnv(xml_path=MODEL_XML, wind_strength=0.0, seed=0, yaw_free=yaw_free)
        try:
            e.reset(seed=0)
            for _ in range(200):
                obs, _, _, _, _ = e.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            return obs[11]
        finally:
            e.close()

    assert abs(final_yaw_rate(False)) < 0.05
    assert abs(final_yaw_rate(True) - 1.0) < 0.2


FORWARD = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
HOLD = np.zeros(ACTION_DIM, dtype=np.float32)


def test_invalid_level_raises():
    with pytest.raises(ValueError, match="level"):
        AgroDroneEnv(xml_path=MODEL_XML, level=7)


def test_level0_is_altitude_hold_only():
    env = AgroDroneEnv(xml_path=MODEL_XML, level=0, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        assert abs(obs[1]) < 1e-9, "L0 must spawn on the row"
        for _ in range(500):
            obs, _, terminated, _, info = env.step(FORWARD)
            np.testing.assert_allclose(obs[12:14], 0.0, atol=1e-9)
            assert not terminated and not info["finished_row"]
        assert info["x"] > env.field_length, "row completion must be off at L0"
    finally:
        env.close()


def test_level1_finishes_the_row():
    env = AgroDroneEnv(xml_path=MODEL_XML, level=1, seed=0)
    try:
        env.reset(seed=0)
        for _ in range(env.max_steps):
            _, _, terminated, truncated, info = env.step(FORWARD)
            if terminated or truncated:
                break
        assert info["finished_row"]
    finally:
        env.close()


def test_level2_wind_is_steady_and_lateral():
    env = AgroDroneEnv(xml_path=MODEL_XML, level=2, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        first = obs[12]
        assert first != 0.0 and obs[13] == 0.0
        for _ in range(50):
            obs, _, _, _, _ = env.step(HOLD)
            assert obs[12] == first and obs[13] == 0.0
    finally:
        env.close()


def test_completing_the_row_beats_hovering_in_place():
    """Part B design rule: the progress term and completion bonus must dominate
    the survival bonus, so hovering is not a degenerate optimum."""
    def total_reward(action):
        e = AgroDroneEnv(xml_path=MODEL_XML, level=1, seed=0)
        try:
            e.reset(seed=0)
            total = 0.0
            for _ in range(e.max_steps):
                _, r, terminated, truncated, _ = e.step(action)
                total += r
                if terminated or truncated:
                    break
            return total
        finally:
            e.close()

    assert total_reward(FORWARD) > total_reward(HOLD)


def test_reward_weights_are_configurable():
    def first_reward(cfg):
        e = AgroDroneEnv(xml_path=MODEL_XML, level=1, seed=0, reward_config=cfg)
        try:
            e.reset(seed=0)
            return e.step(HOLD)[1]
        finally:
            e.close()

    assert first_reward(dict(survival_bonus=1.0)) - first_reward(None) == pytest.approx(0.98)


def test_wind_is_zero_when_wind_strength_is_zero(env):
    """The wind channel of the observation must actually respect the setting."""
    e = AgroDroneEnv(xml_path=MODEL_XML, wind_strength=0.0, seed=0)
    try:
        obs, _ = e.reset(seed=0)
        for _ in range(50):
            obs, _, terminated, truncated, _ = e.step(np.zeros(ACTION_DIM, dtype=np.float32))
            np.testing.assert_allclose(obs[-2:], np.zeros(2), atol=1e-8)
            if terminated or truncated:
                break
    finally:
        e.close()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_rollout():
    """reset(seed=k) must be reproducible -- the per-seed evaluation loop
    depends on it (this is the bug the prototype had: self._rng was never
    reseeded, so seeded evaluation was silently non-deterministic)."""
    def rollout(seed):
        e = AgroDroneEnv(xml_path=MODEL_XML, seed=123)
        try:
            obs, _ = e.reset(seed=seed)
            trace = [obs.copy()]
            for _ in range(30):
                obs, _, terminated, truncated, _ = e.step(np.zeros(ACTION_DIM, dtype=np.float32))
                trace.append(obs.copy())
                if terminated or truncated:
                    break
            return np.array(trace)
        finally:
            e.close()

    np.testing.assert_allclose(rollout(7), rollout(7), rtol=1e-9)


def test_different_seeds_give_different_rollouts():
    """Sanity: the seeding actually varies the episode (start offset + wind)."""
    def first_obs(seed):
        e = AgroDroneEnv(xml_path=MODEL_XML, seed=0)
        try:
            obs, _ = e.reset(seed=seed)
            return obs.copy()
        finally:
            e.close()

    assert not np.allclose(first_obs(1), first_obs(2))
