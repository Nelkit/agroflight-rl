"""Naive altitude-hold baseline (hand-coded, not learned)."""
import numpy as np

from agroflight.envs.agro_drone_env import ACTION_SCALE

DEFAULT_KP_ALT = 1.5   # climb-rate command (m/s) per metre of altitude error


def altitude_hold_action(obs, kp_alt=DEFAULT_KP_ALT):
    """Climb-rate command proportional to altitude error; every other command is zero.

    Holds altitude well, but does nothing about lateral wind drift or forward
    progress along the row. This is the 'naive baseline' that motivates a
    learned policy.
    """
    climb = np.clip(-kp_alt * obs[0] / ACTION_SCALE[2], -1.0, 1.0)
    return np.array([0.0, 0.0, climb, 0.0], dtype=np.float32)
