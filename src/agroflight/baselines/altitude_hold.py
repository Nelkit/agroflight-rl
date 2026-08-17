"""Naive altitude-hold baseline (hand-coded, not learned)."""
import numpy as np

from agroflight.envs.agro_drone_env import OBS_IDX_VZ

# Gains carried over unchanged from the exploration prototype.
# TODO (team): these are the prototype's gains, not tuned ones. If you want the
# baseline comparison in your report to be a fair fight, decide deliberately
# whether to tune them (a well-tuned PD baseline is a much stronger claim than
# beating a sloppy one) and document that choice.
DEFAULT_KP_ALT = 2.2
DEFAULT_KD_ALT = 1.4


def altitude_hold_action(obs, kp_alt=DEFAULT_KP_ALT, kd_alt=DEFAULT_KD_ALT):
    """Uniform thrust correction on all 4 rotors.

    Holds altitude reasonably well (no differential thrust, so it stays level)
    but does nothing about lateral wind drift or forward progress along the
    row. This is the 'naive baseline' that motivates a learned policy.
    """
    alt_error, vz = obs[0], obs[OBS_IDX_VZ]
    lift = -kp_alt * alt_error - kd_alt * vz
    return np.array([lift, lift, lift, lift], dtype=np.float32)
