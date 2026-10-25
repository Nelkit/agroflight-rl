"""
Unit tests for the hand-coded baselines: the commands they send for a given
observation. No env or MuJoCo model needed. How well they fly is an evaluation
result, not a test (see the note at the top of test_env.py).
"""
import numpy as np
import pytest

from agroflight.baselines import altitude_hold_action, cruise_and_centre_action


def _obs(alt_error=0.0, lateral_offset=0.0):
    obs = np.zeros(14, dtype=np.float32)
    obs[0], obs[1] = alt_error, lateral_offset
    return obs


@pytest.mark.parametrize("offset", [-0.4, 0.0, 0.3])
def test_cruise_and_centre_flies_forward_and_steers_back_to_the_row(offset):
    action = cruise_and_centre_action(_obs(lateral_offset=offset))
    assert action[0] == 1.0
    assert action[1] == pytest.approx(-offset)
    assert action[3] == 0.0


def test_cruise_and_centre_climbs_like_altitude_hold():
    obs = _obs(alt_error=-0.2, lateral_offset=0.1)
    assert cruise_and_centre_action(obs)[2] == altitude_hold_action(obs)[2]


def test_cruise_and_centre_stays_inside_the_action_box():
    action = cruise_and_centre_action(_obs(alt_error=5.0, lateral_offset=-3.0))
    assert action.shape == (4,) and action.dtype == np.float32
    assert np.all(np.abs(action) <= 1.0)
