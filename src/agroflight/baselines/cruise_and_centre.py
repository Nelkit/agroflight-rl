"""Cruise-and-centre baseline (hand-coded, not learned), as Juan proposed it on 27 Sep.

Forward at full speed, steer back to the row centreline (sideways = -offset) and
hold altitude with the altitude-hold baseline. It already finishes the row in
L1-L4, so it is the bar a trained policy has to beat, not a floor.
"""
import numpy as np

from agroflight.baselines.altitude_hold import altitude_hold_action


def cruise_and_centre_action(obs):
    """Forward 1.0, lateral command = -lateral offset, altitude-hold climb, no yaw."""
    action = altitude_hold_action(obs)
    action[0] = 1.0
    action[1] = np.clip(-obs[1], -1.0, 1.0)
    return action
