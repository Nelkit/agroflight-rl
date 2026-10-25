"""
Hand-coded (NOT learned) baseline controllers, used to demonstrate the
environment and to give the trained policy something to be compared against.

Altitude-hold and random do not "solve" the task -- that is the point: they show
exactly the gap a trained RL policy needs to close. Cruise-and-centre does finish
the row, so it is the bar to beat rather than a floor.

In the prototype these controllers were copy-pasted across the env module, the
viewer and the evaluation script, and had already drifted apart (the viewer's
`random_action` sampled uniform(-1, 1) while the others sampled the action
space scaled by 0.3). They live here once so all consumers share one definition.
"""
from agroflight.baselines.altitude_hold import altitude_hold_action
from agroflight.baselines.cruise_and_centre import cruise_and_centre_action
from agroflight.baselines.random_policy import random_action

__all__ = ["altitude_hold_action", "cruise_and_centre_action", "random_action"]
