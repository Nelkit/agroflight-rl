"""Random-action baseline (sanity-check floor: expect a quick crash)."""
import numpy as np

# Scale factor carried over from the prototype: full-range random commands make
# the drone thrash; scaling down keeps the baseline in the air long enough to
# be a meaningful lower bound rather than an immediate crash.
DEFAULT_SCALE = 0.3


def random_action(action_space, rng=None, scale=DEFAULT_SCALE):
    """Sample a scaled-down uniform random action.

    Pass `rng` (a numpy Generator) for reproducible rollouts; without it the
    action space's own seeded RNG is used.
    """
    if rng is not None:
        low, high = action_space.low, action_space.high
        sample = rng.uniform(low, high).astype(np.float32)
    else:
        sample = action_space.sample()
    return (sample * scale).astype(np.float32)
