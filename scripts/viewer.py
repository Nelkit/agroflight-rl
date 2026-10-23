"""
Standalone desktop viewer for the agro-drone environment.

Why this exists: rendering MuJoCo inside a Jupyter notebook needs a headless GL
backend (EGL/OSMesa), which is fiddly and platform-dependent. This script
sidesteps that by using MuJoCo's interactive viewer, which opens a REAL desktop
window using the OS's native graphics (Cocoa/OpenGL on macOS -- no OSMesa/EGL
needed). On a normal Mac with a screen this should just work.

It shows the Skydio X2 flying over the crop row using the same environment the
training pipeline uses, driven by a hand-coded baseline controller. This is NOT
a trained agent -- it demonstrates the environment and the problem.

NOTE (macOS): MuJoCo's passive viewer must run on the main thread, which means
you have to launch it with `mjpython`, not `python`:
    mjpython scripts/viewer.py
On Linux/Windows plain `python scripts/viewer.py` is fine. `make viewer`
picks the right one for you.

Run:
    make viewer
    mjpython scripts/viewer.py --policy random
    mjpython scripts/viewer.py --drone-dir /path/to/mujoco_menagerie/skydio_x2
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import re

# Allow running this script directly from a source checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from agroflight.baselines import altitude_hold_action, random_action  # noqa: E402
from agroflight.envs import AgroDroneEnv, build_plantation_scene      # noqa: E402
from agroflight.policies import load_policy                           # noqa: E402

DEFAULT_DRONE_DIR = "mujoco_menagerie/skydio_x2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live desktop viewer for the agro-drone environment."
    )
    parser.add_argument("--drone-dir", default=DEFAULT_DRONE_DIR,
                        help="Path to the cloned skydio_x2 folder (contains x2.xml).")
    parser.add_argument("--policy", choices=["altitude-hold", "random"],
                        default="altitude-hold",
                        help="Which hand-coded baseline controller to run.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a trained .pt checkpoint. Overrides --policy.")
    parser.add_argument("--level", type=int, default=None,
                        help="Difficulty level L0-L4. Must match the level the checkpoint was trained on.")
    parser.add_argument("--field-length", type=float, default=8.0,
                        help="Crop row length in metres (visual + episode end condition).")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Steps before an episode times out.")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()

def level_from_checkpoint(path):
    """Read the level from a checkpoint name like '..._L3_seed0_final.pt' (None if absent)."""
    match = re.search(r"_L(\d)_", os.path.basename(path))
    return int(match.group(1)) if match else None

def main():
    args = parse_args()

    level = args.level
    if args.checkpoint:
        ckpt_level = level_from_checkpoint(args.checkpoint)
        if level is None:
            level = ckpt_level
        elif ckpt_level is not None and ckpt_level != level:
            print(f"WARNING: checkpoint was trained on L{ckpt_level} but --level is L{level}; flying on L{level}.")

    if level is None:
        level = 0

    xml_path = build_plantation_scene(args.drone_dir, field_length=args.field_length)
    env = AgroDroneEnv(
        xml_path=xml_path,
        field_length=args.field_length,
        max_steps=args.max_steps,
        seed=args.seed,
        level=level,
    )
    rng = np.random.default_rng(args.seed)

    if args.checkpoint:
        policy = load_policy(args.checkpoint)
        label = f"trained agent ({args.checkpoint})"
    else:
        def policy(obs):
            if args.policy == "altitude-hold":
                return altitude_hold_action(obs)
            return random_action(env.action_space, rng)
        label = f"hand-coded baseline ({args.policy})"

    print(f"Opening viewer window -- policy = {label}, level = L{env.level}. Close the window to stop.")
    print("(Episode restarts automatically on crash / row completion / timeout.)")

    obs, _ = env.reset(seed=args.seed)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            obs, reward, terminated, truncated, info = env.step(policy(obs))
            viewer.sync()

            if terminated or truncated:
                if info["crashed"]:
                    status = "CRASHED"
                elif info["finished_row"]:
                    status = "FINISHED ROW"
                else:
                    status = "timed out"
                print(f"Episode ended ({status}). Restarting...")
                obs, _ = env.reset()

            # keep roughly real-time
            elapsed = time.time() - step_start
            time.sleep(max(0.0, env.model.opt.timestep - elapsed))


if __name__ == "__main__":
    main()
