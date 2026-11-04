# agroflight-rl

Reinforcement learning for **precision-agriculture crop-row flight control**: a
simulated quadrotor learns to fly along a crop row at a fixed altitude while
random wind gusts push it off course.

- **Physics:** MuJoCo, [Skydio X2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2)
  from Google DeepMind's `mujoco_menagerie` (Apache-2.0).
- **Algorithms:** three policy-gradient methods compared on the same env and
  reward: **PPO**, **REINFORCE** and **A2C**. Each one is implemented and
  trained in its own notebook under `notebooks/`.
- **Status:** trained on levels L0–L4; results live in `results/`.

## Contents

1. [Quick start](#quick-start)
2. [Project layout](#project-layout)
3. [Installation](#installation)
4. [Usage](#usage)
5. [The environment](#the-environment)
6. [Baselines](#baselines)
7. [Results](#results)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Academic integrity](#academic-integrity)
11. [Licence and attribution](#licence-and-attribution)

---

## Quick start

```bash
make setup       # create .venv, install the package (editable) + dev deps
make menagerie   # sparse-clone the Skydio X2 model
make test        # run the test suite
make viewer      # open the live desktop demo
```

`make help` lists every target.

The environment needs to know where the drone model lives. `make` sets
`AGRO_DRONE_XML` for you; if you run scripts by hand, export it yourself:

```bash
export AGRO_DRONE_XML=$(pwd)/mujoco_menagerie/skydio_x2/x2.xml
```

---

## Project layout

```
src/agroflight/
  envs/            AgroDroneEnv, inner controller, Gymnasium registration, plantation scene builder
  baselines/       altitude_hold, random and cruise_and_centre controllers
  training/        train_reinforce.py (headless REINFORCE runs for L0–L4)
  evaluation/      evaluate(): shared evaluation used by every notebook
  policies.py      load_policy(): rebuilds a trained policy from a checkpoint
notebooks/         one notebook per algorithm (PPO, REINFORCE, A2C) and comparison.ipynb
results/           result JSONs per algorithm, figures; archive/ holds superseded runs
scripts/           viewer.py (live desktop demo), assemble_reinforce_results.py, gltf_to_obj.py
tests/             test_env.py (API and sanity tests), test_baselines.py
docs/              SageMaker setup guide
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
| ----------- | ------- | ----- |
| Python | **3.11** (required) | `pyproject.toml` declares `>=3.11`, and `make setup` creates the venv with `python3.11`. The pinned versions in `requirements.txt` (e.g. gymnasium 1.4.0) do not install on 3.10. |
| `git` | any recent | Needed to sparse-clone the drone model |
| `make` | any recent | Preinstalled on macOS/Linux; on Windows use WSL |

### Recommended: `make setup`

This is the supported path. It creates the virtualenv, installs the package in
editable mode, and pulls both the training and dev extras in one step:

```bash
make setup
make menagerie
```

`make setup` is idempotent: it stamps `.venv/.installed` and only reinstalls
when `pyproject.toml` or `requirements.txt` changes.

`make menagerie` fetches the drone model, which is **not vendored** in this
repository. It sparse-clones only `skydio_x2/` from `mujoco_menagerie` (a full
clone is several GB; the sparse checkout is a few MB).

### Manual installation

If you would rather manage the environment yourself:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[train,dev]"
make menagerie
```

### Dependency groups

Dependencies live in `pyproject.toml`. They are split so the environment and
tests can be installed without PyTorch:

| Group | Install with | Contents | Needed for |
| ----- | ------------ | -------- | ---------- |
| core | `pip install -e .` | `mujoco`, `gymnasium`, `numpy` | The env, the baselines, the viewer |
| `train` | `pip install -e ".[train]"` | `torch`, `matplotlib`, `pandas` | The notebooks (training, evaluation, plots) |
| `dev` | `pip install -e ".[dev]"` | `pytest` | `make test` |

> **Note on `requirements.txt`.** That file mirrors the dependency list for
> readability, but **nothing installs from it**. Treat `pyproject.toml` as the
> single source of truth; if you add a dependency, add it there.

### Verifying the installation

```bash
make test
make viewer
```

A healthy test run reports every test as passing, and the viewer opens a window
with the drone over the crop row.

### Starting over

```bash
make clean      # removes .venv, the menagerie clone, runs/, caches
make setup && make menagerie
```

Use `make clean-runs` instead if you only want to discard training runs and
checkpoints while keeping the venv and the model.

---

## Usage

### Viewer (live desktop demo)

```bash
make viewer                                    # altitude-hold baseline
make viewer ARGS="--policy random"             # random baseline
make viewer ARGS="--checkpoint path/to/ckpt.pt"  # trained policy
```

Opens a desktop window showing the drone over the crop row. The window title
shows the level and the policy, a blue arrow shows the wind force on the drone,
and the top-left corner shows its lateral and vertical components.

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--policy` | `altitude-hold` | Hand-coded baseline: `altitude-hold` or `random` |
| `--checkpoint` | none | Trained `.pt` checkpoint. Overrides `--policy` |
| `--level` | inferred | Level L0–L4. Must match the level the checkpoint was trained on; inferred from a `_L<k>_` checkpoint name |
| `--field-length` | `8.0` | Crop row length (m) |
| `--max-steps` | `1000` | Steps before an episode times out |
| `--seed` | `1` | RNG seed |
| `--drone-dir` | menagerie clone | Folder containing `x2.xml` |

The vineyard scene is **cosmetic only** (`contype=0 conaffinity=0`): it has no
collision and no effect on the physics or the learning problem.

#### Trained PPO policies

The commands below run the best PPO seed for each level, meaning the `final`
checkpoint with the highest mean reward on the held-out evaluation seeds:

```bash
# L0, seed 2 (19.78)
make viewer ARGS="--checkpoint notebooks/runs/ppo_25609227/long/checkpoints/ppo_25609227_L0_seed2_final.pt --level 0"

# L1, seed 1 (100.34)
make viewer ARGS="--checkpoint notebooks/runs/ppo_25609227/long/checkpoints/ppo_25609227_L1_seed1_final.pt --level 1"

# L2, seed 2 (99.80)
make viewer ARGS="--checkpoint notebooks/runs/ppo_25609227/long/checkpoints/ppo_25609227_L2_seed2_final.pt --level 2"

# L3, seed 2 (99.91)
make viewer ARGS="--checkpoint notebooks/runs/ppo_25609227/long/checkpoints/ppo_25609227_L3_seed2_final.pt --level 3"

# L4, seed 0 (97.95)
make viewer ARGS="--checkpoint notebooks/runs/ppo_25609227/long/checkpoints/ppo_25609227_L4_seed0_final.pt --level 4"
```

Checkpoints are not versioned (`notebooks/runs/` is gitignored), so these paths
only exist after running the PPO notebook in `long` mode.

### Training

Training happens in one notebook per algorithm (`notebooks/ppo_25609227.ipynb`,
`reinforce_25418551.ipynb`, `a2c_25399217.ipynb`); see `notebooks/README.md`
and `docs/sagemaker_setup.md` for running them on SageMaker or Colab. REINFORCE's
final L0–L4 runs can also be trained headless with
`python -m agroflight.training.train_reinforce`.

### Evaluation

Every notebook evaluates through the same helper, deterministically, on the
held-out seeds 1000–1019:

```python
from agroflight.evaluation.evaluate_policy import evaluate
stats = evaluate(policy_fn, seeds=range(1000, 1020), env=env)
```

It reports mean reward, crash rate, row-completion rate, mean steps and mean
absolute altitude error, so learned policies and baselines are compared on the
same metrics.

---

## The environment

Registered with Gymnasium as **`AgroDrone-v0`**.

```python
import gymnasium as gym
import agroflight.envs  # registers the env

env = gym.make("AgroDrone-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The drone starts near the origin at roughly the target altitude with a small
random lateral offset, and must fly along **+x** (the crop row) while staying
on the **y = 0** centreline.

### Observation space: `Box(-inf, inf, (14,), float32)`

| Index | Component | Meaning |
| ----- | --------- | ------- |
| 0 | `alt_error` | `z − target_altitude` (metres) |
| 1 | `lateral_offset` | `y`, the signed distance off the row centreline |
| 2–5 | `quat` | body orientation quaternion (w, x, y, z) |
| 6–8 | `linvel` | linear velocity (vx, vy, vz) |
| 9–11 | `angvel` | angular velocity |
| 12–13 | `wind` | current wind state (lateral, vertical) |

Wind is observable, so this is a fully-observed MDP rather than a POMDP.

### Action space: `Box(-1.0, 1.0, (4,), float32)`

Four **normalised high-level commands**: forward velocity, lateral velocity,
climb rate, yaw rate. `ACTION_SCALE` maps them to ±2 m/s, ±1 m/s, ±1 m/s and
±1 rad/s, expressed in the drone's heading frame. `action = [0,0,0,0]` means
"hold position". A fixed, non-learned inner controller
(`envs/inner_controller.py`: velocity loop, attitude PD, thrust mixer) turns
the commands into the four rotor thrusts, so the policy never has to learn to
hover from raw motor commands. Yaw is held at 0 unless `yaw_free=True` (L4).

### Reward

```
reward = −(w_alt·alt_error² + w_lat·lateral_offset² + w_tilt·tilt² + w_action·|action|²)
         + w_progress · (metres gained along the row)
         + survival_bonus                     (per step)
         − crash_penalty                      if crashed
         + completion_bonus                   if the row was completed
```

`tilt` is the angle between the drone's up-axis and world-up (0 = level). All
weights live in `RewardConfig` and can be overridden with
`AgroDroneEnv(reward_config=dict(...))` for sensitivity analysis:

| Weight | Default |
| ------ | ------- |
| `w_alt` | 2 |
| `w_lat` | 2 |
| `w_tilt` | 0.5 |
| `w_action` | 0.01 |
| `w_progress` | 10 |
| `survival_bonus` | 0.02 |
| `crash_penalty` | 300 |
| `completion_bonus` | 20 |

The progress term and completion bonus are set so that finishing the row beats
hovering in place (checked in the tests).

### Difficulty levels

Fixed in advance (Part B) and never adjusted per algorithm. Pick one with
`gym.make("AgroDrone-v0", level=k)`; the default is `3`.

| Level | Wind | Spawn | Task |
| ----- | ---- | ----- | ---- |
| L0 | none | on the row | altitude hold only (no progress reward, no finish) |
| L1 | none | ±0.15 m offset | full row following |
| L2 | steady sideways, fixed per episode | ±0.15 m offset | full row following |
| L3 | gusting (random walk) | ±0.15 m offset | full row following |
| L4 | gusting | ±0.15 m offset | full row following, yaw rate free |

### Episode termination

| Condition | Flag | Type |
| --------- | ---- | ---- |
| `z < 0.08` or `tilt > 1.2 rad` | `crashed` | terminated |
| `\|alt_error\| > max_alt_error` or `\|lateral_offset\| > max_lateral_offset` | `crashed`, `out_of_bounds` | terminated |
| `x ≥ field_length` (levels L1–L4) | `finished_row` | terminated |
| `steps ≥ max_steps` (default 1000) | none | truncated |

### Configuration

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `target_altitude` | `1.2` | Target flight height (m) |
| `field_length` | `8.0` | Row length before completion (m) |
| `wind_strength` | `0.6` | Max wind force magnitude (set `0.0` to disable) |
| `max_steps` | `1000` | Truncation limit (10 s at `dt = 0.01`) |
| `level` | `3` | Difficulty level L0–L4 (see above) |
| `reward_config` | `None` | `RewardConfig` or dict of weight overrides |
| `frame_skip` | `1` | Physics steps per action |
| `yaw_free` | `False` | Let the policy command yaw rate |
| `seed` | `None` | RNG seed for wind and start offset |
| `max_alt_error` | `1.0` | Altitude envelope (m); `None` disables it |
| `max_lateral_offset` | `1.0` | Lateral envelope (m); `None` disables it |

---

## Baselines

All three are hand-coded and define what a learned policy has to beat.

- **`altitude_hold`:** PD controller applying uniform thrust to all four rotors.
  Holds altitude and stays level, but does nothing about lateral wind drift or
  forward progress.
- **`random`:** scaled-down uniform random actions. The sanity floor.
- **`cruise_and_centre`:** forward command 1.0, lateral command `clip(-offset)`,
  climb from `altitude_hold`, yaw 0. It completes the row on L1–L4, which makes
  it the bar to beat.

---

## Results

| File | Contents |
| ---- | -------- |
| `results/ppo_25609227.json` | PPO results (shared schema) |
| `results/reinforce_25418551.json` | REINFORCE results (shared schema) |
| `results/a2c_25399217.json` | A2C results (shared schema) |
| `results/a2c_25399217_L*_seed*_*.json` | A2C per-run results, L0–L4 |
| `results/comparison.json`, `results/comparison_summary.csv` | Cross-algorithm comparison, from `notebooks/comparison.ipynb` |
| `results/figures/` | Comparison and per-algorithm figures |

The analysis is in each notebook and in the final report.

---

## Testing

```bash
make test
```

`tests/test_env.py` covers **API compliance and numerical health**: registration,
`reset()`/`step()` shapes and types, no NaN/Inf over rollouts, action clipping,
control staying inside `ctrlrange`, hover holding altitude with zero action,
and seed reproducibility. `tests/test_baselines.py` covers the baseline
controllers.

The suite deliberately contains **no tests asserting that a trained policy flies
well**. Performance claims belong in the notebooks' evaluation and the report.

The suite skips itself entirely if the Skydio X2 model is missing; run
`make menagerie`.

---

## Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| All tests skipped | Drone model not cloned | `make menagerie` |
| `ValueError: No xml_path given and AGRO_DRONE_XML env var not set` | Running a script outside `make` | `export AGRO_DRONE_XML=$(pwd)/mujoco_menagerie/skydio_x2/x2.xml` |
| `.venv/bin/mjpython: Permission denied` or `No such file or directory` on macOS | The venv's interpreter path is broken (the project moved, or the path has a problematic space) | Recreate the venv: `rm -rf .venv && make setup` |
| Viewer exits immediately on macOS | MuJoCo's viewer must own the main thread | Use `make viewer`, not `python scripts/viewer.py`; the Makefile selects `mjpython` |

---

## Licence and attribution

- **Drone model:** the Skydio X2 model comes from
  [`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
  (Google DeepMind, Apache-2.0) and is fetched at setup time, not vendored.
- **PPO:** the PPO notebook follows the design of
  [CleanRL](https://github.com/vwxyzjn/cleanrl)'s `ppo_continuous_action.py` (MIT).
- **Viewer scene:** based on
  ["Day 219: Grapevine plant - 1scanaday"](https://sketchfab.com/3d-models/day-219-grapevine-plant-1scanaday-f59fa3cd369941af8bab9716aac768e2)
  by [emm](https://sketchfab.com/edemaistre), licensed under
  [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/). Changes are listed in
  `src/agroflight/envs/assets/grapevine/README.md`.
