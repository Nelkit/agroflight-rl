# agroflight-rl

Reinforcement learning for **precision-agriculture crop-row flight control**: a
simulated quadrotor learns to fly along a crop row at a fixed altitude while
random wind gusts push it off course.

- **Physics:** MuJoCo, [Skydio X2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2)
  from Google DeepMind's `mujoco_menagerie` (Apache-2.0).
- **Algorithm:** PPO via [CleanRL](https://github.com/vwxyzjn/cleanrl). We call
  CleanRL's reference implementation rather than reimplementing PPO.
- **Status:** scaffolding complete, **nothing trained yet**. The reward
  function, hyperparameters and evaluation protocol are deliberately untuned —
  see [Work still to do](#work-still-to-do).

---

## Quick start

```bash
make setup       # create .venv, install the package (editable) + dev deps
make menagerie   # sparse-clone the Skydio X2 model
make test        # run the sanity test suite
make viewer      # open the live desktop demo
```

`make help` lists every target.

The environment needs to know where the drone model lives. `make` sets
`AGRO_DRONE_XML` for you; if you run scripts by hand, export it yourself:

```bash
export AGRO_DRONE_XML=$(pwd)/mujoco_menagerie/skydio_x2/x2.xml
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
| ----------- | ------- | ----- |
| Python | **3.11** (verified) | `pyproject.toml` declares `>=3.9`, but 3.11.4 is the version this project is actually developed and tested on. See the caveat below. |
| `git` | any recent | Needed to sparse-clone the drone model |
| `make` | any recent | Preinstalled on macOS/Linux; on Windows use WSL |

> **Python version caveat.** Some CleanRL releases constrain themselves to
> `>=3.8,<3.11`, so the declared `>=3.9` floor can produce a dependency
> resolution conflict on older interpreters. If `make setup` fails while
> resolving `cleanrl`, install with Python 3.11 explicitly (see
> [Troubleshooting](#troubleshooting)) or skip the `train` extra.

### Recommended: `make setup`

This is the supported path — it creates the virtualenv, installs the package in
editable mode, and pulls both the training and dev extras in one step:

```bash
make setup
```

Equivalent to:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[train,dev]"
```

`make setup` is idempotent: it stamps `.venv/.installed` and only reinstalls
when `pyproject.toml` or `requirements.txt` changes.

### Manual installation

If you would rather manage the environment yourself:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows (WSL recommended): .venv/bin/activate
pip install --upgrade pip
pip install -e ".[train,dev]"
```

Then fetch the drone model, which is **not vendored** in this repository:

```bash
make menagerie
```

That sparse-clones only `skydio_x2/` from `mujoco_menagerie` (a full clone is
several GB; the sparse checkout is a few MB).

### Dependency groups

Dependencies live in `pyproject.toml`. They are split so a broken CleanRL
release cannot block the environment and tests from installing:

| Group | Install with | Contents | Needed for |
| ----- | ------------ | -------- | ---------- |
| core | `pip install -e .` | `mujoco`, `gymnasium`, `numpy`, `tensorboard` | The env, the baselines, the viewer |
| `train` | `pip install -e ".[train]"` | `cleanrl[mujoco]` | `make train` (PPO) |
| `dev` | `pip install -e ".[dev]"` | `pytest` | `make test` |

To install everything (what `make setup` does):

```bash
pip install -e ".[train,dev]"
```

If CleanRL fails to resolve on your machine, you can still work on the
environment, the baselines, the viewer and the test suite:

```bash
pip install -e ".[dev]"   # core + pytest, no CleanRL
```

> **Note on `requirements.txt`.** That file mirrors the dependency list for
> readability, but **nothing installs from it** — `make setup` and every
> command above install from `pyproject.toml`. Treat `pyproject.toml` as the
> single source of truth; if you add a dependency, add it there. (Reconciling
> or removing the duplicate file is an open task — see
> [Work still to do](#reproducibility).)

### Verifying the installation

```bash
make test
```

A healthy run reports **15 passing tests**. If the suite reports everything as
*skipped*, the Skydio X2 model is missing — run `make menagerie`.

Then confirm the simulator renders:

```bash
make viewer
```

### Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| All tests skipped | Drone model not cloned | `make menagerie` |
| `ValueError: No xml_path given and AGRO_DRONE_XML env var not set` | Running a script outside `make` | `export AGRO_DRONE_XML=$(pwd)/mujoco_menagerie/skydio_x2/x2.xml` |
| `.venv/bin/mjpython: Permission denied` on macOS | A **space in the project path** breaks the interpreter shebang | Move the repo to a path with no spaces (this bit us before — the folder was literally named `" agroflight-rl"`) |
| Viewer exits immediately on macOS | MuJoCo's viewer must own the main thread | Use `make viewer`, not `python scripts/viewer.py` — the Makefile selects `mjpython` |
| `pip` cannot resolve `cleanrl` | Interpreter outside CleanRL's supported range | Recreate the venv with 3.11: `rm -rf .venv && python3.11 -m venv .venv && make setup`, or install `".[dev]"` only |
| `ImportError` from `cleanrl.ppo_continuous_action` | CleanRL API shifted between releases | `pip install "cleanrl[mujoco]"`; worst case vendor `ppo_continuous_action.py` and set `env_id="AgroDrone-v0"` |

### Starting over

```bash
make clean      # removes .venv, the menagerie clone, runs/, caches
make setup && make menagerie
```

Use `make clean-runs` instead if you only want to discard training runs and
checkpoints while keeping the venv and the model.

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

### Observation space — `Box(-inf, inf, (14,), float32)`

| Index | Component | Meaning |
| ----- | --------- | ------- |
| 0 | `alt_error` | `z − target_altitude` (metres) |
| 1 | `lateral_offset` | `y` — signed distance off the row centreline |
| 2–5 | `quat` | body orientation quaternion (w, x, y, z) |
| 6–8 | `linvel` | linear velocity (vx, vy, vz) |
| 9–11 | `angvel` | angular velocity |
| 12–13 | `wind` | current wind state (lateral, vertical) |

Wind is observable, so this is a fully-observed MDP rather than a POMDP. That
was a prototype choice, not a validated one — see the TODOs.

### Action space — `Box(-1.0, 1.0, (4,), float32)`

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
weights live in `RewardConfig` (defaults: 2, 2, 0.5, 0.01, 10, 0.02, 50, 20) and
can be overridden with `AgroDroneEnv(reward_config=dict(...))` for sensitivity
analysis. The progress term and completion bonus are set so that finishing the
row beats hovering in place (checked in the tests).

> **These are first-pass defaults, not tuned.** Tuning them is project work, and
> flights must be watched in the viewer for reward hacking.

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
| `x ≥ field_length` (levels L1–L4) | `finished_row` | terminated |
| `steps ≥ max_steps` (default 1000) | — | truncated |

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

---

## Running things

### Viewer (live desktop demo)

```bash
make viewer                          # altitude-hold baseline
make viewer ARGS="--policy random"   # random baseline
```

Opens a real desktop window showing the drone over the crop row, driven by a
**hand-coded baseline**, not a trained policy. On macOS this must run under
`mjpython` (the Makefile handles it); running `python scripts/viewer.py`
directly on macOS will fail because the viewer needs the main thread.

The crop-row markers are **cosmetic only** (`contype=0 conaffinity=0`) — they
have no collision and no effect on the physics or the learning problem.

### Training

```bash
make train                    # or: python -m agroflight.training.train_ppo_cleanrl
make tensorboard              # watch the reward curve at localhost:6006
```

Runs CleanRL's `ppo_continuous_action` against `AgroDrone-v0`. **Untuned:** it
will train *something*, but nothing here has been set up to fly well.

### Evaluation

```bash
make evaluate                 # or: python -m agroflight.evaluation.evaluate_policy
```

Reports mean reward, crash rate, row-completion rate and mean absolute altitude
error for each baseline. Once you have a checkpoint, wire it in at the marked
TODO in `src/agroflight/evaluation/evaluate_policy.py` — the **three-way
comparison** (learned vs altitude-hold vs random) is the result worth
reporting, not the training curve alone.

---

## Baselines

Both are hand-coded, neither solves the task — that's the point. They define
the gap a learned policy has to close.

- **`altitude_hold`** — PD controller applying uniform thrust to all four
  rotors. Holds altitude, stays level, but does nothing about lateral wind
  drift or forward progress.
- **`random`** — scaled-down uniform random actions. The sanity floor; expect
  it to crash.

---

## Project layout

```
src/agroflight/
  envs/            AgroDroneEnv, inner controller, Gymnasium registration, plantation scene builder
  baselines/       altitude_hold, random controllers
  training/        CleanRL PPO entry point
  evaluation/      baseline + checkpoint evaluation
scripts/viewer.py  live desktop demo
notebooks/         exploration-phase notebooks (historical; package is source of truth)
tests/test_env.py  API/sanity tests
docs/project_plan.md  week-by-week plan template (team fills in)
```

---

## Testing

```bash
make test
```

The suite covers **API compliance and numerical health only**: registration,
`reset()`/`step()` shapes and types, no NaN/Inf over rollouts, action clipping,
control staying inside `ctrlrange`, hover holding altitude with zero action,
and seed reproducibility.

It deliberately contains **no tests asserting that a policy flies well**. There
is no trained policy, and encoding performance thresholds in tests would be
pre-answering the assignment. Performance claims belong in the evaluation
module and the report.

The suite skips itself entirely if the Skydio X2 model is missing — run
`make menagerie`.

---

## Work still to do

Everything below is **the team's own work** and is intentionally left
unresolved in this scaffold.

### Reward design — `src/agroflight/envs/agro_drone_env.py`
- [ ] Tune the weights (altitude, lateral, tilt, action cost, step bonus).
- [x] Forward progress has its own reward term (`w_progress`) in reward v1.
- [ ] Revisit the crash penalty / completion bonus magnitudes.
- [ ] Check for reward hacking — watch actual flight in the viewer, not just
      the reward number.

### Hyperparameters — `src/agroflight/training/train_ppo_cleanrl.py`
- [ ] `total_timesteps` (200k is a pipeline smoke-test value, not a budget).
- [ ] `learning_rate`, `gamma`, `gae_lambda`, `ent_coef`, `num_envs`,
      `num_steps`.
- [ ] Decide whether PPO is the right algorithm at all, or whether to compare
      against SAC.

### Evaluation protocol — `src/agroflight/evaluation/evaluate_policy.py`
- [ ] Number of evaluation episodes and which seeds (held out from training?).
- [ ] Success thresholds — define them *before* seeing trained results.
- [ ] Load and wire in the trained checkpoint.
- [ ] Decide whether to tune the altitude-hold PD gains so the baseline
      comparison is a fair fight.

### Environment design
- [ ] Is observable wind the right choice, or should this be partially
      observed?
- [ ] Are `target_altitude`, `field_length` and `wind_strength` the right task
      difficulty?
- [ ] Domain randomisation across episodes?

### Project management — `docs/project_plan.md`
- [ ] Fill in the week-by-week plan, owners, milestones and risks.
- [ ] Keep the experiment log current as you go.

### Reproducibility
- [ ] Pin dependency versions (`pip freeze`) before submission.

---

## Results

<!-- TODO: fill in once you have training results -->

---

## Academic integrity

The repository **structure**, boilerplate, wiring and sanity tests were
scaffolded with AI assistance, which the unit permits. The RL problem itself —
reward design, hyperparameter tuning, training, evaluation, analysis and the
report — is the team's own work and is left explicitly unsolved here. Every
such decision is marked `TODO (team)` in the source.

## Licence / attribution

The Skydio X2 model comes from
[`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
(Google DeepMind, Apache-2.0) and is fetched at setup time, not vendored.
PPO comes from [CleanRL](https://github.com/vwxyzjn/cleanrl) (MIT).
