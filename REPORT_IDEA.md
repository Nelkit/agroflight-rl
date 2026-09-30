# AgriFlight-RL — Project Report

*Reinforcement Learning for Crop-Row Flight Stabilization in Simulated Agricultural Drones*
Assessment 3, 43008 Reinforcement Learning, Spring 2026

## 1. Project Concept

Train an RL agent to control a simulated quadrotor drone so it can fly along a crop row at a fixed altitude, correcting for lateral wind drift, while making forward progress, a precision-agriculture use case (autonomous crop monitoring). The idea originated from a team member's firsthand observation of inventory/planning pain points in real operations, and was refined into a drone-flight-control formulation after evaluating several alternatives (robotics manipulation, restaurant inventory RL, dynamic pricing, traffic signal control) for originality, feasibility, and available tooling.

## 2. Why This Problem

- Real, recognizable industry application: precision-agriculture UAVs for crop monitoring.
- Genuine continuous-control problem (not a toy grid-world), distinct from typical class projects.
- No mandatory hardware, fully simulation-based, unlike other course-suggested ideas (DonkeyCar, JetRacer, Pepper robot) that need physical kits or lab access.
- Backed by real research precedent: published work exists on 2D, fixed-altitude simplifications for agricultural UAV coverage/path-planning via RL, validating the simplification path discussed in Section 7.

## 3. Technical Formulation

**Physics model:** Skydio X2 quadrotor MJCF from Google DeepMind's `mujoco_menagerie` (Apache-2.0, actively maintained), loaded in MuJoCo. Confirmed working: 4 actuators (per-rotor thrust), 7 position DOF (`nq`), 6 velocity DOF (`nv`), stable hover verified via the model's built-in hover keyframe.

**State (14-dim):** altitude error, lateral offset from the crop-row centerline, orientation quaternion (4), linear velocity (3), angular velocity (3), current wind disturbance (2).

**Action (4-dim, continuous):** thrust delta per rotor, centered on the hover keyframe, clipped to the model's actuator control range.

**Reward:** negative altitude error, negative lateral offset, negative tilt, small action-cost penalty, small per-step "still flying" bonus; -20 on crash, +20 on completing the row. Explicitly flagged throughout as an illustrative starting point, not validated for training, reward shaping is planned as an iterative, team-reviewed process (Section 9).

**Episode termination:** crash (altitude too low or excessive tilt) or reaching the end of the row; truncation at a max step count.

## 4. Ecosystem & Tooling (verified, not assumed)

- **MuJoCo**: free, open-source (Apache-2.0) since 2021/2022, trivially pip-installable.
- **`mujoco_menagerie`**: actively maintained by Google DeepMind; ships both Skydio X2 and Bitcraze Crazyflie 2 drone models.
- **Gymnasium**: standard RL environment API; environment registered as `AgroDrone-v0`.
- **CleanRL**: actively maintained (9.9k stars, 2026 activity), confirmed Gymnasium-compatible with custom registered environments (not just Atari), used for training rather than reimplementing PPO from scratch.
- Investigated and ruled out as unsuitable for this specific niche: `or-gym` (stale, non-Gymnasium, wrong problem shape), `reco-gym`/older MuJoCo-drone Gym wrappers (stale, pre-Gymnasium APIs).

## 5. Prototype / MVP Work Completed

- **Two Colab-ready Jupyter notebooks**: one for AgriFlight-RL, one for a separate restaurant-inventory RL idea (not pursued further). Both execute end-to-end with zero errors (verified via `jupyter nbconvert --execute`).
- **`AgroDroneEnv`**: a working Gymnasium environment implementing the spec in Section 3, tested for correct hover physics and reward computation.
- **Two hand-coded, non-learned baseline controllers** (altitude-hold, random) used only to sanity-check the environment, not to solve the task, both verified: altitude-hold holds altitude tightly but drifts laterally under wind (illustrating exactly the gap a trained policy needs to close); random crashes quickly (confirming crash-detection logic works).
- **`scene_builder.py`**: generates a plantation-style visual scene (drone model + cosmetic crop-row markers + floor + sky), verified to have zero effect on physics (identical hover behavior/reward with or without the markers).
- **`agro_drone_viewer.py`**: standalone desktop Python script using MuJoCo's native interactive viewer (real OS window, not headless rendering), built specifically to sidestep the headless-rendering compatibility problems encountered in notebooks (see Section 6).
- **Training scaffold** (`agro_drone_gym_env.py`, `train_ppo_cleanrl.py`, `evaluate_trained_policy.py`): registers the environment, wires it to CleanRL's existing PPO implementation, and provides an evaluation harness comparing a trained policy against the baselines. Deliberately left untuned, hyperparameters and reward weights are marked as team TODOs, not solved by this scaffolding.

## 6. Rendering / Visual Demo Journey

Getting a 3D visual preview working inside a Jupyter notebook proved to be a significant, multi-attempt debugging effort: no floor/sky in the bare drone model (black-void render), headless GL backend failures (`EGL`, `OSMesa`), a multi-minute hang traced to heavy skybox/checker textures in the official scene file, a `PYOPENGL_PLATFORM` vs `MUJOCO_GL` mismatch causing a Mesa "framebuffer not complete" error, and platform differences (Linux/VS Code vs. Colab vs. Mac). The resolution: build a **standalone desktop viewer** using MuJoCo's native windowed rendering instead of headless notebook rendering, this uses the OS's normal graphics stack (no OSMesa/EGL needed at all) and is expected to work out-of-the-box on a normal Mac/Windows/Linux desktop.

## 7. Design Decisions Under Consideration

### 3D full quadrotor vs. 2D simplified kinematic model

| Dimension | 3D MuJoCo (full quadrotor) | 2D Simplified (kinematic) |
|---|---|---|
| State dimensionality | 14 | ~6 |
| Action dimensionality | 4 continuous (per-rotor thrust) | 2 continuous, or ~5-9 discrete if paired with DQN |
| Core difficulty | Hard: must learn attitude stability before anything else | Easy-to-moderate: no flip/instability failure mode |
| Timesteps to a reasonable policy | ~2-5M+ (per typical continuous-control RL benchmarks) | ~50k-300k |
| Time to "demo-ready" | ~1-3 days of hands-on iteration (crash/no-crash tends to be pass/fail) | A few hours, same-day achievable |
| Hardware | GPU recommended | CPU (laptop) sufficient |
| Suggested algorithm | PPO or SAC | PPO/SAC still fine; DQN is a genuinely good fit if the action space is discretized |
| Visual demo | Real 3D flight, real drone model, strong "real physics" story | Simpler, unless paired with the MuJoCo-for-visuals-only hybrid |
| Main risk | Training instability eating into the timeline | Weaker technical-depth story for the report/defense |

A **hybrid option** was also proposed: keep MuJoCo purely for the visual (the plantation scene, live viewer), but simplify the underlying control interface to something closer to the 2D kinematic model, aiming for the fast/reliable training of 2D with the visual payoff of an actual 3D flying-drone demo.

### PPO vs. DQN

The brief does not mandate any specific algorithm. DQN requires a discrete action space; the current 4-continuous-rotor design would need per-rotor discretization (625 combinations, impractically large) to work with DQN. In the 2D simplified version, discretizing down to ~5-9 directional actions (e.g. forward-left, forward-straight, hover) makes DQN a genuinely well-suited, not forced, choice. PPO/SAC remain valid and simpler in the 2D continuous case regardless.

### Training time reality check

Confirmed with the subject coordinator's in-class guidance: RL training commonly requires millions of environment steps to reach good performance (consistent with published Atari/MuJoCo continuous-control benchmarks). Translated to wall-clock time for this project: a single run at millions of timesteps is plausibly 15 minutes to ~1.5 hours on modest hardware, the real time cost is needing several such runs during reward/hyperparameter iteration, not any single run being slow.

## 8. Team Structure & Reward-Design Risk Mitigation

In response to a team member's concern that reward-function design is easy to get wrong:

- The reward function lives in one shared file, changed only through small, reviewed commits.
- Every reward change is logged (what changed, why, before/after evaluation metrics from `evaluate_trained_policy.py`).
- **Three weekly 30-minute team syncs** scheduled during the Part-C iteration window, reviewing training curves and baseline-comparison metrics together before keeping or reverting any change. No single change ships on one person's judgment alone.

**Proposed roles** (rotating, not siloed, since grading is individual based on contribution): Environment & Physics; Training & Experiments; Evaluation & Demo/GUI.

## 9. Assessment Deliverable Timeline (confirmed dates)

Anchor: Week 1 = 27 Jul 2026. Overall assignment due 23 Oct 2026, 23:59.

| Part | Deliverable | Due | Weight |
|---|---|---|---|
| A | Title, abstract, group details | Week 4 Mon (~20 Aug) | 5pts |
| B | Implementation/development plan | Week 7 Mon (7 Sep) | 5pts |
| C | Initial experimental results | Week 10 Mon (28 Sep) | 5pts |
| D | Interface/GUI design | Week 11 Mon (5 Oct) | 5pts |
| E | Complete project | Week 12 (23 Oct hard deadline) | 40pts |
| F | Presentation + demo video | Week 14 | 15pts |
| G | Oral defense (in-person, individual) | Week 14 | 25pts |

A detailed 32-task breakdown mapped to these dates was created in Notion, each task carries a full explanation, owner, priority, and status.

## 10. Academic Integrity Note

The assessment brief restricts GenAI use to feedback/polishing only ("GenAI must not be used to generate code and report"), with no explicit carve-out for boilerplate/scaffolding. All code and planning artifacts produced during this exploration (environment code, notebooks, viewer, training scaffold, this report) should be treated as scratch material for the team to understand and substantially rewrite in their own words, not as final submission content, pending explicit clarification from the subject coordinator. The full chat transcript should be saved and declared per the brief's requirements (APA citation, 100-word reflection, screenshots).

## 11. Artifacts Produced So Far

- `Part-A_Project_Proposal.docx` (drafted, needs final team details)
- Two MVP notebooks (AgriFlight + a separate, unpursued restaurant-inventory idea)
- `agro_drone_env.py` / `agro_drone_gym_env.py` (environment)
- `scene_builder.py` (plantation visual scene)
- `agro_drone_viewer.py` (standalone desktop demo)
- `train_ppo_cleanrl.py`, `evaluate_trained_policy.py` (training/evaluation scaffold, deliberately unsolved)
- `team_structure_plan.md`, `algorithm_comparison.md` (planning references)
- `claude_code_scaffold_prompt.md` (prompt for reorganizing the prototype into a proper repo)
- Notion "AgriFlight-RL — Task Plan" database (32 detailed tasks)

## 12. Open Decisions for the Team

1. **3D full quadrotor vs. 2D simplified vs. hybrid** (visual in 3D, control simplified), the central open trade-off between technical depth and schedule risk.
2. **PPO/SAC vs. DQN**, contingent on the 3D/2D decision above.
3. Final role assignments and real names/IDs for the Notion task plan and Part-A submission.
