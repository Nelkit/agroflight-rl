# Notebooks

One experiment notebook per team member and algorithm, all on the same
environment (`AgroDrone-v0`), reward and evaluation protocol:

| Notebook | Algorithm | Owner |
| -------- | --------- | ----- |
| `ppo_25609227.ipynb` | PPO (CleanRL-based) | Nelkit Chavez |
| `reinforce_25418551.ipynb` | REINFORCE | Juan David Rodriguez |
| `a2c_25399217.ipynb` | A2C | Panzhao Li |
| `comparison.ipynb` | All three vs the baselines (reads `results/`, trains nothing) | Team |

The cells labelled *Shared, identical in all three notebooks* (results schema,
checkpoints, evaluation) must stay identical so one comparison notebook can read
all three results files. The environment itself lives in `src/agroflight/envs/`:
edit the package, never a notebook copy of it.

**Running them:** open the notebook from a clone of this repo and run cell 2.1
(Bootstrap). On AWS SageMaker, clone the repo first with a GitHub token; see
[`docs/sagemaker_setup.md`](../docs/sagemaker_setup.md).

> Note on rendering: MuJoCo rendering inside a notebook needs a headless GL
> backend (EGL/OSMesa) and is platform-dependent. For a live view use the
> desktop viewer instead: `make viewer`.
