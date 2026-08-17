# Notebooks

Colab-ready notebook from the exploration phase, kept for reference and for the
"how we got here" narrative in the report.

- **`agro_drone_MVP.ipynb`** — the original demonstration of the crop-row drone
  environment (state/action/reward design + a runnable physics simulation).
  Superseded by the packaged env in `src/agroflight/envs/`: treat the package
  as the source of truth and this notebook as a historical artefact. If you
  edit the environment, edit the package, not this file.

> Note on rendering: MuJoCo rendering inside a notebook needs a headless GL
> backend (EGL/OSMesa) and is platform-dependent. For a live view use the
> desktop viewer instead: `make viewer`.
