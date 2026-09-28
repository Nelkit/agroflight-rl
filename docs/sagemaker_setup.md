# Running the notebooks on AWS SageMaker

The experiment notebooks (`notebooks/ppo_25609227.ipynb`, `reinforce_25418551.ipynb`,
`a2c_25399217.ipynb`) run on SageMaker straight from a clone of this repo. There is
nothing to bundle and nothing to regenerate: `git pull` always gives you the current `main`.

Checked on our SageMaker account (28 Sep 2026): internet access (PyPI, GitHub) and `git` are available.

## 1. Create a GitHub token (one-off, per person)

The repo is **private**, so SageMaker needs a *personal access token* to clone it.

1. GitHub → your avatar (top right) → **Settings** → **Developer settings** →
   **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
2. **Token name:** `sagemaker-agroflight` · **Expiration:** 30 days (covers the project, due 23 Oct).
3. **Resource owner:** `Nelkit` · **Repository access:** *Only select repositories* → `agroflight-rl`.
   If `Nelkit` does not appear as an owner, check with Nelkit that you are a collaborator on the repo.
   A *classic* token with the `repo` scope also works.
4. **Permissions → Repository permissions → Contents: Read and write**. Write access lets you push your results.
5. **Generate token** and copy it (`github_pat_...`). GitHub shows it only once. Keep it in a password manager.

> Never paste the token into a notebook cell, a commit or a chat. If it leaks, revoke it on the same page and create a new one.

## 2. Clone the repo in SageMaker (one-off)

Open a **Terminal** (SageMaker Studio / JupyterLab: *File → New → Terminal*):

```bash
cd ~
git clone https://<YOUR_TOKEN>@github.com/Nelkit/agroflight-rl.git
cd agroflight-rl
git remote set-url origin https://github.com/Nelkit/agroflight-rl.git   # keep the token out of .git/config
git config user.name  "Your Name"
git config user.email "your.email@student.uts.edu.au"
git config credential.helper store       # the next pull/push asks once, then remembers
git checkout -b <yourname>/<topic>       # one branch per person, e.g. juan/reinforce-l0
```

When `git pull` or `git push` asks for credentials, the *username* is your GitHub user and the *password* is the token.

## 3. Run a notebook

1. Open `~/agroflight-rl/notebooks/<your notebook>.ipynb`.
2. Pick a **PyTorch** kernel. CPU is enough for Stage 1 (L0).
3. Run the cells in order. Cell 2.1 (Bootstrap) detects the clone and does the rest:
   - fetches the Skydio X2 model (sparse clone of the public `google-deepmind/mujoco_menagerie`, a few MB);
   - runs `pip install -e ".[train]"` only if `mujoco`, `gymnasium`, `torch`, `matplotlib` or `pandas` is missing;
   - prints the platform, branch and commit it is running.

## 4. Day-to-day

| Task | Command (Terminal, inside `~/agroflight-rl`) |
| --- | --- |
| Get the latest code | `git pull origin main` |
| Save your results | `git add results/*.json notebooks/<your notebook>.ipynb && git commit -m "..." && git push -u origin <your branch>` |
| Merge into `main` | Open a pull request on GitHub |

- Results JSON files go to `results/` and are committed.
- Checkpoints (`checkpoints/*.pt`) are git-ignored. They stay in your SageMaker home volume. Share the best one with the team separately (e.g. Google Drive or Teams).
- Stop the SageMaker instance/space when you are not training. It is billed while running.

## Other platforms

- **Google Colab:** open the notebook from GitHub or upload it. Cell 2.1 clones the repo itself.
  Store the token in Colab *Secrets* (key icon on the left → add `GITHUB_TOKEN`, enable notebook access).
  Otherwise the cell asks for it with a hidden prompt.
- **Local Jupyter:** `make setup && make menagerie`, then open the notebook from the repo.
