"""
Build self-contained ("local") editions of the three experiment notebooks.

Each generated notebook bundles, inside its Setup cell:
  * the project package  src/agroflight/            (zip + base64)
  * the Skydio X2 model  mujoco_menagerie/skydio_x2/ (zip + base64)
so it runs on Google Colab, AWS SageMaker or a plain Jupyter install with no
GitHub access and no git clone. Only `pip install mujoco gymnasium` (and torch
etc. if the kernel lacks them) needs the internet.

Usage (from the repo root, after `make menagerie`):
    python scripts/build_local_notebooks.py

Re-run it whenever src/agroflight changes, so the bundled code matches main.
The source notebooks (notebooks/<alg>_<id>.ipynb) stay the single source of
truth for everything except the setup/packaging cells.
"""
import base64
import copy
import datetime
import io
import json
import os
import subprocess
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")
SRC_PKG = os.path.join(ROOT, "src", "agroflight")
MODEL_DIR = os.path.join(ROOT, "mujoco_menagerie", "skydio_x2")
MODEL_FILES = ["x2.xml", "LICENSE", "assets/X2_lowpoly.obj",
               "assets/X2_lowpoly_texture_SpinningProps_1024.png"]

NOTEBOOKS = ["ppo_25609227", "reinforce_25418551", "a2c_25399217"]


def zip_b64(files):
    """files: list of (arcname, abs_path) -> wrapped base64 string of a zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc, path in files:
            z.write(path, arc)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "\n".join(b64[i:i + 100] for i in range(0, len(b64), 100))


def package_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(SRC_PKG):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                out.append((os.path.relpath(p, os.path.dirname(SRC_PKG)), p))
    return sorted(out)


def model_files():
    out = []
    for rel in MODEL_FILES:
        p = os.path.join(MODEL_DIR, rel)
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}: run `make menagerie` first")
        out.append((os.path.join("skydio_x2", rel), p))
    return out


def git_short():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], cwd=ROOT).returncode != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


SETUP_MD = (
    "**2.1 Setup (bundled project).** One cell that unpacks the project code and the Skydio X2 "
    "drone model, and installs `mujoco` / `gymnasium` (plus `torch`, `matplotlib`, `pandas` if the "
    "kernel lacks them). Works on Google Colab, AWS SageMaker and a local Jupyter. Nothing to edit. "
    "It prints the platform and the version of the bundled code."
)

SETUP_CODE = '''#@title Setup: run this first (nothing to edit) { display-mode: "form" }
import os, sys, time, io, base64, zipfile, subprocess, shutil, importlib, importlib.util
NOTEBOOK_T0 = time.time()

IN_COLAB = "google.colab" in sys.modules
IN_SAGEMAKER = bool(
    os.environ.get("SAGEMAKER_INTERNAL_IMAGE_URI") or os.environ.get("SM_CURRENT_HOST")
    or os.path.isdir("/opt/ml/metadata") or os.path.expanduser("~") == "/home/sagemaker-user"
)
PLATFORM = "colab" if IN_COLAB else ("sagemaker" if IN_SAGEMAKER else "local")
BUNDLE_COMMIT = "__COMMIT__"
BUNDLE_BUILT = "__BUILT__"

REPO = "/content/agroflight-rl" if IN_COLAB else os.path.join(os.getcwd(), "agroflight-local")
os.makedirs(REPO, exist_ok=True)
os.chdir(REPO)

def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)

# Optional: a throwaway git repo so tools that call `git rev-parse` do not fail.
if shutil.which("git") and not os.path.isdir(os.path.join(REPO, ".git")):
    try:
        run(["git", "init", "-q"])
        run(["git", "-c", "user.name=bundle", "-c", "user.email=bundle@local",
             "commit", "-q", "--allow-empty", "-m", "bundle " + BUNDLE_COMMIT])
    except Exception:
        pass

# 1. Python dependencies (only what is missing).
_needed = {"mujoco": "mujoco", "gymnasium": "gymnasium", "torch": "torch",
           "matplotlib": "matplotlib", "pandas": "pandas"}
_missing = [pip for mod, pip in _needed.items() if importlib.util.find_spec(mod) is None]
if _missing:
    print("installing:", " ".join(_missing))
    run([sys.executable, "-m", "pip", "install", "-q"] + _missing)
    importlib.invalidate_caches()

# 2. Project code: src/agroflight (bundled, commit shown below).
_BUNDLE = """
__BUNDLE__
"""
SRC = os.path.join(REPO, "src")
shutil.rmtree(os.path.join(SRC, "agroflight"), ignore_errors=True)   # never mix with a stale copy
with zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(_BUNDLE.split())))) as z:
    z.extractall(SRC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# 3. Drone model: Skydio X2 from google-deepmind/mujoco_menagerie (Apache-2.0), bundled.
_MODEL = """
__MODEL__
"""
MENAGERIE = os.path.join(REPO, "mujoco_menagerie")
with zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(_MODEL.split())))) as z:
    z.extractall(MENAGERIE)
DRONE_XML = os.path.join(MENAGERIE, "skydio_x2", "x2.xml")
os.environ["AGRO_DRONE_XML"] = DRONE_XML

import mujoco, gymnasium, agroflight
print("platform     :", PLATFORM)
print("project code :", BUNDLE_COMMIT, "built", BUNDLE_BUILT, "(bundled inside this notebook)")
print("loaded from  :", os.path.dirname(agroflight.__file__))
print("drone model  :", DRONE_XML)
print("mujoco %s | gymnasium %s" % (mujoco.__version__, gymnasium.__version__))
'''

HOWTO_BOX = (
    '<div style="border-left:4px solid rgb(43,90,130);background:rgba(43,90,130,0.10);padding:10px 14px;'
    'border-radius:0 8px 8px 0;margin:6px 0"><b>How to use this notebook (self-contained edition).</b><ol>'
    "<li>Runs on <b>Google Colab</b>, <b>AWS SageMaker</b> (Studio / Notebook Instance, any PyTorch kernel) "
    "or a local Jupyter. No GitHub, no clone: the project code and the drone model are bundled in the "
    "<i>Setup</i> cell (Section 2.1). Run it first. Only <code>pip install</code> needs the internet.</li>"
    "<li>Read the yellow <i>Write here</i> boxes and the <code>TODO</code> cells. They are yours: the "
    "introduction, the theory, your code, the training and the discussion.</li>"
    "<li>Do not edit the Setup cell or the cells labelled <i>Shared, identical in all three notebooks</i>. "
    "The team comparison is only fair if they stay the same.</li>"
    "<li><i>Run all</i> works at any time and already produces the baseline numbers.</li>"
    "<li>Outputs go to <code>agroflight-local/results</code> and <code>agroflight-local/checkpoints</code> "
    "next to the notebook (<code>/content/agroflight-rl</code> on Colab). On SageMaker they persist in "
    "your home volume.</li>"
    "<li>When you are done, run the last cell (<i>Package my results</i>) and share the zip and this "
    "notebook with the team.</li></ol>Delete this box before final submission.</div>"
)

BADGE = ('  <div style="margin-top:12px;display:inline-block;background:rgba(255,255,255,.16);padding:4px 12px;'
         'border-radius:20px;font-size:13px">Self-contained edition: code and drone model bundled, runs on '
         'Colab, SageMaker or locally</div>\n')

HANDBACK_MD1 = '<a id="handback"></a>\n\n### Package my results'
HANDBACK_MD2 = ("Run the next cell when you are finished. It collects your results file and checkpoints into "
                "one zip (downloaded automatically in Colab; on SageMaker or locally the path is printed, "
                "right-click it in the file browser to download). Share the zip and this notebook with the team.")
HANDBACK_CODE = '''#@title Package my results (run when you are done)
import shutil, glob, json, datetime

alg = EXPERIMENT["algorithm"].lower()
out_dir = os.path.join(REPO, f"handback_{alg}_{EXPERIMENT['student_id']}")
shutil.rmtree(out_dir, ignore_errors=True)
os.makedirs(out_dir)

for pattern in (os.path.join(RESULTS_DIR, f"{alg}_{EXPERIMENT['student_id']}*.json"),
                os.path.join(CHECKPOINT_DIR, f"{alg}_{EXPERIMENT['student_id']}_*.pt")):
    for f in glob.glob(pattern):
        shutil.copy(f, out_dir)

with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
    json.dump(dict(algorithm=EXPERIMENT["algorithm"], student=EXPERIMENT["student_name"],
                   student_id=EXPERIMENT["student_id"], bundle_commit=BUNDLE_COMMIT, platform=PLATFORM,
                   packaged=datetime.datetime.now().isoformat(timespec="seconds"),
                   files=sorted(os.listdir(out_dir))), f, indent=2)

archive = shutil.make_archive(out_dir, "zip", out_dir)
print("packaged:", sorted(os.listdir(out_dir)))
if IN_COLAB:
    from google.colab import files
    files.download(archive)
else:
    print("zip is at", archive)'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def src(cell):
    return "".join(cell["source"])


def build(name, setup_code):
    with open(os.path.join(NB_DIR, name + ".ipynb")) as f:
        nb = json.load(f)
    cells = copy.deepcopy(nb["cells"])

    # Title: add the edition badge after the subtitle line.
    title = src(cells[0])
    marker = '<div style="height:1px'
    if "Self-contained edition" not in title and marker in title:
        i = title.index(marker)
        j = title.rfind("\n", 0, i) + 1
        cells[0] = md(title[:j] + BADGE + title[j:])

    # Contents: replace the template-status box with the how-to box.
    toc = src(cells[1])
    k = toc.find('<div style="border-left')
    cells[1] = md((toc[:k] if k >= 0 else toc + "\n\n") + HOWTO_BOX)

    # Setup: replace the git-clone bootstrap (markdown + code).
    i_boot = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "REPO_URL" in src(c))
    cells[i_boot - 1] = md(SETUP_MD)
    cells[i_boot] = code(setup_code)

    # After the shared helpers cell, record the bundle version instead of the git commit.
    i_shared = next(i for i, c in enumerate(cells) if "def new_results" in src(c))
    cells.insert(i_shared + 1, code('RESULTS["git_commit"] = BUNDLE_COMMIT + " (bundled)"'))

    # Reproducibility cell.
    for c in cells:
        s = src(c)
        if c["cell_type"] == "code" and 'print("git commit:", git_commit())' in s:
            s = s.replace('print("git commit:", git_commit())',
                          'print("git commit:", BUNDLE_COMMIT + " (bundled)")\nprint("platform  :", PLATFORM)')
            c["source"] = s.splitlines(keepends=True)

    cells += [md(HANDBACK_MD1), md(HANDBACK_MD2), code(HANDBACK_CODE)]
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["cells"] = cells

    out = os.path.join(NB_DIR, name + "_local.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return out


def main():
    commit = git_short()
    built = datetime.date.today().isoformat()
    setup_code = (SETUP_CODE.replace("__COMMIT__", commit).replace("__BUILT__", built)
                  .replace("__BUNDLE__", zip_b64(package_files()))
                  .replace("__MODEL__", zip_b64(model_files())))
    for name in NOTEBOOKS:
        out = build(name, setup_code)
        print(f"wrote {os.path.relpath(out, ROOT)}  ({os.path.getsize(out) / 1024:.0f} KB, code {commit})")


if __name__ == "__main__":
    main()
