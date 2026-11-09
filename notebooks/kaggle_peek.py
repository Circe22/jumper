"""Train jumper.gesture_peek on a Kaggle GPU notebook, end to end.

Paste into one Kaggle code cell (Accelerator: GPU T4 x2, Internet: on):

    !git clone -b crab-peek https://github.com/Circe22/jumper.git /kaggle/working/jumper
    !python /kaggle/working/jumper/notebooks/kaggle_peek.py

and use "Save Version -> Save & Run All" so it runs with the browser closed. The
result is `/kaggle/working/peek_result.zip` (checkpoints, ONNX, log), in the
version's Output tab.

The install follows `jumper_colab.ipynb`: torch 2.9.1 from the CUDA wheel index
for this GPU, then the repository in editable mode. Training is the command
`tools/colab.py train` runs, without its Colab-only provenance and Drive steps.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

TASK = os.environ.get("PEEK_TASK", "jumper.gesture_peek")
NUM_ENVS = int(os.environ.get("PEEK_ENVS", "1024"))
ITERATIONS = int(os.environ.get("PEEK_ITERS", "2000"))

REPO = Path(__file__).resolve().parents[1]
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else REPO.parent
VENV = WORK / "venv"


def run(cmd: list, **kw) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main() -> int:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if not rows:
        raise SystemExit("No GPU: set Accelerator to GPU T4 x2 in the notebook settings")
    name, cap = (p.strip() for p in rows[0].rsplit(",", 1))
    print("GPU:", name, "compute capability", cap, flush=True)
    if float(cap) < 7.5:
        raise SystemExit(f"{name} is too old for the warp wheels; choose GPU T4 x2, not P100")

    if not VENV.exists():
        run([sys.executable, "-m", "venv", VENV])
    py = VENV / "bin" / "python"
    index = "cu126" if float(cap) < 8.0 else "cu128"
    run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-q", "torch==2.9.1", "torchvision==0.24.1",
         "--index-url", f"https://download.pytorch.org/whl/{index}"])
    run([py, "-m", "pip", "install", "-q", "-e", REPO])

    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", CUDA_VISIBLE_DEVICES="0")
    log = WORK / "peek_train.log"
    with open(log, "w") as f:
        proc = subprocess.Popen(
            [py, "scripts/train.py", "--task", TASK, "--model", "jumper",
             "--backend", "warp", "--device", "cuda:0", "--num_envs", str(NUM_ENVS),
             "--max-iterations", str(ITERATIONS), "--headless", "--no-tensorboard",
             "--logger", "tensorboard"],
            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            f.write(line)
            # Keep the notebook log readable: the iteration header and the headline numbers.
            if any(k in line for k in ("Learning iteration", "Mean reward", "Iteration time",
                                       "ETA", "Error", "Traceback", "[mjrl]")):
                print(line, end="", flush=True)
        code = proc.wait()

    runs = sorted((REPO / "logs" / "jumper" / TASK).glob("*/"), key=lambda p: p.stat().st_mtime)
    out = WORK / "peek_result"
    if runs:
        shutil.copytree(runs[-1], out / runs[-1].name, dirs_exist_ok=True)
    shutil.copy(log, out / log.name) if out.exists() else None
    if out.exists():
        print("result:", shutil.make_archive(str(WORK / "peek_result"), "zip", out), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
