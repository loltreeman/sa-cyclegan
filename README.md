# SA-CycleGAN thesis — code

Implements the methodology in Chapter III. Nothing here is a fork of anything.

## Setup — CUDA (the real path)

The training GPU is the RTX 4050 Laptop (CUDA). Install torch FIRST, by hand,
from the PyTorch index. `pip install torch` alone gives a **CPU-only** wheel
on Windows (~124 MB); the CUDA build is ~2.5 GB.

    py -3.12 -m venv .venv-cuda
    .venv-cuda\Scripts\activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX

Get `cuXXX` from the selector at https://pytorch.org/get-started/locally/
(Stable / Windows / Pip / Python / your CUDA). Do not copy a version out of
a chat log or an old README — it changes.

Verify before doing anything else:

    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Must print a version ending in `+cuXXX`, `True`, and the RTX 4050.
**No `+cu` suffix means you installed the CPU wheel.** A ~124 MB download
means the index URL was wrong.

Then record what you actually got in `configs/base.yaml` under
`versions.torch`, and install the rest:

    pip install ultralytics==8.4.154 albumentations==2.0.8
    pip install -e ".[dev]"
    pytest -q

## Setup — ROCm (contingency only)

Only if the AMD RX 9060 XT materialises. Never mix ROCm and CUDA in one venv.
Python 3.12 only; wheels come from repo.radeon.com.

    1. install ROCm torch from repo.radeon.com
    2. pip install --no-deps ultralytics==8.4.154
    3. pip install cloudpickle nvidia-ml-py polars requests ultralytics-platform ultralytics-thop

Step 2 must be `--no-deps` or pip pulls CUDA torch and clobbers the ROCm install.

## Run the tests

    pytest -q

16 tests. They encode values that appear in Chapter III — if one breaks,
a number in the manuscript is wrong, or the code is.

## Layout

    configs/       base.yaml (all §3.3/§3.4 hyperparameters), thresholds.yaml (pending measurements)
    src/stats/     §3.6.4 — Nadeau-Bengio, Holm, bootstrap, effect size
    src/data/      §3.2   — manifest, partition, pools, oversampling, Condition B
    src/gan/       §3.3   — SA-CycleGAN
    src/detect/    §3.4   — detector training and evaluation
    src/util/      config, seeds, checkpoints, run registry
    scripts/       build_manifest.py, timing_pilot.py, run_all.py

## Two things that are deliberate, do not "fix" them

- `pyproject.toml` omits torch and ultralytics, so `pip install -e .` can
  never clobber a hand-built torch install.
- `src/util/config.py` RAISES `PendingMeasurement` on a null threshold rather
  than defaulting. A silent default would mis-stratify every result in
  Chapter IV.
