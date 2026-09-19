"""Timing pilot — Step 29. §3.7.1.

Extracts ~N evenly-spaced frames from alive_footage/g29_demo.mp4, writes
synthetic YOLO-format labels, then trains YOLOv8m and RT-DETR-L for one
epoch each at imgsz 640.

Labels are synthetic: 2–4 random boxes per frame, classes drawn uniformly
from the 6-class taxonomy. The computational profile is dominated by the
network forward + backward + optimiser step, not the loss, so synthetic
labels give the same timing and memory figures as real ones.
This is a timing measurement — the weights produced are meaningless.

Records per model:
  - wall-clock seconds for the training epoch (perf_counter + CUDA sync)
  - peak GPU memory allocated  (torch.cuda.max_memory_allocated)
  - peak GPU memory reserved   (torch.cuda.max_memory_reserved)

Memory counter is reset immediately before model.train() so the reported
figure is the true high-water mark: weights + activations + gradients +
optimiser state all resident simultaneously.

Writes runs/timing_pilot.json with the same provenance fields as
scripts/model_complexity.py.

Usage:
    python scripts/timing_pilot.py
    python scripts/timing_pilot.py --frames 500 --batch 4
    python scripts/timing_pilot.py --video path/to/other.mp4
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import ultralytics
import yaml
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "alive_footage" / "g29_demo.mp4"
FRAMES_DIR = ROOT / "runs" / "timing_pilot_frames"
DATASET_YAML = FRAMES_DIR / "dataset.yaml"

# §3.5.1 — 6-class taxonomy. Names are placeholders; nc is what shapes the
# detection head. Replace with the real class list before training proper.
NC = 6
CLASS_NAMES = [f"class{i}" for i in range(NC)]


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, n: int, out_dir: Path) -> int:
    """Write n evenly-spaced JPEG frames from video_path into out_dir.
    Returns the number of frames actually written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Could not read frame count from {video_path}")

    indices = sorted(set(
        round(i * (total - 1) / (n - 1)) for i in range(n)
    )) if n > 1 else [0]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        dst = out_dir / f"frame_{idx:07d}.jpg"
        cv2.imwrite(str(dst), frame)
        written += 1

    cap.release()
    return written


# ---------------------------------------------------------------------------
# Synthetic labels
# ---------------------------------------------------------------------------

def write_synthetic_labels(images_dir: Path, labels_dir: Path,
                            rng: random.Random) -> None:
    """Write a YOLO-format .txt label file alongside each image."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    for img in sorted(images_dir.glob("*.jpg")):
        n_boxes = rng.randint(2, 4)
        lines = []
        for _ in range(n_boxes):
            cls = rng.randrange(NC)
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            w = rng.uniform(0.05, 0.4)
            h = rng.uniform(0.05, 0.4)
            # Clamp so the box stays inside the image.
            cx = max(w / 2, min(1 - w / 2, cx))
            cy = max(h / 2, min(1 - h / 2, cy))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        label_path = labels_dir / (img.stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset YAML
# ---------------------------------------------------------------------------

def write_dataset_yaml(path: Path, images_train: Path) -> None:
    doc = {
        "path": str(FRAMES_DIR),
        "train": str(images_train.relative_to(FRAMES_DIR)),
        "val": str(images_train.relative_to(FRAMES_DIR)),
        "nc": NC,
        "names": CLASS_NAMES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(doc, allow_unicode=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hyperparameters from base.yaml
# ---------------------------------------------------------------------------

def load_cfg() -> dict:
    cfg_path = ROOT / "configs" / "base.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Timing a single model
# ---------------------------------------------------------------------------

def time_model(loader, weights: str, train_kwargs: dict) -> dict:
    """Load model, run one training epoch, return timing + memory results."""
    model = loader(weights)

    # Reset the peak counter before train() so the reported figure is the
    # true high-water mark over the entire epoch, including model weights,
    # optimizer state, and peak activation batch — not just activations alone.
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model.train(**train_kwargs)
    torch.cuda.synchronize()   # ensure all CUDA work is complete before stop
    elapsed = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    return {
        "seconds_per_epoch": round(elapsed, 2),
        "peak_memory_allocated_gb": round(peak_alloc / 1024 ** 3, 3),
        "peak_memory_reserved_gb": round(peak_reserved / 1024 ** 3, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=500,
                        help="number of frames to extract (default: 500)")
    parser.add_argument("--batch", type=int, default=4,
                        help="training batch size (default: 4)")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                        help="path to source video")
    parser.add_argument("--keep-frames", action="store_true",
                        help="do not delete extracted frames after timing")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}")
        return 1

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device — timing on CPU is not meaningful for §3.7.1")
        return 1

    cfg = load_cfg()
    det = cfg["detectors"]
    shared = det["shared"]
    imgsz = det["imgsz"]

    print(f"torch       {torch.__version__}")
    print(f"ultralytics {ultralytics.__version__}")
    print(f"device      {torch.cuda.get_device_name(0)}")
    print(f"video       {args.video}")
    print(f"frames      {args.frames}  batch {args.batch}  imgsz {imgsz}")
    print()

    # ---------------------------------------------------------------- frames
    images_dir = FRAMES_DIR / "images" / "train"
    labels_dir = FRAMES_DIR / "labels" / "train"

    if images_dir.exists():
        shutil.rmtree(images_dir)
    if labels_dir.exists():
        shutil.rmtree(labels_dir)

    print("Extracting frames …", flush=True)
    n_written = extract_frames(args.video, args.frames, images_dir)
    print(f"  wrote {n_written} frames to {images_dir.relative_to(ROOT)}")

    rng = random.Random(cfg["seed"])
    write_synthetic_labels(images_dir, labels_dir, rng)
    print(f"  wrote synthetic labels to {labels_dir.relative_to(ROOT)}")

    write_dataset_yaml(DATASET_YAML, images_dir)
    print(f"  dataset yaml: {DATASET_YAML.relative_to(ROOT)}\n")

    # ---------------------------------------------------------------- runs
    # project must be an absolute path; a relative value is joined onto
    # Ultralytics' global settings.runs_dir, giving a doubly-nested path.
    project_dir = str(ROOT / "runs")

    models = [
        {
            "label": "YOLOv8m",
            "loader": YOLO,
            "weights": det["yolov8m"]["weights"],
            "extra": {
                "optimizer": det["yolov8m"]["optimizer"],
                "lr0":       det["yolov8m"]["lr0"],
                "momentum":  det["yolov8m"]["momentum"],
                "weight_decay": det["yolov8m"]["weight_decay"],
            },
        },
        {
            "label": "RT-DETR-L",
            "loader": RTDETR,
            "weights": det["rtdetr_l"]["weights"],
            "extra": {
                "optimizer": det["rtdetr_l"]["optimizer"],
                "lr0":       det["rtdetr_l"]["lr0"],
                "weight_decay": det["rtdetr_l"]["weight_decay"],
            },
        },
    ]

    results = []
    for m in models:
        print(f"─── {m['label']} ───────────────────────────────", flush=True)
        train_kwargs = dict(
            data=str(DATASET_YAML),
            epochs=1,
            imgsz=imgsz,
            batch=args.batch,
            deterministic=shared["deterministic"],   # False — §3.7.3
            warmup_epochs=shared["warmup_epochs"],
            cos_lr=shared["cos_lr"],
            lrf=shared["lrf"],
            mosaic=shared["mosaic"],
            mixup=shared["mixup"],
            save=False,          # skip checkpoint I/O so disk speed doesn't
            plots=False,         # bias the timing measurement
            patience=0,          # disable early stopping for single epoch
            workers=0,           # Windows: DataLoader workers > 0 can hang
            project=project_dir,
            name=f"timing_{m['label'].lower().replace('-', '_').replace(' ', '_')}",
            verbose=True,
            **m["extra"],
        )
        timing = time_model(m["loader"], m["weights"], train_kwargs)
        print(f"\n  seconds / epoch         : {timing['seconds_per_epoch']:.1f} s")
        print(f"  peak memory allocated   : {timing['peak_memory_allocated_gb']:.3f} GB")
        print(f"  peak memory reserved    : {timing['peak_memory_reserved_gb']:.3f} GB")
        print()
        results.append({"model": m["label"], "weights": m["weights"], **timing,
                        "batch": args.batch, "imgsz": imgsz})

    # ---------------------------------------------------------------- JSON
    out = ROOT / "runs" / "timing_pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_video": str(args.video.relative_to(ROOT)),
        "n_frames": n_written,
        "batch": args.batch,
        "imgsz": imgsz,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "note": (
            "Labels synthetic (random boxes). save=False, plots=False to "
            "avoid I/O bias. workers=0 for Windows DataLoader stability. "
            "Peak memory reset immediately before model.train() — includes "
            "weights + activations + gradients + optimiser state."
        ),
        "models": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written: {out.relative_to(ROOT)}")

    # ---------------------------------------------------------------- schedule guidance
    y = next(r for r in results if "YOLO" in r["model"])
    r = next(r for r in results if "DETR" in r["model"])
    print("\n── Schedule guidance (§3.7.1) ──────────────────────")
    for row in results:
        epochs_100 = row["seconds_per_epoch"] * 100 / 3600
        print(f"  {row['model']:<12}  {row['seconds_per_epoch']:.1f} s/epoch  "
              f"→  100 ep ≈ {epochs_100:.1f} h   "
              f"peak {row['peak_memory_allocated_gb']:.2f} GB")

    budget_gb = 6.0
    over = [r for r in results if r["peak_memory_allocated_gb"] > budget_gb]
    if over:
        names = ", ".join(r["model"] for r in over)
        print(f"\n  WARNING: {names} exceed {budget_gb} GB allocated on this card.")
        print("  Consider reducing batch size (--batch 2) and re-running.")
    else:
        print(f"\n  Both architectures fit within {budget_gb} GB at batch {args.batch}.")

    if not args.keep_frames:
        shutil.rmtree(FRAMES_DIR)
        print(f"\n  Extracted frames deleted. Pass --keep-frames to retain them.")
    else:
        print(f"\n  Frames kept at {FRAMES_DIR.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
