"""Inference latency benchmark — §3.6.3.

Measures per-frame GPU latency for YOLOv8m and RT-DETR-L at single-image
batch size (batch=1).  This is NOT the timing pilot (scripts/timing_pilot.py),
which measures training throughput (sec/epoch) for scheduling purposes.
This script measures inference latency for Table 3.11 / §3.6.3 reporting.

Protocol (§3.6.3 lines 797–819):
  - 50 warm-up passes excluded from statistics (kernel autotuning, memory alloc)
  - torch.cuda.synchronize() before AND after each timed pass
  - Statistics: mean, std, p95, p99 per-frame latency in ms
  - Real-time threshold: ≥30 FPS ↔ ≤33.3 ms mean latency
  - Primary device: RTX 3060 12 GB (ALIVE laboratory)
  - Also run on RTX 4050 Laptop 6 GB (reported separately per §3.6.3)
  - Latency additionally broken out by severity bin
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.util.config import PendingMeasurement

__all__ = [
    "benchmark_arch",
    "benchmark_by_severity",
]

# §3.6.3: real-time threshold
_RT_THRESHOLD_MS: float = 33.3

# Bins that require lum_band — mirrors evaluate.py's _LUM_DEPENDENT
_LUM_DEPENDENT = frozenset({"lum_moderate", "lum_severe", "compound_heavyrain_severelowlight"})


# ---------------------------------------------------------------------------
# Image path helper
# ---------------------------------------------------------------------------

def _img_path(ann_path: str, pool_root: Path) -> Path:
    """Reconstruct image path from ann_path (labels/…/*.txt → images/…/*.jpg).

    Self-contained so this script has no import dependency on dataset.py.
    ann_path is POSIX-relative to pool_root, as stored in the manifest.
    """
    img_rel = ann_path.replace("labels/", "images/", 1).replace(".txt", ".jpg")
    return pool_root / img_rel


# ---------------------------------------------------------------------------
# Severity filtering (mirrors evaluate.py _filter_severity)
# ---------------------------------------------------------------------------

def _filter_severity(df: pd.DataFrame, severity: str) -> pd.DataFrame:
    """Filter manifest to test-split rows matching severity.

    Raises PendingMeasurement for lum-dependent bins when lum_band has nulls,
    consistent with evaluate.py behaviour (t1/t2 not yet measured).
    """
    if severity in _LUM_DEPENDENT and df["lum_band"].isna().any():
        raise PendingMeasurement(
            f"lum_band contains nulls; t1/t2 must be measured before "
            f"benchmarking bin '{severity}'."
        )

    test = df[df["split_role"] == "test"].copy()

    if severity == "clear":
        mask = (test["rain_severity"] == "clear") & (
            test["lum_band"].isna() | (test["lum_band"] == "normal")
        )
    elif severity == "rain_light":
        mask = test["rain_severity"] == "rain_light"
    elif severity == "rain_moderate":
        mask = test["rain_severity"] == "rain_moderate"
    elif severity == "rain_heavy":
        mask = test["rain_severity"] == "rain_heavy"
    elif severity == "lum_moderate":
        mask = test["lum_band"] == "lum_moderate"
    elif severity == "lum_severe":
        mask = test["lum_band"] == "lum_severe"
    elif severity == "compound_heavyrain_severelowlight":
        mask = (test["rain_severity"] == "rain_heavy") & (test["lum_band"] == "lum_severe")
    else:
        raise ValueError(f"Unknown severity bin: {severity!r}")

    return test[mask]


# ---------------------------------------------------------------------------
# Image list from manifest subset
# ---------------------------------------------------------------------------

def _collect_images(
    subset: pd.DataFrame,
    pool_roots: dict[str, Path],
    n_limit: int | None,
) -> list[Path]:
    """Build list of existing image paths from a manifest subset."""
    paths: list[Path] = []
    for _, row in subset.iterrows():
        pool_root = pool_roots.get(str(row["pool"]))
        if pool_root is None:
            continue
        ann_rel = row.get("ann_path")
        if pd.notna(ann_rel) and ann_rel:
            img_abs = _img_path(str(ann_rel), pool_root)
        else:
            stem = str(row["frame_id"]).split("_", 1)[-1]
            img_abs = pool_root / "images" / f"{stem}.jpg"
        if img_abs.exists():
            paths.append(img_abs)
        if n_limit is not None and len(paths) >= n_limit:
            break
    return paths


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def benchmark_arch(
    weights: "str | Path",
    arch: str,
    image_paths: "list[Path]",
    *,
    device: str = "0",
    warmup: int = 50,
    imgsz: int = 640,
) -> dict:
    """Run single-image inference on image_paths, return latency statistics.

    Warms up with `warmup` forward passes (excluded from statistics).
    For each timing pass: torch.cuda.synchronize() → perf_counter → forward →
    torch.cuda.synchronize() → perf_counter. This is the protocol of §3.6.3.

    Returns:
        {
            "arch": arch,
            "weights": str(weights),
            "n_images": int,
            "mean_ms": float,
            "std_ms": float,
            "p95_ms": float,
            "p99_ms": float,
            "fps_mean": float,    # 1000 / mean_ms
            "passes_realtime": bool,  # mean_ms <= 33.3
            "device_name": str,
        }
    """
    from ultralytics import RTDETR, YOLO  # deferred — not always installed

    weights = Path(weights)
    if arch == "rtdetr_l":
        model = RTDETR(str(weights))
    else:
        model = YOLO(str(weights))

    # Load images once; cv2 reads BGR uint8 HWC — Ultralytics handles the rest
    imgs: list[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)

    if not imgs:
        raise RuntimeError("No readable images in image_paths")

    n = len(imgs)
    # Cycle through available images for warm-up so count isn't capped by n
    for i in range(warmup):
        model(imgs[i % n], verbose=False, device=device)

    # Timed passes — one image at a time (batch=1, §3.6.3)
    latencies_ms: list[float] = []
    for img in imgs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(img, verbose=False, device=device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    device_name = (
        torch.cuda.get_device_name(int(device)) if device.isdigit() else device
    )

    return {
        "arch":            arch,
        "weights":         str(weights),
        "n_images":        len(latencies_ms),
        "mean_ms":         mean_ms,
        "std_ms":          float(arr.std(ddof=1)),
        "p95_ms":          float(np.percentile(arr, 95)),
        "p99_ms":          float(np.percentile(arr, 99)),
        "fps_mean":        float(1000.0 / mean_ms),
        "passes_realtime": bool(mean_ms <= _RT_THRESHOLD_MS),
        "device_name":     device_name,
    }


# ---------------------------------------------------------------------------
# Per-severity benchmark
# ---------------------------------------------------------------------------

def benchmark_by_severity(
    weights: "str | Path",
    arch: str,
    manifest_path: Path,
    pool_roots: "dict[str, Path]",
    out_dir: Path,
    *,
    device: str = "0",
    warmup: int = 50,
    imgsz: int = 640,
    n_images: int | None = None,
) -> pd.DataFrame:
    """Benchmark latency for each of the 7 severity bins separately.

    Loads test-split images for each severity bin from the manifest.
    Skips lum-band-dependent bins gracefully if PendingMeasurement is raised
    (lum_band not yet measured — t1/t2 null).

    Returns DataFrame with columns: severity, mean_ms, std_ms, p95_ms, p99_ms,
    fps_mean, n_images.  Writes results to out_dir/latency_{arch}.json and
    out_dir/latency_{arch}.parquet.
    """
    df = pd.read_parquet(manifest_path, engine="pyarrow")

    severity_bins = [
        "clear",
        "rain_light",
        "rain_moderate",
        "rain_heavy",
        "lum_moderate",
        "lum_severe",
        "compound_heavyrain_severelowlight",
    ]

    rows: list[dict] = []
    for severity in severity_bins:
        try:
            subset = _filter_severity(df, severity)
        except PendingMeasurement as exc:
            print(f"[skip] {severity}: {exc}", file=sys.stderr)
            continue

        image_paths = _collect_images(subset, pool_roots, n_images)
        if not image_paths:
            print(f"[skip] {severity}: no images found", file=sys.stderr)
            continue

        print(f"[bench] {severity}  ({len(image_paths)} images)…")
        stats = benchmark_arch(
            weights=weights,
            arch=arch,
            image_paths=image_paths,
            device=device,
            warmup=warmup,
            imgsz=imgsz,
        )
        rows.append({
            "severity": severity,
            "mean_ms":  stats["mean_ms"],
            "std_ms":   stats["std_ms"],
            "p95_ms":   stats["p95_ms"],
            "p99_ms":   stats["p99_ms"],
            "fps_mean": stats["fps_mean"],
            "n_images": stats["n_images"],
        })

    result_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out_dir / f"latency_{arch}.parquet", index=False, engine="pyarrow")
    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Inference latency benchmark (§3.6.3). "
            "Measures single-image GPU latency, NOT training throughput."
        )
    )
    p.add_argument("--weights", required=True, type=Path,
                   help="Model weights (.pt)")
    p.add_argument("--arch", required=True, choices=["yolov8m", "rtdetr_l"],
                   help="Architecture identifier")
    p.add_argument("--manifest", type=Path, default=None,
                   dest="manifest",
                   help="Path to manifest.parquet (enables per-severity breakdown)")
    p.add_argument("--pool-a", type=Path, default=None, dest="pool_a")
    p.add_argument("--pool-b", type=Path, default=None, dest="pool_b")
    p.add_argument("--pool-c", type=Path, default=None, dest="pool_c")
    p.add_argument("--out", type=Path, default=Path("runs/latency"), dest="out")
    p.add_argument("--device", default="0")
    p.add_argument("--warmup", type=int, default=50,
                   help="Warm-up passes excluded from statistics (§3.6.3: 50)")
    p.add_argument("--n-images", type=int, default=None, dest="n_images",
                   help="Max images per severity bin (default: all)")
    return p


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    args = _build_parser().parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_roots: dict[str, Path] = {}
    if args.pool_a:
        pool_roots["A"] = args.pool_a
    if args.pool_b:
        pool_roots["B"] = args.pool_b
    if args.pool_c:
        pool_roots["C"] = args.pool_c

    # Overall benchmark — requires at least one pool with images
    if pool_roots and args.manifest:
        df_manifest = pd.read_parquet(args.manifest, engine="pyarrow")
        all_test = df_manifest[df_manifest["split_role"] == "test"]
        overall_paths = _collect_images(all_test, pool_roots, args.n_images)
    else:
        # Fallback: glob images from pool_a directly
        overall_paths = []
        if args.pool_a:
            overall_paths = list((args.pool_a / "images").glob("*.jpg"))
            if args.n_images:
                overall_paths = overall_paths[: args.n_images]

    if not overall_paths:
        print(
            "[error] No images found. Provide --manifest + at least one --pool-*, "
            "or ensure --pool-a/images/ contains .jpg files.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[overall] {len(overall_paths)} images, warmup={args.warmup}…")
    overall = benchmark_arch(
        weights=args.weights,
        arch=args.arch,
        image_paths=overall_paths,
        device=args.device,
        warmup=args.warmup,
    )

    by_severity_rows: list[dict] = []
    if args.manifest and pool_roots:
        sev_df = benchmark_by_severity(
            weights=args.weights,
            arch=args.arch,
            manifest_path=args.manifest,
            pool_roots=pool_roots,
            out_dir=out_dir,
            device=args.device,
            warmup=args.warmup,
            n_images=args.n_images,
        )
        by_severity_rows = sev_df.to_dict(orient="records")

    report = {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "device":               overall["device_name"],
        "arch":                 args.arch,
        "weights":              str(args.weights),
        "warmup_passes":        args.warmup,
        "realtime_threshold_ms": _RT_THRESHOLD_MS,
        "overall":              overall,
        "by_severity":          by_severity_rows,
    }

    json_path = out_dir / f"latency_{args.arch}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResults → {json_path}")
    print(
        f"Overall  mean={overall['mean_ms']:.2f} ms  "
        f"std={overall['std_ms']:.2f}  "
        f"p95={overall['p95_ms']:.2f}  "
        f"p99={overall['p99_ms']:.2f}  "
        f"FPS={overall['fps_mean']:.1f}  "
        f"realtime={'YES' if overall['passes_realtime'] else 'NO'}"
    )
