"""Condition B augmentation pipeline (§3.4.5).

Applies Albumentations transforms to Pool A clear-weather frames to produce
synthetic rain and low-light variants for detector training augmentation.

Low-light parameter ranges are visual approximations calibrated to push mean
luma (µY) into the target luminance bins.  They will be verified empirically
over a sample before the full set is generated (§3.4.5: 'tuned so that the
mean luminance of the augmented output falls within the target luminance bins').
The exact ranges may be updated once t1/t2 are measured and thresholds.yaml
is populated.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import pandas as pd
import albumentations as A

from src.data.manifest import COLUMNS, load
from src.util.config import load_base

__all__ = ["AUGMENTERS", "augment_frame", "generate_condition_b"]

# ---------------------------------------------------------------------------
# Transform definitions — parameters fixed per §3.4.5 and Task description.
# rain_type values follow albumentations API ("drizzle", "heavy", "torrential").
# albumentations 2.0.8 merged slant_lower/slant_upper into slant_range: tuple.
# The intent (±10° and ±15° slant ranges) is preserved exactly.
# ---------------------------------------------------------------------------

AUGMENTERS: dict[str, A.BasicTransform] = {
    "rain_light": A.Compose([
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=15, drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=1,
            brightness_coefficient=0.9,
            rain_type="drizzle",
            p=1.0,
        ),
    ]),
    "rain_moderate": A.Compose([
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=20, drop_width=1,
            drop_color=(180, 180, 180),
            blur_value=3,
            brightness_coefficient=0.8,
            rain_type="heavy",
            p=1.0,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    ]),
    "rain_heavy": A.Compose([
        A.RandomRain(
            slant_range=(-15, 15),
            drop_length=30, drop_width=2,
            drop_color=(150, 150, 150),
            blur_value=5,
            brightness_coefficient=0.7,
            rain_type="torrential",
            p=1.0,
        ),
        A.GaussianBlur(blur_limit=(5, 7), p=0.7),
    ]),
    # Targets lum_moderate band (t1 ≤ µY < t2); gamma after brightness so the
    # combined effect is multiplicative rather than additive only.
    "lowlight_moderate": A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.4, -0.2),
            contrast_limit=(-0.1, 0.1),
            p=1.0,
        ),
        A.RandomGamma(gamma_limit=(70, 90), p=1.0),
    ]),
    # Targets lum_severe band (µY < t1).
    "lowlight_severe": A.Compose([
        A.RandomGamma(gamma_limit=(40, 65), p=1.0),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.5, -0.3),
            contrast_limit=(-0.1, 0.1),
            p=1.0,
        ),
    ]),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_frame(
    img: np.ndarray,
    condition: str,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Apply AUGMENTERS[condition] to a uint8 HWC RGB image.

    Parameters
    ----------
    img       : uint8 HWC RGB ndarray.
    condition : key in AUGMENTERS.
    seed      : if given, seeds numpy and Python random state before the
                transform so results are reproducible per-frame.

    Returns
    -------
    uint8 HWC RGB ndarray.
    """
    if condition not in AUGMENTERS:
        raise KeyError(f"Unknown condition '{condition}'. Valid: {list(AUGMENTERS)}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    result = AUGMENTERS[condition](image=img)
    return result["image"]


def generate_condition_b(
    manifest_path: Path,
    pool_a_root: Path,
    out_root: Path,
    n_aug: int,
    *,
    seed: int = 0,
    allocation: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate Condition B augmented frames and return a manifest DataFrame for them.

    Selects n_aug frames from Pool A (train split) according to allocation percentages,
    applies the corresponding augmentation, writes to out_root/images/train/ as JPEG,
    copies the source annotation to out_root/labels/train/ unchanged (bboxes transfer
    directly since augmentation is pixel-only).

    Returns a DataFrame with the same schema as manifest.py COLUMNS for the generated
    frames:
      - pool = "B" (synthetic augmentation)
      - rain_severity = the condition applied
      - rain_source = f"albumentations:{condition}"
      - frame_id = f"B_{source_frame_id}_{condition}"
      - ann_path = relative to out_root (labels/train/...)
      - split_role = "train"
      - fold = same as source frame
      - all measurement-gated columns (mu_y, is_ir, lum_band, rho_sat) = pd.NA
      - n_instances and class_counts = copied from source
      - sample_weight = 1.0  (oversampling recomputed later by apply_oversampling)
    """
    if allocation is None:
        cfg = load_base()
        allocation = cfg["condition_b"]["allocation"]

    total = sum(allocation.values())
    if abs(total - 1.0) >= 1e-6:
        raise ValueError(f"allocation values must sum to 1.0; got {total}")

    conditions = list(allocation.keys())

    # Per-condition frame counts; last condition absorbs rounding remainder.
    counts: dict[str, int] = {}
    running = 0
    for i, cond in enumerate(conditions):
        if i < len(conditions) - 1:
            n = round(n_aug * allocation[cond])
            counts[cond] = n
            running += n
        else:
            counts[cond] = n_aug - running

    # Load manifest and filter to Pool A train frames with known annotations.
    manifest = load(manifest_path)
    pool_a_train = manifest[
        (manifest["pool"] == "A") &
        (manifest["split_role"] == "train") &
        (manifest["ann_path"].notna())
    ].copy()

    if pool_a_train.empty:
        raise RuntimeError("No Pool A train frames with annotations found in manifest.")

    rng = random.Random(seed)

    out_images = out_root / "images" / "train"
    out_labels = out_root / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    frame_count = 0

    for cond in conditions:
        n_condition = counts[cond]
        pool_a_frames = pool_a_train.to_dict("records")

        # rain_severity is one of the rain_* keys or mapped from lowlight_* → "clear"
        # Rain conditions map directly; low-light frames keep their source rain severity
        # as "clear" since augmentation targets luma, not rain appearance.
        rain_sev = cond if cond.startswith("rain_") else "clear"

        sampled = rng.choices(pool_a_frames, k=n_condition)

        for frame_row in sampled:
            source_frame_id: str = frame_row["frame_id"]
            ann_path_rel: str = frame_row["ann_path"]   # POSIX, relative to pool_a_root

            # Derive image path from annotation path (labels/ → images/, .txt → .jpg).
            ann_posix = PurePosixPath(ann_path_rel)
            # Replace leading "labels" segment with "images" and change suffix.
            parts = list(ann_posix.parts)
            for i, part in enumerate(parts):
                if part == "labels":
                    parts[i] = "images"
                    break
            img_rel = PurePosixPath(*parts).with_suffix(".jpg")
            img_abs = pool_a_root / img_rel

            if not img_abs.exists():
                # Try .png fallback — Pool A sources may be PNG.
                img_abs = img_abs.with_suffix(".png")
                if not img_abs.exists():
                    continue

            img_bgr = cv2.imread(str(img_abs))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            frame_seed = seed ^ hash(f"{source_frame_id}_{cond}_{frame_count}") & 0xFFFFFFFF
            aug_rgb = augment_frame(img_rgb, cond, seed=frame_seed)

            source_stem = img_abs.stem
            out_stem = f"{source_stem}_{cond}"
            out_img_path = out_images / f"{out_stem}.jpg"
            out_lbl_path = out_labels / f"{out_stem}.txt"

            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_path), aug_bgr)

            # Annotation geometry is unchanged by pixel-only augmentation.
            src_ann_abs = pool_a_root / ann_path_rel
            if src_ann_abs.exists():
                out_lbl_path.write_bytes(src_ann_abs.read_bytes())

            ann_out_rel = str(PurePosixPath("labels/train") / f"{out_stem}.txt")

            rows.append({
                "frame_id":      f"B_{source_frame_id}_{cond}",
                "clip_id":       frame_row.get("clip_id", pd.NA),
                "event_id":      frame_row.get("event_id", pd.NA),
                "camera_id":     frame_row.get("camera_id", pd.NA),
                "timestamp":     pd.NA,
                "pool":          "B",
                "rain_severity": rain_sev,
                "rain_source":   f"albumentations:{cond}",
                "mu_y":          pd.NA,
                "is_ir":         pd.NA,
                "rho_sat":       pd.NA,
                "colour_div":    pd.NA,
                "lum_band":      pd.NA,
                "split_role":    "train",
                "fold":          frame_row.get("fold", pd.NA),
                "ann_path":      ann_out_rel,
                "n_instances":   frame_row.get("n_instances", pd.NA),
                "class_counts":  frame_row.get("class_counts", pd.NA),
                "sample_weight": 1.0,
            })

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  [{frame_count}/{n_aug}] {cond}: wrote {out_stem}.jpg")

    df = pd.DataFrame(rows)
    for col, dtype in COLUMNS.items():
        if col not in df.columns:
            df[col] = pd.array([pd.NA] * len(df), dtype=dtype if dtype != "object" else object)
        elif dtype != "object":
            try:
                df[col] = df[col].astype(dtype)
            except (TypeError, ValueError):
                pass
    df = df[[c for c in COLUMNS if c in df.columns]]

    print(f"Condition B: wrote {len(df)} frames to {out_root}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Condition B augmented frames from Pool A."
    )
    parser.add_argument("--manifest",  required=True, type=Path, help="Path to manifest.parquet")
    parser.add_argument("--pool-a",    required=True, type=Path, help="Pool A root (images/ + labels/)")
    parser.add_argument("--out",       required=True, type=Path, help="Output root for Condition B frames")
    parser.add_argument("--n-aug",     required=True, type=int,  help="Total augmented frames to generate")
    parser.add_argument("--seed",      default=0,     type=int,  help="RNG seed (default 0)")
    args = parser.parse_args()

    df = generate_condition_b(
        manifest_path=args.manifest,
        pool_a_root=args.pool_a,
        out_root=args.out,
        n_aug=args.n_aug,
        seed=args.seed,
    )

    out_parquet = args.out.parent / "condition_b_manifest.parquet"
    df.to_parquet(out_parquet, index=False, engine="pyarrow")
    print(f"Manifest written to {out_parquet} ({len(df)} rows)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
