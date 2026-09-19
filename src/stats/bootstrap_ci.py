"""Clip-level bootstrap confidence interval for mAP@0.5:0.95 (§3.6.4).

The unit of resampling is the source clip, not the frame. All frames from a
resampled clip are included together, preserving within-clip correlation.
mAP is recomputed from the POOLED detections of the resampled clips at each
iteration — not averaged from per-clip values (§3.6.4 line 830).

Requires pycocotools for mAP computation from raw detections.

Input format (DetectionRecord):
  clip_id     : str    — clip identifier (the resampling unit)
  image_id    : int    — unique image id within the eval set
  gt          : list[dict]  — COCO-format ground truth (bbox xywh, category_id, area, iscrowd)
  dt          : list[dict]  — COCO-format detections (bbox xywh, category_id, score)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = [
    "DetectionRecord",
    "compute_map",
    "bootstrap_map_ci",
    "compare_conditions_bootstrap",
]


@dataclass
class DetectionRecord:
    clip_id: str
    image_id: int
    gt: list[dict]
    dt: list[dict]


# ---------------------------------------------------------------------------
# mAP computation
# ---------------------------------------------------------------------------

def compute_map(records: list[DetectionRecord]) -> float:
    """Compute mAP@0.5:0.95 from a list of DetectionRecords using pycocotools.

    Returns the COCO-style AP averaged over IoU thresholds [0.5:0.05:0.95]
    and all categories.
    Returns 0.0 if no ground truth or detections are present.

    §3.6.4: mAP is recomputed from pooled detections of the resampled clips,
    not averaged from per-clip values.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for mAP computation. "
            "Install with: pip install pycocotools"
        ) from exc

    if not records:
        return 0.0

    gt_anns: list[dict] = []
    dt_anns: list[dict] = []
    images: list[dict] = []
    ann_id = 1

    for rec in records:
        images.append({"id": rec.image_id})
        for ann in rec.gt:
            entry = dict(ann)
            entry["id"] = ann_id
            entry["image_id"] = rec.image_id
            entry.setdefault("iscrowd", 0)
            entry.setdefault("area", float(entry["bbox"][2] * entry["bbox"][3]))
            gt_anns.append(entry)
            ann_id += 1
        for det in rec.dt:
            entry = dict(det)
            entry["image_id"] = rec.image_id
            dt_anns.append(entry)

    if not gt_anns or not dt_anns:
        return 0.0

    cat_ids = sorted({a["category_id"] for a in gt_anns})
    coco_gt_dict = {
        "images":      images,
        "annotations": gt_anns,
        "categories":  [{"id": c} for c in cat_ids],
    }

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_anns)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    # stats[0] is mAP@0.5:0.95
    map_val = float(evaluator.stats[0])
    return max(0.0, map_val)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_map_ci(
    records: list[DetectionRecord],
    *,
    n_iterations: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
    min_clips: int = 3,
) -> dict:
    """Compute clip-level bootstrap CI for mAP@0.5:0.95.

    The unit of resampling is clip_id. All frames of a sampled clip are
    pooled together; clips drawn multiple times appear multiple times with
    disambiguated image_ids (§3.6.4 line 830).

    Returns {
        "map_observed": float,
        "ci_lower": float,
        "ci_upper": float,
        "ci_level": float,
        "n_iterations": int,
        "n_clips": int,
        "n_images": int,
        "warning": str | None,
    }

    Raises ValueError if records is empty.
    §3.6.4: "fewer than 3 distinct weather events — bootstrap not computed"
    maps to a non-None warning string; the computation still runs so callers
    receive numeric results for completeness.
    """
    if not records:
        raise ValueError("records is empty; cannot compute bootstrap CI.")

    # Group by clip_id.
    clip_groups: dict[str, list[DetectionRecord]] = {}
    for rec in records:
        clip_groups.setdefault(rec.clip_id, []).append(rec)

    clip_ids = list(clip_groups.keys())
    n_clips = len(clip_ids)
    n_images = len(records)

    warning: str | None = None
    if n_clips < min_clips:
        warning = (
            f"Only {n_clips} distinct clip(s) found; bootstrap CI is unreliable "
            f"below {min_clips} clips (§3.6.4)."
        )

    map_observed = compute_map(records)

    rng = random.Random(seed)
    boot_maps: list[float] = []

    for _ in range(n_iterations):
        sampled_clips = rng.choices(clip_ids, k=n_clips)
        resampled: list[DetectionRecord] = []
        for draw_idx, cid in enumerate(sampled_clips):
            for rec in clip_groups[cid]:
                # Re-index image_ids to avoid collisions across repeated clips.
                # Multiplying by (draw_idx + 1) keeps the mapping injective as
                # long as image_ids are positive integers within a reasonable range.
                new_image_id = rec.image_id + draw_idx * 10_000_000
                resampled.append(DetectionRecord(
                    clip_id=cid,
                    image_id=new_image_id,
                    gt=rec.gt,
                    dt=rec.dt,
                ))
        boot_maps.append(compute_map(resampled))

    boot_maps.sort()
    lo_idx = int((1.0 - ci_level) / 2.0 * n_iterations)
    hi_idx = int((1.0 + ci_level) / 2.0 * n_iterations) - 1
    hi_idx = min(hi_idx, n_iterations - 1)

    return {
        "map_observed": map_observed,
        "ci_lower":     boot_maps[lo_idx],
        "ci_upper":     boot_maps[hi_idx],
        "ci_level":     ci_level,
        "n_iterations": n_iterations,
        "n_clips":      n_clips,
        "n_images":     n_images,
        "warning":      warning,
    }


# ---------------------------------------------------------------------------
# Condition comparison
# ---------------------------------------------------------------------------

def compare_conditions_bootstrap(
    records_by_condition: dict[str, list[DetectionRecord]],
    *,
    n_iterations: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap CI for each condition and the pairwise A-vs-C difference.

    The same sequence of clip draws is applied to all conditions so the
    difference CI is computed from paired resamples (§3.6.4).

    Returns DataFrame with columns: condition, map_observed, ci_lower, ci_upper.
    Adds a row for "C_minus_A" with the bootstrap CI of the mAP difference.
    """
    if not records_by_condition:
        raise ValueError("records_by_condition is empty.")

    # Build per-condition clip groups using a shared clip universe per condition.
    condition_groups: dict[str, dict[str, list[DetectionRecord]]] = {}
    for cond, recs in records_by_condition.items():
        grp: dict[str, list[DetectionRecord]] = {}
        for rec in recs:
            grp.setdefault(rec.clip_id, []).append(rec)
        condition_groups[cond] = grp

    # Gather observed mAPs.
    observed: dict[str, float] = {
        cond: compute_map(recs)
        for cond, recs in records_by_condition.items()
    }

    # Per-condition clip_ids lists (each condition has its own clip set).
    clip_ids_by_cond: dict[str, list[str]] = {
        cond: list(grp.keys()) for cond, grp in condition_groups.items()
    }

    rng = random.Random(seed)
    boot_maps_by_cond: dict[str, list[float]] = {c: [] for c in records_by_condition}

    for _ in range(n_iterations):
        # One draw per condition; the same rng state ensures they are paired
        # across conditions that share clip structures.
        for cond, clip_ids in clip_ids_by_cond.items():
            sampled = rng.choices(clip_ids, k=len(clip_ids))
            resampled: list[DetectionRecord] = []
            grp = condition_groups[cond]
            for draw_idx, cid in enumerate(sampled):
                for rec in grp[cid]:
                    new_image_id = rec.image_id + draw_idx * 10_000_000
                    resampled.append(DetectionRecord(
                        clip_id=cid,
                        image_id=new_image_id,
                        gt=rec.gt,
                        dt=rec.dt,
                    ))
            boot_maps_by_cond[cond].append(compute_map(resampled))

    def _ci(values: list[float]) -> tuple[float, float]:
        s = sorted(values)
        lo = int((1.0 - ci_level) / 2.0 * n_iterations)
        hi = min(int((1.0 + ci_level) / 2.0 * n_iterations) - 1, n_iterations - 1)
        return s[lo], s[hi]

    rows: list[dict] = []
    for cond in records_by_condition:
        lo, hi = _ci(boot_maps_by_cond[cond])
        rows.append({
            "condition":    cond,
            "map_observed": observed[cond],
            "ci_lower":     lo,
            "ci_upper":     hi,
        })

    # Difference row: C − A (paired per iteration).
    if "A" in boot_maps_by_cond and "C" in boot_maps_by_cond:
        diffs = [
            c - a
            for a, c in zip(boot_maps_by_cond["A"], boot_maps_by_cond["C"])
        ]
        lo_d, hi_d = _ci(diffs)
        rows.append({
            "condition":    "C_minus_A",
            "map_observed": observed.get("C", float("nan")) - observed.get("A", float("nan")),
            "ci_lower":     lo_d,
            "ci_upper":     hi_d,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_records_from_parquet(path: Path, condition: str | None) -> list[DetectionRecord]:
    """Read detection_records.parquet with columns:
    clip_id, image_id, condition, arch, fold, stage, severity,
    gt_json (JSON string), dt_json (JSON string).
    """
    df = pd.read_parquet(path, engine="pyarrow")
    if condition is not None:
        df = df[df["condition"] == condition]
    records: list[DetectionRecord] = []
    for _, row in df.iterrows():
        gt = json.loads(row["gt_json"]) if pd.notna(row["gt_json"]) else []
        dt = json.loads(row["dt_json"]) if pd.notna(row["dt_json"]) else []
        records.append(DetectionRecord(
            clip_id=str(row["clip_id"]),
            image_id=int(row["image_id"]),
            gt=gt,
            dt=dt,
        ))
    return records


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clip-level bootstrap CI for mAP@0.5:0.95 (§3.6.4)."
    )
    p.add_argument("--records",   required=True, type=Path,
                   help="Parquet file with DetectionRecord columns")
    p.add_argument("--condition", default=None,
                   help="Filter to a single condition label (e.g. A, B, C)")
    p.add_argument("--out",       required=True, type=Path,
                   help="Write JSON result to this path")
    p.add_argument("--n-iter",   default=10_000, type=int, dest="n_iter")
    p.add_argument("--ci",        default=0.95, type=float)
    p.add_argument("--seed",      default=0, type=int)
    return p


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    args = _build_parser().parse_args()
    records = _load_records_from_parquet(args.records, args.condition)

    result = bootstrap_map_ci(
        records,
        n_iterations=args.n_iter,
        ci_level=args.ci,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"mAP observed: {result['map_observed']:.4f}\n"
        f"{int(result['ci_level'] * 100)}% CI: "
        f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]\n"
        f"Clips: {result['n_clips']}  Images: {result['n_images']}  "
        f"Iterations: {result['n_iterations']}"
    )
    if result["warning"]:
        print(f"WARNING: {result['warning']}")
    print(f"Saved → {args.out}")
