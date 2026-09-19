"""Evaluation harness — §3.5.1.

Runs Ultralytics .val() on held-out test frames stratified by the seven
severity bins defined in §3.5.1.  420 evaluations = 60 checkpoints × 7 bins.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.util.config import PendingMeasurement, load_base

__all__ = [
    "SEVERITY_BINS",
    "build_eval_yaml",
    "evaluate_checkpoint",
    "evaluate_all",
]

# Fallback if base.yaml is unreachable at import time.
SEVERITY_BINS: list[str] = [
    "clear",
    "rain_light",
    "rain_moderate",
    "rain_heavy",
    "lum_moderate",
    "lum_severe",
    "compound_heavyrain_severelowlight",
]

# Bins that require lum_band to be non-null.
_LUM_DEPENDENT = frozenset({"lum_moderate", "lum_severe", "compound_heavyrain_severelowlight"})

# Parsed from checkpoint path; groups: stage / condition / arch / fold.
_CKPT_RE = re.compile(
    r"stage(?P<stage>[12])_(?P<condition>[A-C])_(?P<arch>yolov8m|rtdetr_l)_fold(?P<fold>\d+)"
)


def _severity_bins_from_config() -> list[str]:
    try:
        cfg = load_base()
        return cfg["design"]["severity_bins"]
    except Exception:
        return SEVERITY_BINS


def _filter_severity(df: pd.DataFrame, severity: str) -> pd.DataFrame:
    """Return rows matching severity rules from §3.5.1.

    Raises PendingMeasurement if lum_band has any nulls for lum-dependent bins,
    indicating that t1/t2 have not been measured yet.
    """
    if severity in _LUM_DEPENDENT and df["lum_band"].isna().any():
        raise PendingMeasurement(
            f"lum_band contains nulls; t1/t2 thresholds must be measured before "
            f"evaluating bin '{severity}'.  Set them in configs/thresholds.yaml "
            f"and recompute lum_band via derive_lum_band()."
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
        # rain_heavy regardless of lum_band (compound frames included here too)
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


def build_eval_yaml(
    severity: str,
    manifest_path: Path,
    pool_roots: dict[str, Path],
    out_dir: Path,
) -> Path:
    """Write a YOLO dataset.yaml containing only frames matching `severity`.

    Severity matching rules (from §3.5.1):
    - "clear"        : rain_severity=="clear" and lum_band=="normal" (or lum_band is NA)
    - "rain_light"   : rain_severity=="rain_light"
    - "rain_moderate": rain_severity=="rain_moderate"
    - "rain_heavy"   : rain_severity=="rain_heavy"
    - "lum_moderate" : lum_band=="lum_moderate"
    - "lum_severe"   : lum_band=="lum_severe"
    - "compound_heavyrain_severelowlight": rain_severity=="rain_heavy" AND lum_band=="lum_severe"

    Only frames with split_role=="test" are included.
    Raises PendingMeasurement if lum_band column has any nulls (proxy for t1/t2
    not measured yet) — only for lum_moderate, lum_severe, compound bins.

    Returns path to the written YAML.
    """
    df = pd.read_parquet(manifest_path, engine="pyarrow")
    subset = _filter_severity(df, severity)

    images_dir = out_dir / "images" / severity
    labels_dir = out_dir / "labels" / severity
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Write a symlink-free file list; YOLO dataset.yaml accepts a path list.
    img_paths: list[str] = []
    for _, row in subset.iterrows():
        pool_root = pool_roots.get(str(row["pool"]))
        if pool_root is None:
            continue
        # ann_path is POSIX-relative to the pool root.
        ann_rel = row.get("ann_path")
        # Reconstruct image path from ann_path by swapping labels→images and .txt→.jpg.
        if pd.notna(ann_rel) and ann_rel:
            img_rel = str(ann_rel).replace("labels/", "images/", 1).replace(".txt", ".jpg")
            img_abs = pool_root / img_rel
        else:
            # No annotation; frame_id encodes pool and stem.
            stem = str(row["frame_id"]).split("_", 1)[-1]
            img_abs = pool_root / "images" / f"{stem}.jpg"
        if img_abs.exists():
            img_paths.append(str(img_abs))

    list_file = out_dir / f"{severity}_test.txt"
    list_file.write_text("\n".join(img_paths), encoding="utf-8")

    nc = 6  # §3.5.1 six-class vehicle taxonomy
    names = ["Car", "Motorcycle", "Tricycle", "Van", "Bus", "Truck"]

    yaml_content = {
        "path": str(out_dir),
        "train": "",         # unused for .val()
        "val": str(list_file),
        "test": str(list_file),
        "nc": nc,
        "names": names,
    }
    yaml_path = out_dir / f"{severity}.yaml"
    yaml_path.write_text(yaml.dump(yaml_content, sort_keys=False), encoding="utf-8")
    return yaml_path


def _parse_ckpt_meta(ckpt_path: Path) -> dict[str, str]:
    """Extract condition/arch/fold/stage from the checkpoint path convention.

    Expected: stage{1,2}_{condition}_{arch}_fold{N}/weights/{best,last}.pt
    Falls back to "unknown" for any group that does not match.
    """
    m = _CKPT_RE.search(str(ckpt_path))
    if m:
        return {
            "stage":     f"stage{m.group('stage')}",
            "condition": m.group("condition"),
            "arch":      m.group("arch"),
            "fold":      m.group("fold"),
        }
    return {"stage": "unknown", "condition": "unknown", "arch": "unknown", "fold": "unknown"}


def evaluate_checkpoint(
    ckpt_path: Path,
    arch: str,
    severity: str,
    manifest_path: Path,
    pool_roots: dict[str, Path],
    out_dir: Path,
    *,
    device: str = "0",
    imgsz: int = 640,
) -> dict:
    """Run .val() on ckpt_path for the given severity bin.

    Returns dict with keys:
      condition, arch, fold, stage, severity,
      mAP50_95, mAP50, precision, recall, ckpt_path (str)

    Parses condition/arch/fold/stage from the checkpoint path name pattern:
      stage{1,2}_{condition}_{arch}_fold{N}/weights/{best,last}.pt
    Falls back to "unknown" if pattern doesn't match.
    """
    from ultralytics import RTDETR, YOLO  # deferred — not always installed

    yaml_path = build_eval_yaml(severity, manifest_path, pool_roots, out_dir)

    if arch == "rtdetr_l":
        model = RTDETR(str(ckpt_path))
    else:
        model = YOLO(str(ckpt_path))

    results = model.val(
        data=str(yaml_path),
        imgsz=imgsz,
        device=device,
        workers=0,
        verbose=False,
    )

    meta = _parse_ckpt_meta(ckpt_path)

    return {
        "condition": meta["condition"],
        "arch":      arch,
        "fold":      meta["fold"],
        "stage":     meta["stage"],
        "severity":  severity,
        "mAP50_95":  float(results.box.map),
        "mAP50":     float(results.box.map50),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
        "ckpt_path": str(ckpt_path),
    }


def evaluate_all(
    registry_path: Path,
    manifest_path: Path,
    pool_roots: dict[str, Path],
    out_dir: Path,
    *,
    device: str = "0",
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Loop over all checkpoints × severity bins (420 evaluations = 60 × 7).

    Reads the checkpoint registry, iterates every (condition, arch, fold, stage)
    entry, evaluates against all 7 severity bins, writes per-evaluation JSON files.
    Returns a DataFrame of all results.

    Skips lum_band-dependent bins gracefully if PendingMeasurement is raised
    (prints a warning and continues).
    """
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    registry: list[dict[str, Any]] = json.loads(
        registry_path.read_text(encoding="utf-8")
    )
    bins = _severity_bins_from_config()

    all_rows: list[dict] = []

    for entry in registry:
        ckpt = Path(entry["ckpt_path"])
        arch = str(entry.get("arch", "yolov8m"))
        meta = _parse_ckpt_meta(ckpt)
        tag = (
            f"{meta['condition']}_{arch}_fold{meta['fold']}"
            f"_{meta['stage']}"
        )

        for severity in bins:
            json_path = eval_dir / f"{tag}_{severity}.json"
            if skip_existing and json_path.exists():
                try:
                    row = json.loads(json_path.read_text(encoding="utf-8"))
                    all_rows.append(row)
                    continue
                except Exception:
                    pass  # corrupt cache; re-evaluate

            try:
                row = evaluate_checkpoint(
                    ckpt_path=ckpt,
                    arch=arch,
                    severity=severity,
                    manifest_path=manifest_path,
                    pool_roots=pool_roots,
                    out_dir=out_dir,
                    device=device,
                )
            except PendingMeasurement as exc:
                print(
                    f"[warn] Skipping {tag} × {severity}: {exc}",
                    file=sys.stderr,
                )
                continue

            json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        parquet_path = out_dir / "all_results.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"Saved {len(df)} evaluation rows → {parquet_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate detector checkpoints across all severity bins (§3.5.1)."
    )
    p.add_argument("--registry",   required=True, type=Path,
                   help="Path to checkpoint_registry.json from detect/train.py")
    p.add_argument("--manifest",   required=True, type=Path,
                   help="Path to manifest.parquet")
    p.add_argument("--pool-a",     required=True, type=Path, dest="pool_a")
    p.add_argument("--pool-b",     required=False, type=Path, dest="pool_b", default=None)
    p.add_argument("--pool-c",     required=False, type=Path, dest="pool_c", default=None)
    p.add_argument("--out",        required=True, type=Path,
                   help="Output directory for YAML files and result JSON/parquet")
    p.add_argument("--device",     default="0")
    p.add_argument("--skip-existing", action="store_true", dest="skip_existing")
    return p


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    args = _build_parser().parse_args()

    pool_roots: dict[str, Path] = {"A": args.pool_a}
    if args.pool_b:
        pool_roots["B"] = args.pool_b
    if args.pool_c:
        pool_roots["C"] = args.pool_c

    df = evaluate_all(
        registry_path=args.registry,
        manifest_path=args.manifest,
        pool_roots=pool_roots,
        out_dir=args.out,
        device=args.device,
        skip_existing=args.skip_existing,
    )
    print(df.to_string(index=False) if not df.empty else "No evaluations completed.")
