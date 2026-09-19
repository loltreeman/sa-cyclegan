"""Manifest schema, builder, and validator — Step 10.

One row per frame.  Every downstream component (data loading, fold splits,
severity stratification, oversampling, evaluation) is a query on this table.
Footage arrival is an ingest problem, not a refactor: add rows, call
derive_lum_band() when t1/t2 are measured, call apply_is_ir() when tau_ir
is measured.  Nothing else changes.

Schema decisions (locked — change here means change the manuscript too)
-----------------------------------------------------------------------
fold         — held-out fold index 0 … k-1.  Train on fold != query_fold.
               One row per frame; the training harness, not the table, knows
               which folds are train vs val.
lum_band     — "lum_severe" | "lum_moderate" | "normal"; null until t1/t2
               measured.  Names match base.yaml severity_bins vocabulary.
               Derivation raises PendingMeasurement while thresholds are null.
rain_severity — "clear" | "rain_light" | "rain_moderate" | "rain_heavy".
               Orthogonal to lum_band: a compound frame has rain_severity=
               "rain_heavy" AND lum_band="lum_severe".
rain_source  — null (Pool A, no rain) | "synthetic:<gen_id>" (Pool B,
               identifies which generator made the frame) | "measured" (Pool C,
               PAGASA/WMO mm/hr adjudicated per clip).
               D1 (CLAUDE.md): rain_source keeps constructed severity apart
               from measured severity.
ann_path     — POSIX path relative to the pool root passed to the builder;
               null when no label file exists.  Reconstruct the absolute path
               as pool_root / ann_path in the training harness.  Pool B
               synthetic frames may inherit the source Pool A annotation path
               (geometry is preserved by image translation).
sample_weight — float ≥ 1.0, ≤ cap.  Default 1.0; updated by
               apply_oversampling().  DataLoader uses WeightedRandomSampler.

Extra columns beyond CLAUDE.md's listed 16
-------------------------------------------
colour_div   — mean per-pixel channel range (Float64).  Cached so apply_is_ir()
               can populate is_ir from tau_ir without re-reading images.
class_counts — JSON string "{class_id: count, …}" (string).  Cached so
               apply_oversampling() can weight frames without re-reading labels.

Usage
-----
    from src.data.manifest import build_pool_a, assign_folds, validate, save, load

    df = build_pool_a("data/pool_a", camera_id="gate1")
    df = assign_folds(df)
    violations = validate(df)
    save(df, "runs/manifest.parquet")
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

from src.util.config import PendingMeasurement, load_thresholds

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

RAIN_SEVERITIES = frozenset({"clear", "rain_light", "rain_moderate", "rain_heavy"})
LUM_BANDS       = frozenset({"lum_severe", "lum_moderate", "normal"})
POOLS           = frozenset({"A", "B", "C"})
SPLIT_ROLES     = frozenset({"train", "test"})

# Canonical column → pandas dtype.  Nullable variants (capital letter) allow
# pd.NA in integer/boolean/string columns without converting to float.
COLUMNS: dict[str, str] = {
    "frame_id":      "string",
    "clip_id":       "string",
    "event_id":      "string",
    "camera_id":     "string",
    "timestamp":     "object",      # datetime | None; object survives parquet
    "pool":          "string",
    "rain_severity": "string",
    "rain_source":   "string",      # nullable
    "mu_y":          "Float64",     # nullable; always computed during build
    "is_ir":         "boolean",     # nullable until tau_ir measured
    "rho_sat":       "Float64",     # nullable
    "colour_div":    "Float64",     # nullable; used by apply_is_ir()
    "lum_band":      "string",      # nullable until t1/t2 measured
    "split_role":    "string",
    "fold":          "Int8",        # nullable until assign_folds() called
    "ann_path":      "string",      # nullable
    "n_instances":   "Int32",       # nullable
    "class_counts":  "string",      # nullable; JSON {"cls": count}
    "sample_weight": "Float64",
}


# ---------------------------------------------------------------------------
# Per-frame pixel measurements
# ---------------------------------------------------------------------------

def _open_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _mu_y(img: np.ndarray) -> float:
    """Mean luma (Y) in BT.601 YCbCr space, 0–255."""
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    return float((0.299 * r + 0.587 * g + 0.114 * b).mean())


def _rho_sat(img: np.ndarray, threshold: int = 250) -> float:
    """Fraction of pixels where any channel exceeds threshold."""
    return float(np.any(img > threshold, axis=2).mean())


def _colour_div(img: np.ndarray) -> float:
    """Mean per-pixel range across colour channels.

    Low value → channels are similar (grayscale-like) → candidate IR frame.
    Compared against tau_ir by apply_is_ir().
    """
    hi = img.max(axis=2).astype(np.int32)
    lo = img.min(axis=2).astype(np.int32)
    return float((hi - lo).mean())


def _extract_timestamp(path: Path) -> datetime | None:
    """Read DateTimeOriginal from EXIF; return None if absent or unreadable."""
    try:
        from PIL.ExifTags import TAGS
        exif = Image.open(path)._getexif()  # type: ignore[attr-defined]
        if exif:
            for tag_id, val in exif.items():
                if TAGS.get(tag_id) == "DateTimeOriginal":
                    return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def _read_yolo_label(path: Path) -> tuple[int, dict[int, int]]:
    """Return (n_instances, {class_id: count}) from a YOLO .txt label file.

    Empty file → (0, {}) — background frame, explicitly annotated.
    Missing file → (None, None) — annotation status unknown.
    """
    if not path.exists():
        return None, None  # type: ignore[return-value]
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    counts: dict[int, int] = {}
    for line in lines:
        cls = int(line.split()[0])
        counts[cls] = counts.get(cls, 0) + 1
    return len(lines), counts


# ---------------------------------------------------------------------------
# Schema casting
# ---------------------------------------------------------------------------

def _cast(df: pd.DataFrame) -> pd.DataFrame:
    """Apply COLUMNS dtypes; add missing columns as all-NA; preserve order."""
    df = df.copy()
    for col, dtype in COLUMNS.items():
        if col not in df.columns:
            df[col] = pd.array([pd.NA] * len(df), dtype=dtype if dtype != "object" else object)
        else:
            if dtype != "object":
                try:
                    df[col] = df[col].astype(dtype)
                except (TypeError, ValueError):
                    pass
    return df[[c for c in COLUMNS if c in df.columns]]


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def derive_lum_band(mu_y: "pd.Series[float]", thresholds=None) -> "pd.Series[str]":
    """Derive lum_band from µY values and t1/t2 thresholds.

    Raises PendingMeasurement if t1 or t2 are null in thresholds.yaml.

    Bands (t1 < t2, both in 0–255 YCbCr luma):
        "lum_severe"   µY < t1   (darkest)
        "lum_moderate" t1 ≤ µY < t2
        "normal"       µY ≥ t2

    These names match base.yaml severity_bins so evaluation join keys are
    consistent across the table and the config.
    """
    if thresholds is None:
        thresholds = load_thresholds()
    t1: float = thresholds.t1   # raises PendingMeasurement if null
    t2: float = thresholds.t2   # raises PendingMeasurement if null
    if t1 >= t2:
        raise ValueError(f"Thresholds require t1 < t2; got t1={t1}, t2={t2}")
    bands = np.where(
        mu_y < t1, "lum_severe",
        np.where(mu_y < t2, "lum_moderate", "normal"),
    )
    return pd.array(bands, dtype="string")


def apply_is_ir(df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """Populate is_ir from the cached colour_div column and tau_ir threshold.

    Raises PendingMeasurement if tau_ir is null.
    Raises RuntimeError if colour_div column is absent (rebuild from source).
    """
    if thresholds is None:
        thresholds = load_thresholds()
    tau: float = thresholds.tau_ir  # raises PendingMeasurement if null
    if "colour_div" not in df.columns:
        raise RuntimeError(
            "colour_div column is absent. Re-run build_pool_a() to recompute it, "
            "or supply it from the source images."
        )
    df = df.copy()
    df["is_ir"] = pd.array(
        (df["colour_div"].to_numpy(dtype=float, na_value=np.nan) < tau),
        dtype="boolean",
    )
    return df


# ---------------------------------------------------------------------------
# Pool A builder
# ---------------------------------------------------------------------------

_SUFFIX_RE = re.compile(r"[_-]\d+$")


def _default_clip_id(img_path: Path) -> str:
    """Strip a trailing numeric suffix from the stem to get a clip id.

    "gate1_20231205_042" → "gate1_20231205"
    "clip001_frame0037"  → "clip001_frame"   (stops at first numeric suffix)
    Falls back to full stem if no suffix found.
    """
    return _SUFFIX_RE.sub("", img_path.stem) or img_path.stem


def build_pool_a(
    root: "Path | str",
    *,
    camera_id: str = "unknown",
    clip_id_fn: "Callable[[Path], str] | None" = None,
    event_id_fn: "Callable[[str], str] | None" = None,
    sat_threshold: int = 250,
    thresholds=None,
    compute_lum_band: bool = False,
    compute_is_ir: bool = False,
) -> pd.DataFrame:
    """Build manifest rows for Pool A (annotated clear-weather ALIVE footage).

    Pool A properties enforced here:
      pool = "A", rain_severity = "clear", rain_source = null,
      split_role = "train"

    Parameters
    ----------
    root
        Directory with images/ and labels/ subdirs in YOLO layout.
        Flat structure expected (all files directly inside each subdir).
    camera_id
        Gate / viewpoint identifier applied to every frame in this root.
    clip_id_fn
        (img_path) → clip_id string.  Default strips trailing numeric suffix
        from the stem.  Provide this when your filename convention is known.
    event_id_fn
        (clip_id) → event_id string.  Default: identity (each clip is its own
        event, conservative but safe).  Provide a mapping when you know which
        clips belong to the same filming session — fold grouping depends on it.
    sat_threshold
        Per-channel pixel value above which a pixel counts as saturated.
    thresholds
        Loaded Thresholds object.  Loaded from thresholds.yaml if not provided.
    compute_lum_band
        If True, derive lum_band immediately.  Raises PendingMeasurement while
        t1/t2 are null.  Default False — leave null and call derive_lum_band()
        once thresholds are measured.
    compute_is_ir
        If True, populate is_ir from tau_ir.  Raises PendingMeasurement while
        tau_ir is null.  Default False.

    Returns
    -------
    pd.DataFrame with all COLUMNS in canonical order.  lum_band and is_ir
    are null unless the corresponding compute_* flags are True.
    """
    root = Path(root)
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"images/ not found under {root}")

    if thresholds is None:
        thresholds = load_thresholds()

    _clip_id = clip_id_fn if clip_id_fn is not None else _default_clip_id
    _event_id = event_id_fn if event_id_fn is not None else (lambda cid: cid)

    img_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    rows = []
    for img_path in img_paths:
        img = _open_rgb(img_path)

        cid = _clip_id(img_path)
        eid = _event_id(cid)
        fid = f"A_{img_path.stem}"

        label_path = labels_dir / (img_path.stem + ".txt")
        n_inst, counts = _read_yolo_label(label_path)
        ann = (
            str(PurePosixPath(label_path.relative_to(root)))
            if label_path.exists() else None
        )

        rows.append({
            "frame_id":      fid,
            "clip_id":       cid,
            "event_id":      eid,
            "camera_id":     camera_id,
            "timestamp":     _extract_timestamp(img_path),
            "pool":          "A",
            "rain_severity": "clear",
            "rain_source":   pd.NA,
            "mu_y":          _mu_y(img),
            "is_ir":         pd.NA,
            "rho_sat":       _rho_sat(img, sat_threshold),
            "colour_div":    _colour_div(img),
            "lum_band":      pd.NA,
            "split_role":    "train",
            "fold":          pd.NA,
            "ann_path":      ann,
            "n_instances":   n_inst,
            "class_counts":  json.dumps(counts) if counts is not None else pd.NA,
            "sample_weight": 1.0,
        })

    df = _cast(pd.DataFrame(rows))

    if compute_lum_band:
        df = df.copy()
        df["lum_band"] = derive_lum_band(df["mu_y"], thresholds)

    if compute_is_ir:
        df = apply_is_ir(df, thresholds)

    return df


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------

def assign_folds(
    df: pd.DataFrame,
    k: int | None = None,
    *,
    group_col: str = "event_id",
    seed: int = 0,
    thresholds=None,
) -> pd.DataFrame:
    """Assign each frame a held-out fold index, grouped by event_id.

    All frames sharing an event_id go into the same fold, so no single
    weather event spans train and validation.  This is the GroupKFold
    strategy mandated by §3.2.5.

    Parameters
    ----------
    df
        Manifest DataFrame; must have the group_col column.
    k
        Fold count.  If None, read from thresholds.yaml (k_folds = 5).
    group_col
        Column whose unique values define groups (default: event_id).
    seed
        Random seed for shuffling group order before round-robin assignment.

    Returns
    -------
    Copy of df with fold column populated.

    Raises
    ------
    ValueError
        If the number of distinct groups is less than k.
    """
    if thresholds is None:
        thresholds = load_thresholds()
    if k is None:
        k = int(thresholds.k_folds)  # k_folds = 5, not null

    groups = sorted(df[group_col].dropna().unique().tolist())
    if len(groups) < k:
        raise ValueError(
            f"Cannot create {k} folds from {len(groups)} distinct {group_col} values. "
            f"Either reduce k or supply more events."
        )

    rng = random.Random(seed)
    rng.shuffle(groups)
    fold_map = {g: i % k for i, g in enumerate(groups)}

    df = df.copy()
    df["fold"] = pd.array(
        [fold_map.get(g, pd.NA) for g in df[group_col]],
        dtype="Int8",
    )
    return df


# ---------------------------------------------------------------------------
# Oversampling
# ---------------------------------------------------------------------------

def apply_oversampling(
    df: pd.DataFrame,
    minority_class_ids: list[int],
    cap: float = 3.0,
) -> pd.DataFrame:
    """Set sample_weight for frames containing minority-class instances.

    §3.2.6: frame duplication capped at 3× to rebalance Truck, Bus, Tricycle.
    Implementation uses a uniform weight = min(cap, n_majority / n_minority)
    applied to all frames that contain at least one minority-class instance.
    Majority-class-only frames keep weight = 1.0.

    Parameters
    ----------
    minority_class_ids
        Class integer IDs (from the 6-class taxonomy) to oversample.
    cap
        Maximum sample weight (default 3.0 per §3.2.6).

    Returns
    -------
    Copy of df with sample_weight updated.
    """
    if "class_counts" not in df.columns:
        raise ValueError("class_counts column is required for oversampling")
    if not minority_class_ids:
        return df

    minority_set = set(minority_class_ids)

    def _has_minority(counts_json) -> bool:
        if pd.isna(counts_json):
            return False
        return any(int(k) in minority_set for k in json.loads(counts_json))

    has_min = df["class_counts"].apply(_has_minority)
    n_min = int(has_min.sum())
    n_maj = len(df) - n_min

    if n_min == 0:
        return df  # nothing to oversample

    weight = min(cap, max(1.0, n_maj / n_min))

    df = df.copy()
    df["sample_weight"] = pd.array(
        np.where(has_min, weight, 1.0),
        dtype="Float64",
    )
    return df


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> list[str]:
    """Check manifest invariants.  Returns a list of violation strings.

    An empty list means the manifest is valid.  Call this before save() and
    before handing the table to the training harness.

    Checks performed
    ----------------
    - All COLUMNS present
    - frame_id unique
    - Categorical values within allowed sets
    - Pool A invariants: rain_severity=clear, rain_source=null
    - mu_y in [0, 255], rho_sat in [0, 1], colour_div ≥ 0
    - sample_weight ≥ 1.0
    - fold values non-negative
    - lum_band consistency with mu_y (only if t1/t2 not pending)
    """
    errs: list[str] = []

    # Column presence
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        errs.append(f"Missing columns: {missing}")
        return errs  # further checks assume columns exist

    # frame_id uniqueness
    n_dupes = int(df["frame_id"].duplicated().sum())
    if n_dupes:
        errs.append(f"frame_id: {n_dupes} duplicate values")

    # Categorical values
    _bad(errs, df, "pool",          POOLS,            "Unknown pool values")
    _bad(errs, df, "rain_severity", RAIN_SEVERITIES,  "Unknown rain_severity values")
    _bad(errs, df, "split_role",    SPLIT_ROLES,       "Unknown split_role values")

    lum_notna = df["lum_band"].notna()
    if lum_notna.any():
        bad = lum_notna & ~df["lum_band"].isin(LUM_BANDS)
        if bad.any():
            errs.append(f"Unknown lum_band values: {df.loc[bad, 'lum_band'].unique().tolist()}")

    # Pool A invariants
    pool_a = df["pool"] == "A"
    if pool_a.any():
        wrong_sev = pool_a & (df["rain_severity"] != "clear")
        if wrong_sev.any():
            errs.append(f"Pool A rain_severity must be 'clear'; {wrong_sev.sum()} rows violate this")
        wrong_src = pool_a & df["rain_source"].notna()
        if wrong_src.any():
            errs.append(f"Pool A rain_source must be null; {wrong_src.sum()} rows violate this")

    # Numeric range checks
    _range(errs, df, "mu_y",       0.0, 255.0, "mu_y out of [0, 255]")
    _range(errs, df, "rho_sat",    0.0,   1.0, "rho_sat out of [0, 1]")
    _range(errs, df, "colour_div", 0.0, 255.0, "colour_div out of [0, 255]")

    sw = df["sample_weight"].dropna()
    if (sw < 1.0).any():
        errs.append(f"sample_weight < 1.0: {int((sw < 1.0).sum())} rows")

    fold_vals = df["fold"].dropna()
    if (fold_vals < 0).any():
        errs.append(f"fold: negative values found")

    # lum_band consistency (only if thresholds are available)
    try:
        thr = load_thresholds()
        t1: float = thr.t1
        t2: float = thr.t2
        notna = df["lum_band"].notna() & df["mu_y"].notna()
        if notna.any():
            mu = df.loc[notna, "mu_y"].astype(float)
            lb = df.loc[notna, "lum_band"]
            expected = pd.array(
                np.where(mu < t1, "lum_severe", np.where(mu < t2, "lum_moderate", "normal")),
                dtype="string",
            )
            mismatch = int((lb.values != expected).sum())
            if mismatch:
                errs.append(f"lum_band inconsistent with mu_y and t1/t2: {mismatch} rows")
    except PendingMeasurement:
        pass  # thresholds not yet measured; skip consistency check

    return errs


def _bad(errs: list, df: pd.DataFrame, col: str,
         allowed: frozenset, msg: str) -> None:
    vals = df[col].dropna()
    bad = vals[~vals.isin(allowed)]
    if not bad.empty:
        errs.append(f"{msg}: {bad.unique().tolist()}")


def _range(errs: list, df: pd.DataFrame, col: str,
           lo: float, hi: float, msg: str) -> None:
    vals = df[col].dropna().astype(float)
    if ((vals < lo) | (vals > hi)).any():
        errs.append(msg)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, path: "Path | str") -> None:
    """Write manifest to Parquet (pyarrow engine)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def load(path: "Path | str") -> pd.DataFrame:
    """Read manifest from Parquet and apply canonical dtypes."""
    return _cast(pd.read_parquet(Path(path), engine="pyarrow"))
