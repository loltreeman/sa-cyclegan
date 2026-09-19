"""Tests for src/data/manifest.py — Step 10.

Covers: schema completeness, lum_band derivation (PendingMeasurement guard),
Pool A builder, fold assignment, oversampling weights, validator, and
save/load round-trip.

All values are chosen to exercise boundary conditions from the manuscript:
  - lum_band thresholds t1 < t2 divide three bands
  - fold grouping by event_id, not clip_id
  - Pool A invariants (rain_severity=clear, rain_source=null)
  - sample_weight cap at 3.0 (§3.2.6)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.util.config import PendingMeasurement
from src.data.manifest import (
    COLUMNS, POOLS, RAIN_SEVERITIES, LUM_BANDS, SPLIT_ROLES,
    derive_lum_band, apply_is_ir, apply_oversampling,
    assign_folds, build_pool_a,
    validate, save, load,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Thresholds:
    """Minimal Thresholds stub for tests; raises PendingMeasurement for None."""
    def __init__(self, **kw):
        self._d = kw

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._d:
            raise AttributeError(name)
        v = self._d[name]
        if v is None:
            raise PendingMeasurement(f"'{name}' is pending")
        return v

    def is_pending(self, name):
        return self._d.get(name) is None


def _make_image(path: Path, rgb: tuple[int, int, int], size: int = 32) -> None:
    """Write a solid-colour PNG to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((size, size, 3), rgb, dtype=np.uint8)
    ).save(path)


def _make_label(path: Path, rows: list[str]) -> None:
    """Write a YOLO-format label file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def _minimal_df(n: int = 3) -> pd.DataFrame:
    """Return a minimal valid Pool A manifest for validator tests."""
    return pd.DataFrame({
        "frame_id":      pd.array([f"A_frame{i:03d}" for i in range(n)], dtype="string"),
        "clip_id":       pd.array(["clip0"] * n, dtype="string"),
        "event_id":      pd.array(["event0"] * n, dtype="string"),
        "camera_id":     pd.array(["gate1"] * n, dtype="string"),
        "timestamp":     [None] * n,
        "pool":          pd.array(["A"] * n, dtype="string"),
        "rain_severity": pd.array(["clear"] * n, dtype="string"),
        "rain_source":   pd.array([pd.NA] * n, dtype="string"),
        "mu_y":          pd.array([120.0] * n, dtype="Float64"),
        "is_ir":         pd.array([pd.NA] * n, dtype="boolean"),
        "rho_sat":       pd.array([0.01] * n, dtype="Float64"),
        "colour_div":    pd.array([80.0] * n, dtype="Float64"),
        "lum_band":      pd.array([pd.NA] * n, dtype="string"),
        "split_role":    pd.array(["train"] * n, dtype="string"),
        "fold":          pd.array([0] * n, dtype="Int8"),
        "ann_path":      pd.array([pd.NA] * n, dtype="string"),
        "n_instances":   pd.array([2] * n, dtype="Int32"),
        "class_counts":  pd.array(['{"0": 2}'] * n, dtype="string"),
        "sample_weight": pd.array([1.0] * n, dtype="Float64"),
    })


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_all_columns_defined():
    expected = {
        "frame_id", "clip_id", "event_id", "camera_id", "timestamp",
        "pool", "rain_severity", "rain_source",
        "mu_y", "is_ir", "rho_sat", "colour_div", "lum_band",
        "split_role", "fold", "ann_path", "n_instances",
        "class_counts", "sample_weight",
    }
    assert set(COLUMNS.keys()) == expected


def test_constant_sets_non_empty():
    assert POOLS == {"A", "B", "C"}
    assert "clear" in RAIN_SEVERITIES
    assert "lum_severe" in LUM_BANDS and "normal" in LUM_BANDS
    assert SPLIT_ROLES == {"train", "test"}


# ---------------------------------------------------------------------------
# lum_band derivation
# ---------------------------------------------------------------------------

def test_derive_lum_band_raises_pending_t1():
    thr = _Thresholds(t1=None, t2=100.0)
    with pytest.raises(PendingMeasurement):
        derive_lum_band(pd.array([50.0], dtype="Float64"), thr)


def test_derive_lum_band_raises_pending_t2():
    thr = _Thresholds(t1=50.0, t2=None)
    with pytest.raises(PendingMeasurement):
        derive_lum_band(pd.array([50.0], dtype="Float64"), thr)


def test_derive_lum_band_raises_both_pending():
    thr = _Thresholds(t1=None, t2=None)
    with pytest.raises(PendingMeasurement):
        derive_lum_band(pd.array([50.0], dtype="Float64"), thr)


def test_derive_lum_band_rejects_invalid_thresholds():
    thr = _Thresholds(t1=100.0, t2=50.0)  # t1 >= t2
    with pytest.raises(ValueError, match="t1 < t2"):
        derive_lum_band(pd.array([75.0], dtype="Float64"), thr)


def test_derive_lum_band_correct_bands():
    # t1=50, t2=100 → three bands:
    #   µY < 50   → lum_severe
    #   50 ≤ µY < 100 → lum_moderate
    #   µY ≥ 100  → normal
    thr = _Thresholds(t1=50.0, t2=100.0)
    mu = pd.array([30.0, 49.9, 50.0, 75.0, 99.9, 100.0, 200.0], dtype="Float64")
    result = derive_lum_band(mu, thr)
    assert list(result) == [
        "lum_severe", "lum_severe",
        "lum_moderate", "lum_moderate", "lum_moderate",
        "normal", "normal",
    ]


def test_derive_lum_band_boundary_exactly_t1():
    thr = _Thresholds(t1=50.0, t2=100.0)
    result = derive_lum_band(pd.array([50.0], dtype="Float64"), thr)
    assert result[0] == "lum_moderate"


def test_derive_lum_band_boundary_exactly_t2():
    thr = _Thresholds(t1=50.0, t2=100.0)
    result = derive_lum_band(pd.array([100.0], dtype="Float64"), thr)
    assert result[0] == "normal"


# ---------------------------------------------------------------------------
# apply_is_ir
# ---------------------------------------------------------------------------

def test_apply_is_ir_raises_pending():
    thr = _Thresholds(tau_ir=None)
    df = _minimal_df(1)
    with pytest.raises(PendingMeasurement):
        apply_is_ir(df, thr)


def test_apply_is_ir_raises_missing_column():
    thr = _Thresholds(tau_ir=30.0)
    df = _minimal_df(1).drop(columns=["colour_div"])
    with pytest.raises(RuntimeError, match="colour_div"):
        apply_is_ir(df, thr)


def test_apply_is_ir_correct_values():
    # tau_ir = 30: colour_div < 30 → is_ir=True, else False
    thr = _Thresholds(tau_ir=30.0)
    df = _minimal_df(3)
    df = df.copy()
    df["colour_div"] = pd.array([10.0, 30.0, 80.0], dtype="Float64")
    result = apply_is_ir(df, thr)
    assert list(result["is_ir"]) == [True, False, False]


# ---------------------------------------------------------------------------
# Pool A builder (smoke test with synthetic images)
# ---------------------------------------------------------------------------

@pytest.fixture()
def pool_a_root(tmp_path: Path) -> Path:
    """Create a minimal Pool A directory: 4 images, 3 label files."""
    # Three images with distinct luminance values, one without a label file
    images = [
        ("clip01_020", (20, 20, 20)),    # dark  → lum_severe  with t1=50, t2=100
        ("clip01_080", (80, 80, 80)),    # mid   → lum_moderate
        ("clip02_150", (150, 150, 150)), # bright → normal
        ("clip02_255", (255, 255, 255)), # saturated, no label file
    ]
    for stem, rgb in images:
        _make_image(tmp_path / "images" / f"{stem}.jpg", rgb)

    labels = [
        ("clip01_020", ["0 0.5 0.5 0.3 0.3", "2 0.2 0.2 0.1 0.1"]),  # classes 0, 2
        ("clip01_080", ["1 0.5 0.5 0.4 0.4"]),                          # class 1
        ("clip02_150", []),                                              # background
    ]
    for stem, rows in labels:
        _make_label(tmp_path / "labels" / f"{stem}.txt", rows)

    return tmp_path


def test_build_pool_a_returns_all_columns(pool_a_root):
    df = build_pool_a(pool_a_root, camera_id="gate1")
    for col in COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_build_pool_a_row_count(pool_a_root):
    df = build_pool_a(pool_a_root, camera_id="gate1")
    assert len(df) == 4


def test_build_pool_a_pool_a(pool_a_root):
    df = build_pool_a(pool_a_root)
    assert (df["pool"] == "A").all()


def test_build_pool_a_rain_severity_clear(pool_a_root):
    df = build_pool_a(pool_a_root)
    assert (df["rain_severity"] == "clear").all()


def test_build_pool_a_rain_source_null(pool_a_root):
    df = build_pool_a(pool_a_root)
    assert df["rain_source"].isna().all()


def test_build_pool_a_split_role_train(pool_a_root):
    df = build_pool_a(pool_a_root)
    assert (df["split_role"] == "train").all()


def test_build_pool_a_frame_ids_unique(pool_a_root):
    df = build_pool_a(pool_a_root)
    assert df["frame_id"].nunique() == len(df)


def test_build_pool_a_frame_ids_prefixed(pool_a_root):
    df = build_pool_a(pool_a_root)
    assert df["frame_id"].str.startswith("A_").all()


def test_build_pool_a_camera_id(pool_a_root):
    df = build_pool_a(pool_a_root, camera_id="gate3")
    assert (df["camera_id"] == "gate3").all()


def test_build_pool_a_n_instances(pool_a_root):
    df = build_pool_a(pool_a_root).sort_values("frame_id").reset_index(drop=True)
    # clip01_020 → 2 instances, clip01_080 → 1, clip02_150 → 0, clip02_250 → null
    counts = df.set_index("frame_id")["n_instances"]
    assert counts["A_clip01_020"] == 2
    assert counts["A_clip01_080"] == 1
    assert counts["A_clip02_150"] == 0
    assert pd.isna(counts["A_clip02_255"])


def test_build_pool_a_class_counts(pool_a_root):
    df = build_pool_a(pool_a_root)
    row = df[df["frame_id"] == "A_clip01_020"].iloc[0]
    counts = json.loads(row["class_counts"])
    assert counts == {"0": 1, "2": 1}


def test_build_pool_a_mu_y_range(pool_a_root):
    df = build_pool_a(pool_a_root)
    assert (df["mu_y"].dropna() >= 0).all()
    assert (df["mu_y"].dropna() <= 255).all()


def test_build_pool_a_rho_sat_range(pool_a_root):
    df = build_pool_a(pool_a_root)
    assert (df["rho_sat"].dropna() >= 0).all()
    assert (df["rho_sat"].dropna() <= 1).all()


def test_build_pool_a_saturated_frame(pool_a_root):
    # The 250,250,250 image has rho_sat > 0
    df = build_pool_a(pool_a_root)
    row = df[df["frame_id"] == "A_clip02_255"].iloc[0]
    assert row["rho_sat"] > 0


def test_build_pool_a_lum_band_null_by_default(pool_a_root):
    df = build_pool_a(pool_a_root)
    assert df["lum_band"].isna().all()


def test_build_pool_a_lum_band_computed_when_requested(pool_a_root):
    thr = _Thresholds(t1=50.0, t2=100.0, k_folds=5,
                      tau_ir=None, tau_glare=None, n_aug=None,
                      ir_disposition=None, alive_extraction_interval_s=None)
    df = build_pool_a(pool_a_root, thresholds=thr, compute_lum_band=True)
    assert df["lum_band"].notna().all()
    # The dark image (20,20,20) should be lum_severe
    row = df[df["frame_id"] == "A_clip01_020"].iloc[0]
    assert row["lum_band"] == "lum_severe"


def test_build_pool_a_custom_clip_id_fn(pool_a_root):
    df = build_pool_a(pool_a_root, clip_id_fn=lambda p: "fixed_clip")
    assert (df["clip_id"] == "fixed_clip").all()


def test_build_pool_a_custom_event_id_fn(pool_a_root):
    df = build_pool_a(pool_a_root,
                      clip_id_fn=lambda p: p.stem.split("_")[0],
                      event_id_fn=lambda cid: "shared_event")
    assert (df["event_id"] == "shared_event").all()


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------

@pytest.fixture()
def multi_event_df() -> pd.DataFrame:
    """DataFrame with 3 events × 4 frames each = 12 frames total."""
    rows = []
    for event in ["ev_A", "ev_B", "ev_C", "ev_D", "ev_E"]:
        for i in range(4):
            rows.append({
                "frame_id": f"A_{event}_{i}",
                "event_id": event,
                "clip_id":  event,
            })
    df = pd.DataFrame(rows)
    for col in ["pool", "rain_severity", "split_role", "camera_id"]:
        df[col] = "A" if col == "pool" else ("clear" if col == "rain_severity"
                  else "train" if col == "split_role" else "gate1")
    return df


def test_assign_folds_all_assigned(multi_event_df):
    df = assign_folds(multi_event_df, k=5)
    assert df["fold"].notna().all()


def test_assign_folds_values_in_range(multi_event_df):
    k = 5
    df = assign_folds(multi_event_df, k=k)
    assert df["fold"].min() >= 0
    assert df["fold"].max() <= k - 1


def test_assign_folds_event_grouping(multi_event_df):
    """All frames with the same event_id must share the same fold."""
    df = assign_folds(multi_event_df, k=5)
    for event, grp in df.groupby("event_id"):
        assert grp["fold"].nunique() == 1, f"event {event} spans multiple folds"


def test_assign_folds_each_event_in_exactly_one_fold(multi_event_df):
    df = assign_folds(multi_event_df, k=5)
    for fold in range(5):
        events_in_fold = df[df["fold"] == fold]["event_id"].unique()
        for event in events_in_fold:
            assert (df[df["event_id"] == event]["fold"] == fold).all()


def test_assign_folds_covers_all_k_folds(multi_event_df):
    k = 5
    df = assign_folds(multi_event_df, k=k)
    assert set(df["fold"].dropna().unique()) == set(range(k))


def test_assign_folds_too_few_events():
    df = pd.DataFrame({"frame_id": ["A_f0"], "event_id": ["ev_A"], "clip_id": ["c0"]})
    with pytest.raises(ValueError, match="folds"):
        assign_folds(df, k=5)


def test_assign_folds_deterministic(multi_event_df):
    df1 = assign_folds(multi_event_df, k=5, seed=42)
    df2 = assign_folds(multi_event_df, k=5, seed=42)
    assert (df1["fold"].values == df2["fold"].values).all()


def test_assign_folds_different_seeds_differ(multi_event_df):
    df1 = assign_folds(multi_event_df, k=5, seed=0)
    df2 = assign_folds(multi_event_df, k=5, seed=99)
    # With 5 events and 2 seeds, at least one event should land in a different fold
    assert not (df1["fold"].values == df2["fold"].values).all()


# ---------------------------------------------------------------------------
# Oversampling
# ---------------------------------------------------------------------------

def test_apply_oversampling_default_weight_one():
    df = _minimal_df(4)
    df["class_counts"] = pd.array(['{"0": 3}'] * 4, dtype="string")
    result = apply_oversampling(df, minority_class_ids=[5])  # class 5 absent
    assert (result["sample_weight"] == 1.0).all()


def test_apply_oversampling_weight_applied():
    df = _minimal_df(4)
    # 2 majority (class 0 only), 2 minority (class 2 present)
    df["class_counts"] = pd.array(
        ['{"0": 3}', '{"0": 2}', '{"2": 1}', '{"0": 1, "2": 2}'],
        dtype="string",
    )
    result = apply_oversampling(df, minority_class_ids=[2])
    # n_majority=2, n_minority=2 → weight = min(3.0, 2/2) = 1.0
    majority_w = result.loc[result["class_counts"].str.contains('"2"') == False, "sample_weight"]
    minority_w = result.loc[result["class_counts"].str.contains('"2"') == True, "sample_weight"]
    assert (majority_w == 1.0).all()
    # weight = min(3.0, 1.0) = 1.0 in this balanced case
    assert (minority_w >= 1.0).all()


def test_apply_oversampling_cap_respected():
    df = _minimal_df(10)
    # 9 majority, 1 minority → weight = min(3.0, 9/1) = 3.0
    counts = ['{"0": 1}'] * 9 + ['{"5": 1}']
    df["class_counts"] = pd.array(counts, dtype="string")
    result = apply_oversampling(df, minority_class_ids=[5], cap=3.0)
    minority_row = result[result["class_counts"] == '{"5": 1}']
    assert float(minority_row["sample_weight"].iloc[0]) <= 3.0


def test_apply_oversampling_empty_minority_list():
    df = _minimal_df(3)
    result = apply_oversampling(df, minority_class_ids=[])
    assert (result["sample_weight"] == 1.0).all()


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_validate_valid_df_no_violations():
    df = _minimal_df(3)
    assert validate(df) == []


def test_validate_missing_column():
    df = _minimal_df(2).drop(columns=["mu_y"])
    errs = validate(df)
    assert any("mu_y" in e for e in errs)


def test_validate_duplicate_frame_id():
    df = _minimal_df(3)
    df = df.copy()
    df.iloc[0, df.columns.get_loc("frame_id")] = df.iloc[1]["frame_id"]
    errs = validate(df)
    assert any("duplicate" in e.lower() for e in errs)


def test_validate_unknown_pool():
    df = _minimal_df(2)
    df = df.copy()
    df.iloc[0, df.columns.get_loc("pool")] = "X"
    errs = validate(df)
    assert any("pool" in e.lower() for e in errs)


def test_validate_unknown_rain_severity():
    df = _minimal_df(2)
    df = df.copy()
    df.iloc[0, df.columns.get_loc("rain_severity")] = "monsoon"
    errs = validate(df)
    assert any("rain_severity" in e for e in errs)


def test_validate_pool_a_wrong_rain_severity():
    df = _minimal_df(2)
    df = df.copy()
    df.iloc[0, df.columns.get_loc("rain_severity")] = "rain_heavy"
    errs = validate(df)
    assert any("Pool A rain_severity" in e for e in errs)


def test_validate_pool_a_wrong_rain_source():
    df = _minimal_df(2)
    df = df.copy()
    df.iloc[0, df.columns.get_loc("rain_source")] = "measured"
    errs = validate(df)
    assert any("Pool A rain_source" in e for e in errs)


def test_validate_mu_y_out_of_range():
    df = _minimal_df(2)
    df = df.copy()
    df.iloc[0, df.columns.get_loc("mu_y")] = 300.0  # > 255
    errs = validate(df)
    assert any("mu_y" in e for e in errs)


def test_validate_sample_weight_below_one():
    df = _minimal_df(2)
    df = df.copy()
    df.iloc[0, df.columns.get_loc("sample_weight")] = 0.5
    errs = validate(df)
    assert any("sample_weight" in e for e in errs)


def test_validate_unknown_lum_band():
    df = _minimal_df(2)
    df = df.copy()
    df.iloc[0, df.columns.get_loc("lum_band")] = "dark"
    errs = validate(df)
    assert any("lum_band" in e for e in errs)


def test_validate_valid_lum_bands():
    df = _minimal_df(3)
    df = df.copy()
    df["lum_band"] = pd.array(["normal", "lum_moderate", "lum_severe"], dtype="string")
    assert validate(df) == []


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path, pool_a_root):
    df = build_pool_a(pool_a_root, camera_id="gate1")
    df = assign_folds(df, k=2)  # k=2 since we have ≥2 events by default clip_id
    path = tmp_path / "manifest.parquet"
    save(df, path)
    df2 = load(path)
    assert list(df.columns) == list(df2.columns)
    assert len(df) == len(df2)
    # Key columns survive the round-trip
    assert df["frame_id"].tolist() == df2["frame_id"].tolist()
    assert df["mu_y"].tolist() == df2["mu_y"].tolist()


def test_save_creates_parent_dir(tmp_path):
    df = _minimal_df(2)
    path = tmp_path / "subdir" / "nested" / "manifest.parquet"
    save(df, path)
    assert path.exists()


def test_load_applies_dtypes(tmp_path, pool_a_root):
    df = build_pool_a(pool_a_root)
    path = tmp_path / "manifest.parquet"
    save(df, path)
    df2 = load(path)
    assert str(df2["pool"].dtype) == "string"
    assert str(df2["mu_y"].dtype) == "Float64"
    assert str(df2["n_instances"].dtype) == "Int32"
