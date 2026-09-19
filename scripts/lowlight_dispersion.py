"""Low-light output dispersion check (§3.6.1 line 776).

Computes the µY (mean luminance, Y channel of YCbCr) distribution of the
low-light generator's outputs over a sample of source frames and compares it
to the µY distribution of real low-light footage.

If the synthetic distribution is insufficiently dispersed to support
illumination-stratified evaluation, the contingency in §3.3.7 applies:
the two low-light severity bins are collapsed into one.

Output: CSV + summary JSON with per-frame µY, distribution statistics, and
a dispersion flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

__all__ = ["compute_mu_y", "mu_y_stats", "dispersion_check"]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _img_paths(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMG_EXTS)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def compute_mu_y(img_bgr: np.ndarray) -> float:
    """Mean luminance of the Y channel (YCbCr), normalised to [0, 1]."""
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return float(ycbcr[:, :, 0].mean() / 255.0)


def mu_y_stats(paths: list[Path]) -> pd.DataFrame:
    """Return a DataFrame with columns: frame, mu_y."""
    rows: list[dict] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        rows.append({"frame": p.name, "mu_y": compute_mu_y(img)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dispersion check
# ---------------------------------------------------------------------------

def dispersion_check(
    synthetic_dir: Path,
    real_dir: Path,
    *,
    t1: float | None = None,
    t2: float | None = None,
) -> dict:
    """Compare µY distributions of synthetic and real low-light images.

    t1, t2 are the luminance bin thresholds from thresholds.yaml
    (lum_severe: µY < t1, lum_moderate: t1 ≤ µY < t2).
    If provided, reports the fraction of synthetic frames that fall in each bin.

    Returns a summary dict:
        synthetic_*   — stats for synthetic outputs
        real_*        — stats for real low-light footage
        dispersion_ratio — synthetic IQR / real IQR (1.0 = matches real spread)
        sufficient_dispersion — True if ratio >= 0.5 (heuristic; report regardless)
    """
    syn_df  = mu_y_stats(_img_paths(synthetic_dir))
    real_df = mu_y_stats(_img_paths(real_dir))

    def _stats(df: pd.DataFrame) -> dict:
        col = df["mu_y"]
        return {
            "n":      len(df),
            "mean":   float(col.mean()),
            "std":    float(col.std(ddof=1)),
            "min":    float(col.min()),
            "q25":    float(col.quantile(0.25)),
            "median": float(col.median()),
            "q75":    float(col.quantile(0.75)),
            "max":    float(col.max()),
        }

    syn_stats  = _stats(syn_df)
    real_stats = _stats(real_df)

    real_iqr = real_stats["q75"] - real_stats["q25"]
    syn_iqr  = syn_stats["q75"]  - syn_stats["q25"]
    ratio    = (syn_iqr / real_iqr) if real_iqr > 1e-6 else float("nan")

    result: dict = {
        "synthetic_n":              syn_stats["n"],
        "synthetic_mean_mu_y":      syn_stats["mean"],
        "synthetic_std_mu_y":       syn_stats["std"],
        "synthetic_median_mu_y":    syn_stats["median"],
        "synthetic_iqr_mu_y":       syn_iqr,
        "real_n":                   real_stats["n"],
        "real_mean_mu_y":           real_stats["mean"],
        "real_std_mu_y":            real_stats["std"],
        "real_median_mu_y":         real_stats["median"],
        "real_iqr_mu_y":            real_iqr,
        "dispersion_ratio":         ratio,
        "sufficient_dispersion":    (ratio >= 0.5) if not (ratio != ratio) else False,
    }

    if t1 is not None and t2 is not None:
        col = syn_df["mu_y"]
        result["synthetic_pct_lum_severe"]   = float((col < t1).mean() * 100)
        result["synthetic_pct_lum_moderate"] = float(((col >= t1) & (col < t2)).mean() * 100)
        result["synthetic_pct_above_t2"]     = float((col >= t2).mean() * 100)
        result["t1"] = t1
        result["t2"] = t2

    return result, syn_df, real_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Low-light output dispersion check (§3.6.1 line 776)."
    )
    p.add_argument("--synthetic", required=True, type=Path,
                   help="Directory of low-light generator outputs")
    p.add_argument("--real",      required=True, type=Path,
                   help="Directory of real low-light footage frames")
    p.add_argument("--out",       required=True, type=Path,
                   help="Output directory for CSV and summary JSON")
    p.add_argument("--t1",  default=None, type=float,
                   help="Luminance threshold t1 (lum_severe < t1)")
    p.add_argument("--t2",  default=None, type=float,
                   help="Luminance threshold t2 (lum_moderate < t2)")
    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary, syn_df, real_df = dispersion_check(
        args.synthetic, args.real,
        t1=args.t1, t2=args.t2,
    )

    syn_df["domain"]  = "synthetic"
    real_df["domain"] = "real"
    combined = pd.concat([syn_df, real_df], ignore_index=True)
    combined.to_csv(args.out / "lowlight_mu_y.csv", index=False)

    summary_path = args.out / "lowlight_dispersion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Synthetic  n={summary['synthetic_n']}  "
          f"mean µY={summary['synthetic_mean_mu_y']:.4f}  "
          f"IQR={summary['synthetic_iqr_mu_y']:.4f}")
    print(f"Real       n={summary['real_n']}  "
          f"mean µY={summary['real_mean_mu_y']:.4f}  "
          f"IQR={summary['real_iqr_mu_y']:.4f}")
    print(f"Dispersion ratio: {summary['dispersion_ratio']:.3f}  "
          f"({'OK' if summary['sufficient_dispersion'] else 'INSUFFICIENT — collapse bins per §3.3.7'})")
    if args.t1 and args.t2:
        print(f"  lum_severe (<{args.t1}):   {summary['synthetic_pct_lum_severe']:.1f}%")
        print(f"  lum_moderate [{args.t1},{args.t2}): {summary['synthetic_pct_lum_moderate']:.1f}%")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
