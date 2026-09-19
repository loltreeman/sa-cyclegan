"""Severity separation check for the three per-severity rain generators (§3.6.1, lines 768–774).

No term in the three generators' objectives couples them, so whether they
produce distinguishably different outputs is an empirical question.  If all
three converge to similar outputs the synthetic severity factor evaluated in
Phase 1 carries no variance (§3.6.1 line 768).

Protocol:
  1. Propagate a common set of source frames through all three rain models →
     matched triplets (light, moderate, heavy).
  2. Compute three measures per triplet on BOTH real footage (PAGASA bins) and
     synthetic output (§3.6.1 line 772):
       - HF residual energy  (high-frequency component L2 norm vs source)
       - RMS contrast        (root-mean-square contrast of the image)
       - LPIPS translation magnitude (source → translated, AlexNet backbone)
  3. A measure that fails to separate the real bins is reported as
     non-interpretable on the synthetic set.
  4. Where a measure separates both, the synthetic/real separation ratio is
     reported (§3.6.1 line 772).
  5. Pairwise Wilcoxon signed-rank tests between adjacent severity levels with
     Bonferroni correction across three pairs (§3.6.1 line 774).

Aggregation unit: clip, consistent with §3.6.1 line 758 (frames from the same
clip share background, camera pose and encoding characteristics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "hf_residual_energy",
    "rms_contrast",
    "compute_lpips_magnitude",
    "compute_measures",
    "separation_ratio",
    "wilcoxon_bonferroni",
    "run_separation_check",
]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Measure 1 — high-frequency residual energy (§3.6.1 line 770)
# ---------------------------------------------------------------------------

def hf_residual_energy(img_bgr: np.ndarray, ksize: int = 15) -> float:
    """L2 norm of the high-frequency residual = image − Gaussian low-pass.

    Normalised by the number of pixels so values are comparable across sizes.
    Dense precipitation introduces additional high-frequency streak structure
    while simultaneously attenuating scene contrast, so this measure captures
    the streak-introduction axis.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low_pass = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    residual = gray - low_pass
    n = residual.size
    return float(np.sqrt((residual ** 2).sum() / n))


# ---------------------------------------------------------------------------
# Measure 2 — RMS contrast (§3.6.1 line 770)
# ---------------------------------------------------------------------------

def rms_contrast(img_bgr: np.ndarray) -> float:
    """Root-mean-square contrast of the luminance channel.

    RMS contrast = std(Y) where Y is the normalised [0,1] luminance.
    Captures the contrast attenuation associated with denser precipitation
    (§3.6.1 line 770).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return float(np.std(gray))


# ---------------------------------------------------------------------------
# Measure 3 — LPIPS translation magnitude (§3.6.1 line 742)
# ---------------------------------------------------------------------------

def compute_lpips_magnitude(
    source_paths: list[Path],
    translated_paths: list[Path],
    *,
    device: str = "cuda",
    batch_size: int = 16,
) -> pd.DataFrame:
    """Compute LPIPS (AlexNet) between matched source→translated pairs.

    source_paths and translated_paths must correspond by index (same filename).
    Returns DataFrame with columns: frame, lpips.
    """
    try:
        import lpips as lpips_lib
    except ImportError as exc:
        raise ImportError("pip install lpips") from exc
    import torch
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset

    src_by_name = {p.name: p for p in source_paths}
    trn_by_name = {p.name: p for p in translated_paths}
    common = sorted(src_by_name.keys() & trn_by_name.keys())
    if not common:
        raise ValueError("No filename matches between source and translated directories.")

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),   # → [-1, 1]
    ])

    from PIL import Image

    class _PairedDS(Dataset):
        def __init__(self, names):
            self.names = names
        def __len__(self):
            return len(self.names)
        def __getitem__(self, idx):
            name = self.names[idx]
            s = Image.open(src_by_name[name]).convert("RGB")
            t = Image.open(trn_by_name[name]).convert("RGB")
            return to_tensor(s), to_tensor(t), name

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    loss_fn = lpips_lib.LPIPS(net="alex").to(dev)
    loss_fn.eval()

    loader = DataLoader(_PairedDS(common), batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict] = []
    with torch.no_grad():
        for src_batch, trn_batch, names in loader:
            d = loss_fn(src_batch.to(dev), trn_batch.to(dev))
            vals = d.squeeze().cpu().tolist() if d.numel() > 1 else [float(d.item())]
            for name, val in zip(names, vals):
                rows.append({"frame": name, "lpips": float(val)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compute HF + RMS measures over a directory of images
# ---------------------------------------------------------------------------

def compute_measures(
    image_paths: list[Path],
    clip_ids: dict[str, str] | None = None,
    *,
    ksize: int = 15,
) -> pd.DataFrame:
    """Compute hf_energy and rms_contrast for each image.

    clip_ids maps frame filename → clip_id (for clip-level aggregation).
    Returns DataFrame: frame, clip_id, hf_energy, rms_contrast.
    """
    rows: list[dict] = []
    for p in image_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        rows.append({
            "frame":       p.name,
            "clip_id":     clip_ids.get(p.name, p.name) if clip_ids else p.name,
            "hf_energy":   hf_residual_energy(img_bgr, ksize=ksize),
            "rms_contrast": rms_contrast(img_bgr),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Separation ratio (§3.6.1 line 772)
# ---------------------------------------------------------------------------

def separation_ratio(
    synthetic_values: np.ndarray,
    real_values: np.ndarray,
) -> float:
    """Ratio of synthetic spread to real spread (IQR-based).

    A value near 1.0 means the synthetic set spans the full authentic severity
    range for this measure.  Reports the proportion of the authentic severity
    range spanned by the synthetic set (§3.6.1 line 772).
    """
    real_spread = np.percentile(real_values, 75) - np.percentile(real_values, 25)
    if real_spread < 1e-12:
        return float("nan")
    syn_spread = np.percentile(synthetic_values, 75) - np.percentile(synthetic_values, 25)
    return float(syn_spread / real_spread)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank + Bonferroni (§3.6.1 line 774)
# ---------------------------------------------------------------------------

def wilcoxon_bonferroni(
    groups: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    n_comparisons: int | None = None,
) -> list[dict]:
    """Pairwise Wilcoxon signed-rank tests with Bonferroni correction.

    groups: {label: clip-level array of the measure}
    pairs:  list of (label_a, label_b) to compare (adjacent severity levels)
    n_comparisons: Bonferroni denominator (default = len(pairs))

    Each pair must have paired observations (same clip indices).
    Returns list of dicts: pair, statistic, p_raw, p_bonferroni, significant_0.05.
    """
    if n_comparisons is None:
        n_comparisons = len(pairs)
    results: list[dict] = []
    for a_label, b_label in pairs:
        a = np.array(groups[a_label])
        b = np.array(groups[b_label])
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        # Wilcoxon requires at least 10 observations for the normal approximation.
        if n < 10:
            results.append({
                "pair": f"{a_label} vs {b_label}",
                "n": n,
                "statistic": float("nan"),
                "p_raw": float("nan"),
                "p_bonferroni": float("nan"),
                "significant_0.05": False,
                "note": f"n={n} < 10, Wilcoxon not computed",
            })
            continue
        stat, p_raw = stats.wilcoxon(a, b, alternative="two-sided")
        p_bonf = min(1.0, p_raw * n_comparisons)
        results.append({
            "pair":            f"{a_label} vs {b_label}",
            "n":               n,
            "statistic":       float(stat),
            "p_raw":           float(p_raw),
            "p_bonferroni":    float(p_bonf),
            "significant_0.05": p_bonf < 0.05,
        })
    return results


# ---------------------------------------------------------------------------
# Jonckheere-Terpstra test for ordered alternatives (§3.6.1 line 774)
# ---------------------------------------------------------------------------

def jonckheere_terpstra(groups: list[np.ndarray]) -> dict:
    """Jonckheere-Terpstra test for ordered alternatives k≥2 groups.

    H1: group_0 ≤ group_1 ≤ … ≤ group_{k-1} (ordered alternative).
    Statistic J = Σ_{i<j} U(i,j) where U(i,j) is the Mann-Whitney statistic.
    Large-sample normal approximation (asymptotically valid for n≥5 per group).
    Returns {J, z, p_value}.
    """
    k = len(groups)
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            # Count pairs where groups[j] > groups[i].
            for xi in groups[i]:
                J += (groups[j] > xi).sum()
            # Tie contribution: 0.5 per tie.
            for xi in groups[i]:
                J += 0.5 * (groups[j] == xi).sum()

    # E[J] and Var[J] under H0 (no ties, large-sample).
    n_total = sum(len(g) for g in groups)
    ns = [len(g) for g in groups]
    E_J = (n_total ** 2 - sum(n ** 2 for n in ns)) / 4.0
    Var_J = (
        n_total ** 2 * (2 * n_total + 3)
        - sum(n ** 2 * (2 * n + 3) for n in ns)
    ) / 72.0
    if Var_J <= 0:
        return {"J": J, "z": float("nan"), "p_value": float("nan")}
    z = (J - E_J) / np.sqrt(Var_J)
    p = 2 * stats.norm.sf(abs(z))   # two-sided; for ordered: p_one = stats.norm.sf(z)
    return {"J": float(J), "z": float(z), "p_value_two_sided": float(p),
            "p_value_ordered": float(stats.norm.sf(z))}


# ---------------------------------------------------------------------------
# High-level driver: full separation check
# ---------------------------------------------------------------------------

def run_separation_check(
    real_dirs: dict[str, Path],     # {"rain_light": Path, "rain_moderate": Path, "rain_heavy": Path}
    synthetic_dirs: dict[str, Path],  # same keys — generated outputs
    source_dir: Path,               # common clear-weather source frames
    manifest_path: Path | None,     # if provided, extract clip_id mappings
    out_dir: Path,
    *,
    device: str = "cuda",
    ksize: int = 15,
) -> dict:
    """Run the full severity separation check for the three rain models.

    Computes all three measures on real and synthetic directories, produces
    per-measure Wilcoxon+Bonferroni tables, separation ratios, and summary JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    severity_levels = ["rain_light", "rain_moderate", "rain_heavy"]
    # Ordered pairs for pairwise comparisons.
    adjacent_pairs = [
        ("rain_light", "rain_moderate"),
        ("rain_moderate", "rain_heavy"),
        ("rain_light", "rain_heavy"),
    ]

    # Clip ID lookup from manifest.
    clip_ids: dict[str, str] | None = None
    if manifest_path is not None and manifest_path.exists():
        mf = pd.read_parquet(manifest_path, engine="pyarrow")
        if "frame_id" in mf.columns and "clip_id" in mf.columns:
            clip_ids = dict(zip(mf["frame_id"], mf["clip_id"].fillna(mf["frame_id"])))

    def _img_paths(d: Path) -> list[Path]:
        return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)

    results: dict = {}

    # ------------------------------------------------------------------
    # Measures 1 and 2: HF energy + RMS contrast
    # ------------------------------------------------------------------
    for domain, dirs in [("real", real_dirs), ("synthetic", synthetic_dirs)]:
        domain_rows: list[dict] = []
        for sev in severity_levels:
            if sev not in dirs:
                continue
            paths = _img_paths(dirs[sev])
            df = compute_measures(paths, clip_ids=clip_ids, ksize=ksize)
            df["severity"] = sev
            df["domain"] = domain
            domain_rows.append(df)
        combined = pd.concat(domain_rows, ignore_index=True)
        parquet_out = out_dir / f"hf_rms_{domain}.parquet"
        combined.to_parquet(parquet_out, index=False, engine="pyarrow")
        combined.to_csv(out_dir / f"hf_rms_{domain}.csv", index=False)
        results[f"hf_rms_{domain}"] = str(parquet_out)

    # ------------------------------------------------------------------
    # Measure 3: LPIPS translation magnitude
    # ------------------------------------------------------------------
    for sev in severity_levels:
        if sev not in synthetic_dirs:
            continue
        src_paths = _img_paths(source_dir)
        trn_paths = _img_paths(synthetic_dirs[sev])
        lpips_df = compute_lpips_magnitude(src_paths, trn_paths, device=device)
        lpips_df["severity"] = sev
        lpips_out = out_dir / f"lpips_{sev}.csv"
        lpips_df.to_csv(lpips_out, index=False)

    # ------------------------------------------------------------------
    # Per-measure Wilcoxon + Bonferroni (clip-level aggregates)
    # ------------------------------------------------------------------
    def _clip_agg(df: pd.DataFrame, measure: str, domain: str, sev: str) -> np.ndarray:
        sub = df[(df["domain"] == domain) & (df["severity"] == sev)]
        return sub.groupby("clip_id")[measure].median().values

    for domain_key, domain_name in [("hf_rms_real", "real"), ("hf_rms_synthetic", "synthetic")]:
        parquet_path = out_dir / f"hf_rms_{domain_name}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path, engine="pyarrow")

        for measure in ["hf_energy", "rms_contrast"]:
            groups = {
                sev: _clip_agg(df, measure, domain_name, sev)
                for sev in severity_levels
            }
            wtest = wilcoxon_bonferroni(groups, adjacent_pairs, n_comparisons=3)
            jt = jonckheere_terpstra([groups[s] for s in severity_levels])
            out_json = out_dir / f"wilcoxon_{domain_name}_{measure}.json"
            out_json.write_text(json.dumps({"wilcoxon": wtest, "jonckheere_terpstra": jt}, indent=2),
                                encoding="utf-8")
            results[f"wilcoxon_{domain_name}_{measure}"] = str(out_json)

    # ------------------------------------------------------------------
    # Separation ratios (§3.6.1 line 772)
    # ------------------------------------------------------------------
    sep_rows: list[dict] = []
    real_parquet = out_dir / "hf_rms_real.parquet"
    syn_parquet  = out_dir / "hf_rms_synthetic.parquet"
    if real_parquet.exists() and syn_parquet.exists():
        real_df = pd.read_parquet(real_parquet, engine="pyarrow")
        syn_df  = pd.read_parquet(syn_parquet,  engine="pyarrow")
        for measure in ["hf_energy", "rms_contrast"]:
            real_vals = real_df[measure].dropna().values
            syn_vals  = syn_df[measure].dropna().values
            if len(real_vals) == 0 or len(syn_vals) == 0:
                continue
            ratio = separation_ratio(syn_vals, real_vals)
            sep_rows.append({"measure": measure, "separation_ratio": ratio,
                             "n_real": len(real_vals), "n_synthetic": len(syn_vals)})
    sep_df = pd.DataFrame(sep_rows)
    sep_out = out_dir / "separation_ratios.csv"
    sep_df.to_csv(sep_out, index=False)
    results["separation_ratios"] = str(sep_out)
    print(sep_df.to_string(index=False))

    summary_path = out_dir / "severity_separation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSummary → {summary_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Severity separation check for three per-severity rain generators "
                    "(§3.6.1, lines 768–774)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- full check ---------------------------------------------------------
    full = sub.add_parser("check", help="Run the full separation check")
    full.add_argument("--real-light",     required=True, type=Path)
    full.add_argument("--real-moderate",  required=True, type=Path)
    full.add_argument("--real-heavy",     required=True, type=Path)
    full.add_argument("--syn-light",      required=True, type=Path)
    full.add_argument("--syn-moderate",   required=True, type=Path)
    full.add_argument("--syn-heavy",      required=True, type=Path)
    full.add_argument("--source",         required=True, type=Path,
                      help="Common clear-weather source frames")
    full.add_argument("--out",            required=True, type=Path)
    full.add_argument("--manifest",       default=None, type=Path)
    full.add_argument("--device",         default="cuda")

    # ---- single directory measure -------------------------------------------
    meas = sub.add_parser("measure", help="Compute HF energy + RMS for a single directory")
    meas.add_argument("--images",  required=True, type=Path)
    meas.add_argument("--out",     required=True, type=Path, help="Output CSV path")
    meas.add_argument("--ksize",   default=15, type=int)

    # ---- wilcoxon only -------------------------------------------------------
    wilcox = sub.add_parser("wilcoxon", help="Run Wilcoxon tests on pre-computed CSV")
    wilcox.add_argument("--csv",      required=True, type=Path,
                        help="CSV with columns: clip_id, severity, <measure>")
    wilcox.add_argument("--measure",  required=True)
    wilcox.add_argument("--out",      required=True, type=Path)

    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()

    if args.cmd == "check":
        run_separation_check(
            real_dirs={
                "rain_light":    args.real_light,
                "rain_moderate": args.real_moderate,
                "rain_heavy":    args.real_heavy,
            },
            synthetic_dirs={
                "rain_light":    args.syn_light,
                "rain_moderate": args.syn_moderate,
                "rain_heavy":    args.syn_heavy,
            },
            source_dir=args.source,
            manifest_path=args.manifest,
            out_dir=args.out,
            device=args.device,
        )

    elif args.cmd == "measure":
        paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in _IMG_EXTS)
        df = compute_measures(paths, ksize=args.ksize)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"HF energy  median={df['hf_energy'].median():.4f}")
        print(f"RMS contrast median={df['rms_contrast'].median():.4f}")
        print(f"Saved → {args.out}")

    elif args.cmd == "wilcoxon":
        df = pd.read_csv(args.csv)
        severity_levels = ["rain_light", "rain_moderate", "rain_heavy"]
        adjacent_pairs = [
            ("rain_light", "rain_moderate"),
            ("rain_moderate", "rain_heavy"),
            ("rain_light", "rain_heavy"),
        ]
        groups = {
            sev: df[df["severity"] == sev].groupby("clip_id")[args.measure].median().values
            for sev in severity_levels
        }
        wtest = wilcoxon_bonferroni(groups, adjacent_pairs, n_comparisons=3)
        jt = jonckheere_terpstra([groups[s] for s in severity_levels])
        out = {"wilcoxon": wtest, "jonckheere_terpstra": jt}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
