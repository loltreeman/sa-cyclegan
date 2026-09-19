"""FID and LPIPS evaluation for the D4 ablation (§3.6.1, §3.6.2).

Compares generated images (encoder-only vs dual self-attention SA-CycleGAN)
against real target-domain images.  Lower is better for both metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

__all__ = [
    "compute_fid",
    "compute_kid",
    "compute_lpips",
    "evaluate_generator",
    "compare_ablations",
]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _image_paths(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMG_EXTS)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _ImgDataset(Dataset):
    """Loads images from a directory, applying `transform`."""

    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class _PairedDataset(Dataset):
    """Loads matched (real, fake) pairs by filename."""

    def __init__(self, pairs: list[tuple[Path, Path]], transform) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        real_img = Image.open(self.pairs[idx][0]).convert("RGB")
        fake_img = Image.open(self.pairs[idx][1]).convert("RGB")
        return self.transform(real_img), self.transform(fake_img)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(
    real_dir: Path,
    fake_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    max_samples: int | None = None,
) -> float:
    """Compute FID between real_dir and fake_dir using torchmetrics InceptionV3.

    Returns FID score (lower is better).
    Images are resized to 299×299 (InceptionV3 input) and normalized to [0, 255] uint8.
    """
    try:
        from torchmetrics.image import FrechetInceptionDistance
    except ImportError as exc:
        raise ImportError(
            "torchmetrics is required for FID.  "
            "Install with: pip install torchmetrics[image]"
        ) from exc

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # FID expects uint8 tensors in [0, 255], shape (N, 3, H, W).
    to_tensor = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),                        # float [0, 1]
        transforms.Lambda(lambda t: (t * 255).to(torch.uint8)),
    ])

    real_paths = _image_paths(real_dir)
    fake_paths = _image_paths(fake_dir)
    if max_samples is not None:
        real_paths = real_paths[:max_samples]
        fake_paths = fake_paths[:max_samples]

    fid = FrechetInceptionDistance(normalize=False).to(dev)

    real_loader = DataLoader(
        _ImgDataset(real_paths, to_tensor),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    fake_loader = DataLoader(
        _ImgDataset(fake_paths, to_tensor),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    with torch.no_grad():
        for batch in real_loader:
            fid.update(batch.to(dev), real=True)
        for batch in fake_loader:
            fid.update(batch.to(dev), real=False)

    return float(fid.compute())


# ---------------------------------------------------------------------------
# KID
# ---------------------------------------------------------------------------

def compute_kid(
    real_dir: Path,
    fake_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    max_samples: int | None = None,
    subsets: int = 100,
    subset_size: int = 1000,
) -> tuple[float, float]:
    """Compute KID between real_dir and fake_dir.

    Returns (kid_mean, kid_std). Lower is better.
    Uses torchmetrics.image.KernelInceptionDistance with polynomial kernel.
    Images resized to 299×299, normalized to [0, 255] uint8.

    KID is an unbiased estimator of squared MMD under a polynomial kernel
    (Binkowski et al.). Unlike FID it requires no Gaussian assumption and no
    covariance estimation; at the sample sizes in this study it is the more
    appropriate distributional metric (§3.6.1).
    """
    try:
        from torchmetrics.image import KernelInceptionDistance
    except ImportError as exc:
        raise ImportError(
            "torchmetrics is required for KID.  "
            "Install with: pip install torchmetrics[image]"
        ) from exc

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # KID expects uint8 tensors in [0, 255], shape (N, 3, H, W) when normalize=False.
    # Matches compute_fid's preprocessing for consistency.
    to_tensor = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),                        # float [0, 1]
        transforms.Lambda(lambda t: (t * 255).to(torch.uint8)),
    ])

    real_paths = _image_paths(real_dir)
    fake_paths = _image_paths(fake_dir)
    if max_samples is not None:
        real_paths = real_paths[:max_samples]
        fake_paths = fake_paths[:max_samples]

    kid = KernelInceptionDistance(
        normalize=False,
        subsets=subsets,
        subset_size=subset_size,
    ).to(dev)

    real_loader = DataLoader(
        _ImgDataset(real_paths, to_tensor),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    fake_loader = DataLoader(
        _ImgDataset(fake_paths, to_tensor),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    with torch.no_grad():
        for batch in real_loader:
            kid.update(batch.to(dev), real=True)
        for batch in fake_loader:
            kid.update(batch.to(dev), real=False)

    kid_mean, kid_std = kid.compute()
    return float(kid_mean), float(kid_std)


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def compute_lpips(
    real_dir: Path,
    fake_dir: Path,
    *,
    net: str = "alex",
    device: str = "cuda",
    batch_size: int = 16,
    max_samples: int | None = None,
) -> float:
    """Compute mean LPIPS between matched pairs real_dir/img.jpg ↔ fake_dir/img.jpg.

    Only files present in BOTH directories are evaluated (matched by filename).
    Returns mean LPIPS (lower is better).
    Images normalized to [-1, 1].
    """
    try:
        import lpips as lpips_lib
    except ImportError as exc:
        raise ImportError(
            "lpips is required for LPIPS computation.  "
            "Install with: pip install lpips"
        ) from exc

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    real_by_name = {p.name: p for p in _image_paths(real_dir)}
    fake_by_name = {p.name: p for p in _image_paths(fake_dir)}
    common = sorted(real_by_name.keys() & fake_by_name.keys())
    if not common:
        raise ValueError(
            f"No filename matches between {real_dir} and {fake_dir}.  "
            f"Paired LPIPS requires identical filenames in both directories."
        )
    pairs = [(real_by_name[n], fake_by_name[n]) for n in common]
    if max_samples is not None:
        pairs = pairs[:max_samples]

    # lpips expects float tensors in [-1, 1], shape (N, 3, H, W).
    to_tensor = transforms.Compose([
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # → [-1, 1]
    ])

    loss_fn = lpips_lib.LPIPS(net=net).to(dev)
    loss_fn.eval()

    loader = DataLoader(
        _PairedDataset(pairs, to_tensor),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    scores: list[float] = []
    with torch.no_grad():
        for real_batch, fake_batch in loader:
            d = loss_fn(real_batch.to(dev), fake_batch.to(dev))
            scores.extend(d.squeeze().cpu().tolist()
                          if d.numel() > 1 else [float(d.item())])

    return float(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_generator(
    real_dir: Path,
    fake_dir: Path,
    *,
    device: str = "cuda",
    fid_batch: int = 32,
    kid_batch: int = 32,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    lpips_batch: int = 16,
    lpips_net: str = "alex",
) -> dict:
    """Run FID, KID, and LPIPS; return a summary dict.

    Returns {
        "fid": float,
        "kid_mean": float,
        "kid_std": float,
        "lpips_mean": float,
        "n_pairs": int,
        "n_real": int,
        "n_fake": int,
    }.

    KID is taken as authoritative when FID and KID order configurations
    differently (§3.6.1 line 764).
    """
    n_real = len(_image_paths(real_dir))
    n_fake = len(_image_paths(fake_dir))

    fid_score = compute_fid(real_dir, fake_dir, device=device, batch_size=fid_batch)
    kid_mean, kid_std = compute_kid(
        real_dir, fake_dir,
        device=device, batch_size=kid_batch,
        subsets=kid_subsets, subset_size=kid_subset_size,
    )

    real_by_name = {p.name for p in _image_paths(real_dir)}
    fake_by_name = {p.name for p in _image_paths(fake_dir)}
    n_pairs = len(real_by_name & fake_by_name)

    lpips_score = compute_lpips(
        real_dir, fake_dir,
        net=lpips_net, device=device, batch_size=lpips_batch,
    )

    return {
        "fid":        fid_score,
        "kid_mean":   kid_mean,
        "kid_std":    kid_std,
        "lpips_mean": lpips_score,
        "n_pairs":    n_pairs,
        "n_real":     n_real,
        "n_fake":     n_fake,
    }


# ---------------------------------------------------------------------------
# Ablation comparison
# ---------------------------------------------------------------------------

def compare_ablations(
    results_dir: Path,
    ablation_configs: dict[str, tuple[Path, Path]],
    *,
    device: str = "cuda",
) -> pd.DataFrame:
    """Evaluate multiple generator configurations and return a comparison DataFrame.

    Intended for the D4 ablation: encoder-only vs dual attention at heavy rain.
    Columns: name, fid, kid_mean, kid_std, lpips_mean, n_pairs.
    KID is the authoritative ordering metric when FID and KID disagree (§3.6.1).
    Saves results to results_dir/fid_lpips_ablation.parquet and .csv.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name, (real_dir, fake_dir) in ablation_configs.items():
        print(f"Evaluating {name} …", flush=True)
        metrics = evaluate_generator(real_dir, fake_dir, device=device)
        rows.append({"name": name, **metrics})
        print(
            f"  FID={metrics['fid']:.4f}"
            f"  KID={metrics['kid_mean']:.6f}±{metrics['kid_std']:.6f}"
            f"  LPIPS={metrics['lpips_mean']:.4f}"
            f"  pairs={metrics['n_pairs']}"
        )

    df = pd.DataFrame(rows)
    parquet_out = results_dir / "fid_lpips_ablation.parquet"
    csv_out     = results_dir / "fid_lpips_ablation.csv"
    df.to_parquet(parquet_out, index=False, engine="pyarrow")
    df.to_csv(csv_out, index=False)
    print(f"Saved ablation results → {parquet_out} and {csv_out}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FID and LPIPS evaluation for generated images (§3.6.1, §3.6.2)."
    )
    p.add_argument("--real",   required=True, type=Path,
                   help="Directory of real target-domain images")
    p.add_argument("--fake",   required=True, type=Path,
                   help="Directory of generated images")
    p.add_argument("--net",    default="alex", choices=["alex", "vgg", "squeeze"],
                   help="LPIPS backbone (default: alex, recommended by lpips authors)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out",    default=None, type=Path,
                   help="Write JSON results to this path (optional)")
    return p


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    args = _build_parser().parse_args()

    metrics = evaluate_generator(
        real_dir=args.real,
        fake_dir=args.fake,
        device=args.device,
    )
    print(
        f"FID:        {metrics['fid']:.4f}\n"
        f"KID mean:   {metrics['kid_mean']:.6f}\n"
        f"KID std:    {metrics['kid_std']:.6f}\n"
        f"LPIPS mean: {metrics['lpips_mean']:.4f}\n"
        f"Pairs:      {metrics['n_pairs']}\n"
        f"Real imgs:  {metrics['n_real']}\n"
        f"Fake imgs:  {metrics['n_fake']}"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved → {args.out}")
