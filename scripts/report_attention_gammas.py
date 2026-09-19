"""Attention gamma reporting — §3.6.1.

Extracts and reports the learned attention gating scalars (γ) for every
trained SA-CycleGAN checkpoint.  Two measures are produced per checkpoint:

1. Raw gamma values: G.enc_attn.gamma, G.dec_attn.gamma, F_.enc_attn.gamma,
   F_.dec_attn.gamma — signed, as §3.6.1 lines 752–753 require.

2. Scale-normalised contribution (§3.6.1 lines 753–756):
       ||γ · o_i||_F / ||x_i||_F  =  ||output - input||_F / ||input||_F
   averaged over a fixed sample of source frames.  Since the SelfAttention
   forward computes y_i = γ·o_i + x_i, the gated attention output is
   y_i − x_i = γ·o_i, so no re-implementation of the internals is needed.

Trajectory across checkpoints (§3.6.1 line 756) is assembled in
report_trajectory(), which loads every epoch_NNNN.pt in ckpt/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from src.gan.models import ResNet9Generator


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def _load_generator(state_dict: dict, cfg: dict | None = None) -> ResNet9Generator:
    """Instantiate ResNet9Generator and load state_dict.

    cfg is the gan.generator block from base.yaml.  Falls back to defaults
    (ngf=64, enc_attn=True, dec_attn=True) so the script works standalone.
    """
    ngf       = cfg["ngf"]       if cfg else 64
    enc_attn  = cfg.get("enc_attn", True)  if cfg else True
    dec_attn  = cfg.get("dec_attn", True)  if cfg else True
    gamma_init = 0.0

    # Infer attention flags from state_dict keys so ablation checkpoints load
    # correctly even when cfg flags are wrong.
    has_enc = any("enc_attn.gamma" in k for k in state_dict)
    has_dec = any("dec_attn.gamma" in k for k in state_dict)

    G = ResNet9Generator(ngf=ngf, enc_attn=has_enc, dec_attn=has_dec,
                         gamma_init=gamma_init)
    G.load_state_dict(state_dict)
    return G


# ---------------------------------------------------------------------------
# Raw gamma extraction
# ---------------------------------------------------------------------------

def extract_gammas(checkpoint_path: Path) -> dict:
    """Extract enc_attn.gamma and dec_attn.gamma from a GAN checkpoint.

    Returns {
        "epoch": int,
        "G_enc_gamma": float | None,
        "G_dec_gamma": float | None,
        "F_enc_gamma": float | None,
        "F_dec_gamma": float | None,
    }

    Returns None for a key when the corresponding module is nn.Identity()
    (ablation configuration — encoder-only or baseline CycleGAN).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def _gamma(state_dict: dict, key: str) -> "float | None":
        return float(state_dict[key].item()) if key in state_dict else None

    return {
        "epoch":       int(ckpt["epoch"]),
        "G_enc_gamma": _gamma(ckpt["G"],  "enc_attn.gamma"),
        "G_dec_gamma": _gamma(ckpt["G"],  "dec_attn.gamma"),
        "F_enc_gamma": _gamma(ckpt["F_"], "enc_attn.gamma"),
        "F_dec_gamma": _gamma(ckpt["F_"], "dec_attn.gamma"),
    }


# ---------------------------------------------------------------------------
# Scale-normalised contribution
# ---------------------------------------------------------------------------

def _preprocess(img_path: Path, load_size: int, crop_size: int) -> torch.Tensor:
    """Resize then centre-crop to crop_size (deterministic, matches pool_b_builder)."""
    img = Image.open(img_path).convert("RGB")
    img = TF.resize(img, load_size)
    img = TF.center_crop(img, crop_size)
    t = TF.to_tensor(img)              # [0, 1]
    t = TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    return t.unsqueeze(0)              # 1×C×H×W


def _norm_contribution_for(
    G: ResNet9Generator,
    attr: str,
    source_frames: "list[Path]",
    device: str,
    load_size: int,
    crop_size: int,
) -> "float | None":
    """Mean ||output - input||_F / ||input||_F over source_frames for one attn module.

    Returns None if the module is nn.Identity() (no gamma parameter exists).
    """
    module: nn.Module = getattr(G, attr)
    if not hasattr(module, "gamma"):
        return None

    G.eval()
    ratios: list[float] = []
    hook_data: dict = {}

    def _hook(mod, inp, out):
        # inp is a tuple; inp[0] is the feature map entering attention
        hook_data["input"] = inp[0].detach()
        hook_data["output"] = out.detach()

    handle = module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for p in source_frames:
                x = _preprocess(p, load_size, crop_size).to(device)
                G(x)
                inp_f  = hook_data["input"]
                out_f  = hook_data["output"]
                # gamma * o_i = out - inp  (since y_i = gamma*o_i + x_i)
                gated  = out_f - inp_f
                num = torch.norm(gated.float(), p="fro").item()
                den = torch.norm(inp_f.float(),  p="fro").item()
                if den > 0:
                    ratios.append(num / den)
    finally:
        handle.remove()

    return float(np.mean(ratios)) if ratios else None


def compute_scale_normalized_contribution(
    checkpoint_path: Path,
    source_frames: "list[Path]",
    *,
    device: str = "cuda",
    crop_size: int = 256,
    load_size: int = 286,
) -> dict:
    """Compute mean ||gamma * attention_out||_F / ||input||_F for enc and dec.

    This is the scale-normalised contribution measure described in §3.6.1.
    Hooks into the SelfAttention forward pass to capture attention_out before
    the gamma multiplication, then computes the ratio.

    Returns {
        "epoch": int,
        "G_enc_norm_contribution": float | None,
        "G_dec_norm_contribution": float | None,
        "F_enc_norm_contribution": float | None,
        "F_dec_norm_contribution": float | None,
    }
    Uses centre crop (same as pool_b_builder.py — deterministic).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = int(ckpt["epoch"])

    result: dict = {"epoch": epoch}
    for gen_key, gen_label in [("G", "G"), ("F_", "F")]:
        G = _load_generator(ckpt[gen_key]).to(device)
        for attn_attr, label in [("enc_attn", "enc"), ("dec_attn", "dec")]:
            col = f"{gen_label}_{attn_attr.split('_')[0]}_norm_contribution"
            result[col] = _norm_contribution_for(
                G, attn_attr, source_frames, device, load_size, crop_size
            )

    return result


# ---------------------------------------------------------------------------
# Trajectory across checkpoints
# ---------------------------------------------------------------------------

def report_trajectory(
    runs_dir: Path,
    source_frames: "list[Path]",
    *,
    device: str = "cuda",
    include_scale_norm: bool = True,
) -> pd.DataFrame:
    """Load every checkpoint in runs_dir/ckpt/ and extract gamma trajectory.

    Returns DataFrame with columns:
      epoch, G_enc_gamma, G_dec_gamma, F_enc_gamma, F_dec_gamma,
      [G_enc_norm_contribution, G_dec_norm_contribution,
       F_enc_norm_contribution, F_dec_norm_contribution]  (if include_scale_norm)

    Checkpoints are sorted by epoch number extracted from the filename
    epoch_NNNN.pt or from the checkpoint dict for final.pt.
    Saves to runs_dir/attention_gammas.parquet and .csv.
    """
    ckpt_dir = runs_dir / "ckpt"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No ckpt/ directory found under {runs_dir}")

    ckpt_files = sorted(
        ckpt_dir.glob("*.pt"),
        key=lambda p: (
            int(p.stem.split("_")[-1]) if p.stem.startswith("epoch_") else int(1e9)
        ),
    )

    if not ckpt_files:
        raise FileNotFoundError(f"No .pt checkpoints found in {ckpt_dir}")

    rows: list[dict] = []
    for ckpt_path in ckpt_files:
        print(f"  loading {ckpt_path.name}…")
        row = extract_gammas(ckpt_path)

        if include_scale_norm and source_frames:
            contrib = compute_scale_normalized_contribution(
                ckpt_path,
                source_frames,
                device=device,
            )
            # merge, skipping duplicate "epoch"
            row.update({k: v for k, v in contrib.items() if k != "epoch"})

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)

    df.to_parquet(runs_dir / "attention_gammas.parquet", index=False, engine="pyarrow")
    df.to_csv(runs_dir / "attention_gammas.csv", index=False, encoding="utf-8")
    print(f"  saved → {runs_dir / 'attention_gammas.parquet'}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Report attention gamma scalars and scale-normalised contributions "
            "across all checkpoints in a GAN run directory (§3.6.1)."
        )
    )
    p.add_argument("--runs-dir", required=True, type=Path, dest="runs_dir",
                   help="Path to a single GAN run dir, e.g. runs/gan/rain_heavy")
    p.add_argument("--source-frames", type=Path, default=None, dest="source_frames",
                   help="Directory of source images for scale-norm computation")
    p.add_argument("--n-frames", type=int, default=50, dest="n_frames",
                   help="How many source frames to use for scale-norm (default: 50)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-scale-norm", action="store_true", dest="no_scale_norm",
                   help="Skip forward-pass scale-norm measure (faster)")
    return p


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    args = _build_parser().parse_args()

    frames: list[Path] = []
    if args.source_frames and not args.no_scale_norm:
        src_dir = Path(args.source_frames)
        all_imgs = sorted(
            p for p in src_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        frames = all_imgs[: args.n_frames]
        if not frames:
            print(
                f"[warn] No images found in {src_dir}; skipping scale-norm.",
                file=sys.stderr,
            )

    df = report_trajectory(
        runs_dir=args.runs_dir,
        source_frames=frames,
        device=args.device,
        include_scale_norm=(not args.no_scale_norm and bool(frames)),
    )

    print(f"\nGamma trajectory ({len(df)} checkpoints):")
    print(df.to_string(index=False))
