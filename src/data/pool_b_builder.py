"""Pool B builder — §3.3.6 bounding-box transfer protocol.

Applies a trained SA-CycleGAN generator G to Pool A clear-weather frames and
writes the synthesized frames to pool_b_root, copying source annotations
unchanged (§3.3.6: spatial structure is preserved by the feature consistency
loss, so bbox coordinates transfer directly).

Each run of this script produces one severity subset of Pool B:
  pool_b_root/images/train/{source_stem}_{run_name}.jpg
  pool_b_root/labels/train/{source_stem}_{run_name}.txt

The returned DataFrame has the same schema as manifest.py COLUMNS with:
  pool        = "B"
  rain_severity  = run config's severity (e.g. "rain_heavy")
  rain_source = f"synthetic:{run_name}"
  frame_id    = f"B_{source_frame_id}_{run_name}"
  ann_path    = relative to pool_b_root
  split_role  = "train"
  fold        = same as source frame
  mu_y, is_ir, lum_band = pd.NA  (measured after generation, not before)
  rho_sat, colour_div   = pd.NA
  n_instances, class_counts = copied from source manifest row
  sample_weight = 1.0  (recomputed by apply_oversampling later)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.data.manifest import COLUMNS, _cast
from src.gan.models import ResNet9Generator

__all__ = ["build_pool_b"]


def _load_generator(checkpoint_path: Path, device: torch.device) -> ResNet9Generator:
    """Load G from a train.py checkpoint dict.

    Reads enc_attn/dec_attn from the checkpoint when present so the
    architecture matches the one that was actually trained.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    enc_attn = state.get("enc_attn", True)
    dec_attn = state.get("dec_attn", True)
    ngf = state.get("ngf", 64)
    G = ResNet9Generator(ngf=ngf, enc_attn=enc_attn, dec_attn=dec_attn)
    G.load_state_dict(state["G"])
    G.to(device).eval()
    return G


def _preprocess(img_path: Path, load_size: int, crop_size: int) -> torch.Tensor:
    """Open → RGB → SmallestMaxSize(load_size) → CenterCrop(crop_size) → [-1, 1] tensor."""
    img = Image.open(img_path).convert("RGB")

    # SmallestMaxSize: scale so the smaller dimension == load_size.
    w, h = img.size
    if w < h:
        new_w = load_size
        new_h = int(h * load_size / w)
    else:
        new_h = load_size
        new_w = int(w * load_size / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Center crop to crop_size (deterministic — reproducible output per source frame).
    img = TF.center_crop(img, crop_size)

    # [0, 1] float → [-1, 1]
    t = TF.to_tensor(img)                     # C×H×W in [0, 1]
    t = TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return t


def _postprocess(tensor: torch.Tensor) -> np.ndarray:
    """tanh output in [-1, 1] → [0, 255] uint8 BGR ndarray for cv2.imwrite."""
    arr = tensor.squeeze(0).cpu().float()      # C×H×W
    arr = (arr * 0.5 + 0.5).clamp(0.0, 1.0)  # [0, 1]
    arr = (arr * 255).to(torch.uint8).numpy()  # C×H×W uint8
    arr = np.transpose(arr, (1, 2, 0))         # H×W×C RGB
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def build_pool_b(
    manifest_path: Path,
    pool_a_root: Path,
    pool_b_root: Path,
    checkpoint_path: Path,
    run_name: str,
    severity: str,
    *,
    device: str = "cuda",
    batch_size: int = 1,
    seed: int = 0,
    load_size: int = 286,
    crop_size: int = 256,
) -> pd.DataFrame:
    """Generate Pool B frames and return manifest rows.

    Processes all Pool A train-split frames in the manifest.
    Prints progress every 100 frames.
    Does NOT overwrite existing output files (idempotent — safe to resume).
    Returns DataFrame with manifest schema rows for the generated frames.

    §3.3.6: bounding boxes transfer unchanged because the generator's feature
    consistency loss preserves spatial structure; pixel coordinates are not
    modified.
    """
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    manifest = pd.read_parquet(manifest_path, engine="pyarrow")
    pool_a_frames = manifest[
        (manifest["pool"] == "A") & (manifest["split_role"] == "train")
    ].copy()

    if pool_a_frames.empty:
        raise ValueError("No Pool A train-split frames found in manifest.")

    G = _load_generator(checkpoint_path, dev)

    images_out = pool_b_root / "images" / "train"
    labels_out = pool_b_root / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_total = len(pool_a_frames)

    for i, (_, src) in enumerate(pool_a_frames.iterrows()):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{n_total}] generating …", flush=True)

        src_stem = src["frame_id"].removeprefix("B_") if src["frame_id"].startswith("B_") else src["frame_id"]
        out_stem = f"{src_stem}_{run_name}"
        out_img_path = images_out / f"{out_stem}.jpg"
        out_lbl_path = labels_out / f"{out_stem}.txt"

        # Idempotent: skip if output already exists.
        if not out_img_path.exists():
            src_img_path = pool_a_root / "images" / "train" / f"{src['frame_id'].removeprefix('A_')}.jpg"
            if not src_img_path.exists():
                # Try common extensions.
                for ext in (".jpeg", ".png"):
                    candidate = src_img_path.with_suffix(ext)
                    if candidate.exists():
                        src_img_path = candidate
                        break

            t = _preprocess(src_img_path, load_size, crop_size).unsqueeze(0).to(dev)
            with torch.no_grad():
                fake = G(t)
            bgr = _postprocess(fake)
            cv2.imwrite(str(out_img_path), bgr)

        # Copy annotation (bbox coordinates are unchanged — §3.3.6).
        if not out_lbl_path.exists() and pd.notna(src.get("ann_path")):
            src_ann = pool_a_root / src["ann_path"]
            if src_ann.exists():
                shutil.copy2(src_ann, out_lbl_path)

        ann_rel = str(PurePosixPath("labels/train") / f"{out_stem}.txt") if out_lbl_path.exists() else None

        # frame_id must not nest B_ prefixes if source already had one.
        new_frame_id = f"B_{src_stem}_{run_name}"

        rows.append({
            "frame_id":      new_frame_id,
            "clip_id":       src.get("clip_id", pd.NA),
            "event_id":      src.get("event_id", pd.NA),
            "camera_id":     src.get("camera_id", pd.NA),
            "timestamp":     src.get("timestamp", pd.NA),
            "pool":          "B",
            "rain_severity": severity,
            "rain_source":   f"synthetic:{run_name}",
            "mu_y":          pd.NA,
            "is_ir":         pd.NA,
            "rho_sat":       pd.NA,
            "colour_div":    pd.NA,
            "lum_band":      pd.NA,
            "split_role":    "train",
            "fold":          src.get("fold", pd.NA),
            "ann_path":      ann_rel,
            "n_instances":   src.get("n_instances", pd.NA),
            "class_counts":  src.get("class_counts", pd.NA),
            "sample_weight": 1.0,
        })

    print(f"Done. Generated {len(rows)} frames.", flush=True)
    df = _cast(pd.DataFrame(rows))
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pool B builder — §3.3.6 bounding-box transfer protocol."
    )
    p.add_argument("--manifest",   required=True, type=Path,
                   help="Path to manifest.parquet (Pool A must be present)")
    p.add_argument("--pool-a",     required=True, type=Path, dest="pool_a",
                   help="Root of Pool A data (images/ and labels/ subdirs)")
    p.add_argument("--pool-b",     required=True, type=Path, dest="pool_b",
                   help="Root to write Pool B frames into")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Generator checkpoint .pt saved by src/gan/train.py")
    p.add_argument("--run",        required=True,
                   help="Run name (e.g. rain_heavy); used in filenames and rain_source")
    p.add_argument("--severity",   required=True,
                   help="rain_severity value for output rows (e.g. rain_heavy)")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--load-size",  default=286, type=int, dest="load_size")
    p.add_argument("--crop-size",  default=256, type=int, dest="crop_size")
    p.add_argument("--seed",       default=0, type=int)
    return p


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    args = _build_parser().parse_args()

    df = build_pool_b(
        manifest_path=args.manifest,
        pool_a_root=args.pool_a,
        pool_b_root=args.pool_b,
        checkpoint_path=args.checkpoint,
        run_name=args.run,
        severity=args.severity,
        device=args.device,
        load_size=args.load_size,
        crop_size=args.crop_size,
        seed=args.seed,
    )

    out_path = args.pool_b.parent / f"pool_b_manifest_{args.run}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"Manifest rows written → {out_path}  ({len(df)} rows)")
