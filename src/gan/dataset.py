"""Unpaired image DataLoader for CycleGAN training.

Domain A: Pool A clear-weather frames (annotated, split_role="train").
Domain B: Pool C real rainy footage or Pool B synthetic frames, filtered
          by rain_severity and split_role="train".

Images are returned as (img_a, img_b) float tensors in [-1, 1],
at crop_size × crop_size (256 × 256 per §3.3.7).
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _img_path(ann_path: str, pool_root: Path) -> Path:
    """Derive image path from YOLO-layout ann_path.
    ann_path is relative to pool_root: labels/train/stem.txt → images/train/stem.jpg
    Tries .jpg, .jpeg, .png in that order.
    """
    rel = Path(ann_path)
    # Replace the leading "labels" component with "images" and drop the extension
    parts = list(rel.parts)
    parts[0] = "images"
    stem = Path(*parts).with_suffix("")
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = pool_root / stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No image found for annotation {ann_path!r} under {pool_root}"
    )


def _build_transform(load_size: int, crop_size: int, hflip: bool) -> A.Compose:
    ops: list[A.BasicTransform] = [
        A.SmallestMaxSize(max_size=load_size),
        A.RandomCrop(crop_size, crop_size),
    ]
    if hflip:
        ops.append(A.HorizontalFlip(p=0.5))
    ops += [
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    return A.Compose(ops)


class UnpairedDataset(Dataset):
    """Two independent lists of image paths sampled cyclically (len = max of both).

    Transforms applied to each image:
      - A.SmallestMaxSize(max_size=load_size)
      - A.RandomCrop(crop_size, crop_size)
      - A.HorizontalFlip(p=0.5)  (if hflip=True)
      - A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
      - ToTensorV2()
    Returns (img_a, img_b) tensors in [-1, 1].
    """

    def __init__(
        self,
        paths_a: list[Path],
        paths_b: list[Path],
        load_size: int = 286,
        crop_size: int = 256,
        hflip: bool = True,
    ) -> None:
        if not paths_a:
            raise ValueError("Domain A image list is empty")
        if not paths_b:
            raise ValueError("Domain B image list is empty")
        self.paths_a = paths_a
        self.paths_b = paths_b
        self._len = max(len(paths_a), len(paths_b))
        self.transform = _build_transform(load_size, crop_size, hflip)

    def __len__(self) -> int:
        return self._len

    def _load(self, path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    def __getitem__(self, idx: int):
        img_a = self._load(self.paths_a[idx % len(self.paths_a)])
        img_b = self._load(self.paths_b[idx % len(self.paths_b)])
        tensor_a = self.transform(image=img_a)["image"]
        tensor_b = self.transform(image=img_b)["image"]
        return tensor_a, tensor_b


def build_unpaired_loader(
    manifest_path: Path,
    pool_a_root: Path,
    pool_b_root: Path,
    severity: str,
    *,
    pool_b_pool_id: str = "C",
    fold_exclude: int | None = None,
    load_size: int = 286,
    crop_size: int = 256,
    hflip: bool = True,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Return a DataLoader[UnpairedDataset] for (domain_A_clear, domain_B_severity).

    Domain A = pool="A", split_role="train", fold != fold_exclude
    Domain B = pool=pool_b_pool_id, rain_severity=severity, split_role="train",
               fold != fold_exclude
    batch_size is always 1 (§3.3.7).
    """
    import pandas as pd  # lazy import: pandas not needed on non-training paths

    df = pd.read_parquet(manifest_path, engine="pyarrow")

    mask_a = (df["pool"] == "A") & (df["split_role"] == "train")
    mask_b = (
        (df["pool"] == pool_b_pool_id)
        & (df["rain_severity"] == severity)
        & (df["split_role"] == "train")
    )
    if fold_exclude is not None:
        mask_a = mask_a & (df["fold"] != fold_exclude)
        mask_b = mask_b & (df["fold"] != fold_exclude)

    rows_a = df.loc[mask_a & df["ann_path"].notna()]
    rows_b = df.loc[mask_b & df["ann_path"].notna()]

    paths_a = [_img_path(r, pool_a_root) for r in rows_a["ann_path"]]
    paths_b = [_img_path(r, pool_b_root) for r in rows_b["ann_path"]]

    dataset = UnpairedDataset(
        paths_a, paths_b,
        load_size=load_size,
        crop_size=crop_size,
        hflip=hflip,
    )

    # worker_init_fn seeds each worker so random crops differ across workers
    # even when the global seed is fixed; relevant when num_workers > 0.
    def _worker_init(worker_id: int) -> None:
        import random
        import numpy as np
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init,
        pin_memory=True,
    )
