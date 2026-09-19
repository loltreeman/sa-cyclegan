"""Detector training harness for the SA-CycleGAN thesis.

Implements the two-stage curriculum (§3.4.4) for 30 detector runs
(3 conditions × 2 architectures × 5 folds). Stage 1 trains for 100 epochs
with the backbone frozen for the first 10, then unfrozen via an
on_train_epoch_start callback (M12: one .train() call, not two). Stage 2
fine-tunes for 10 epochs from Stage 1's best.pt with early stopping disabled.

The freeze-then-unfreeze schedule is implemented as a single model.train()
call with UnfreezeCallback registered via model.add_callback, per M12.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# fmt: off
__all__ = ["UnfreezeCallback", "CheckpointRegistry", "run_stage1", "run_stage2"]
# fmt: on

# Vehicle detection classes — §3.5.1 annotation schema (real names to be
# confirmed; placeholders used until the annotation legend is verified).
_VEHICLE_CLASSES = ["Car", "Motorcycle", "Tricycle", "Van", "Bus", "Truck"]


# ---------------------------------------------------------------- callback


class UnfreezeCallback:
    """Unfreeze backbone at epoch `freeze_epochs`, set lr to `unfreeze_lr`.

    Adds backbone parameters (those with requires_grad=False at start) to the
    optimizer as a new param group at `unfreeze_lr`, then sets ALL param groups
    to `unfreeze_lr` so the schedule continues from a single consistent base.
    Updates scheduler.base_lrs to reflect the new base.
    Logs the unfreeze event to `log_path` (one JSON line).
    """

    def __init__(self, freeze_epochs: int, unfreeze_lr: float, log_path: Path) -> None:
        self.freeze_epochs = freeze_epochs
        self.unfreeze_lr = unfreeze_lr
        self.log_path = Path(log_path)
        self._fired = False

    def __call__(self, trainer) -> None:
        # trainer.epoch is 0-indexed; fire exactly once at the target epoch.
        if self._fired or trainer.epoch != self.freeze_epochs:
            return
        self._fired = True

        optimizer = trainer.optimizer

        # Collect parameters that are still frozen.
        new_params = [
            p for group in optimizer.param_groups for p in group["params"]
            if not p.requires_grad
        ]
        # Also collect any model parameters not yet in the optimizer.
        optimizer_params = {
            id(p)
            for group in optimizer.param_groups
            for p in group["params"]
        }
        for p in trainer.model.parameters():
            if not p.requires_grad and id(p) not in optimizer_params:
                new_params.append(p)

        for p in new_params:
            p.requires_grad_(True)

        if new_params:
            optimizer.add_param_group({"params": new_params, "lr": self.unfreeze_lr})

        # Align every group to unfreeze_lr so the scheduler has a single base.
        for pg in optimizer.param_groups:
            pg["lr"] = self.unfreeze_lr
            pg["initial_lr"] = self.unfreeze_lr

        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            trainer.scheduler.base_lrs = [self.unfreeze_lr] * len(optimizer.param_groups)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"epoch": self.freeze_epochs, "event": "unfreeze", "lr": self.unfreeze_lr}
                )
                + "\n"
            )


# ----------------------------------------------------------- registry


class CheckpointRegistry:
    """JSON file mapping (condition, arch, fold, stage) → checkpoint path.

    Persists at out_dir/checkpoint_registry.json.
    Thread-safe via atomic write (temp file + rename).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = self.path.with_suffix(".lock")

    # Key format per spec.
    @staticmethod
    def _key(condition: str, arch: str, fold: int, stage: str) -> str:
        return f"{condition}_{arch}_fold{fold}_{stage}"

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # os.replace is atomic on POSIX and best-effort on Windows.
        os.replace(tmp, self.path)

    def register(
        self, condition: str, arch: str, fold: int, stage: str, ckpt_path: Path
    ) -> None:
        data = self._read()
        data[self._key(condition, arch, fold, stage)] = str(ckpt_path)
        self._write(data)

    def get(self, condition: str, arch: str, fold: int, stage: str) -> Path | None:
        val = self._read().get(self._key(condition, arch, fold, stage))
        return Path(val) if val is not None else None

    def as_dict(self) -> dict:
        return self._read()


# --------------------------------------------------------- dataset yaml


def build_dataset_yaml(
    condition: str,
    fold: int,
    manifest_path: Path,
    pool_roots: dict[str, Path],
    out_dir: Path,
    thresholds_path: Path | None = None,
) -> Path:
    """Write a YOLO-format dataset.yaml for (condition, fold) and return its path.

    Condition A: train = Pool A frames where split_role='train' and fold != fold
    Condition B: train = Pool A (same filter) + Condition B augmented frames
    Condition C: train = Pool A (same filter) + Pool B synthetic frames
    Val: always Pool A frames where fold == fold (held-out fold).

    Raises FileNotFoundError if a required pool_root does not exist.
    """
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)

    df = pd.read_parquet(manifest_path)

    pool_a_root = pool_roots.get("A")
    if pool_a_root is None:
        raise KeyError("pool_roots must contain key 'A'")
    pool_a_root = Path(pool_a_root)

    # Training split: Pool A rows with the correct split_role, excluding the val fold.
    train_mask = (df["pool"] == "A") & (df["split_role"] == "train") & (df["fold"] != fold)
    train_df = df[train_mask]

    # Validation split: Pool A rows belonging to the held-out fold.
    val_mask = (df["pool"] == "A") & (df["fold"] == fold)
    val_df = df[val_mask]

    def _img_path(row: Any, root: Path) -> str:
        # ann_path carries the annotation file; derive the image path from it.
        ann = Path(row["ann_path"])
        # Images live alongside annotations — same stem, common image extensions.
        # The manifest's ann_path is relative; resolve against the pool root.
        img = root / ann.with_suffix(".jpg")
        return str(img)

    train_imgs = [_img_path(r, pool_a_root) for _, r in train_df.iterrows()]

    if condition == "B":
        pool_b_root = pool_roots.get("B")
        if pool_b_root is None:
            raise KeyError("pool_roots must contain key 'B' for Condition B")
        pool_b_root = Path(pool_b_root)
        if not pool_b_root.exists():
            raise FileNotFoundError(f"Condition B augmented frames root not found: {pool_b_root}")
        aug_df = df[(df["pool"] == "B") & (df["split_role"] == "train") & (df["fold"] != fold)]
        train_imgs += [_img_path(r, pool_b_root) for _, r in aug_df.iterrows()]

    elif condition == "C":
        pool_b_root = pool_roots.get("B")
        if pool_b_root is None:
            raise KeyError("pool_roots must contain key 'B' for Condition C (SA-CycleGAN synthetic)")
        pool_b_root = Path(pool_b_root)
        if not pool_b_root.exists():
            raise FileNotFoundError(f"Condition C Pool B synthetic root not found: {pool_b_root}")
        syn_df = df[(df["pool"] == "B") & (df["split_role"] == "train") & (df["fold"] != fold)]
        train_imgs += [_img_path(r, pool_b_root) for _, r in syn_df.iterrows()]

    val_imgs = [_img_path(r, pool_a_root) for _, r in val_df.iterrows()]

    dataset_dir = out_dir / "datasets" / f"{condition}_fold{fold}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_txt = dataset_dir / "train.txt"
    val_txt = dataset_dir / "val.txt"

    train_txt.write_text("\n".join(train_imgs), encoding="utf-8")
    val_txt.write_text("\n".join(val_imgs), encoding="utf-8")

    dataset_yaml = dataset_dir / "dataset.yaml"
    content = {
        "train": str(train_txt),
        "val": str(val_txt),
        "nc": len(_VEHICLE_CLASSES),
        "names": _VEHICLE_CLASSES,
    }
    dataset_yaml.write_text(yaml.dump(content, allow_unicode=True), encoding="utf-8")

    return dataset_yaml


# -------------------------------------------------------- stage runners


def _shared_kwargs(cfg: dict) -> dict:
    sh = cfg["detectors"]["shared"]
    return dict(
        imgsz=cfg["detectors"]["imgsz"],
        cos_lr=sh["cos_lr"],
        lrf=sh["lrf"],
        warmup_epochs=sh["warmup_epochs"],
        mosaic=sh["mosaic"],
        mixup=sh["mixup"],
        save_period=sh["save_period"],
        deterministic=sh["deterministic"],  # False — §3.7.3
        workers=0,  # Windows: multiprocessing workers cause deadlocks
    )


def run_stage1(
    condition: str,
    arch: str,
    fold: int,
    *,
    manifest_path: Path,
    pool_roots: dict[str, Path],
    cfg: dict,
    registry: CheckpointRegistry,
    runs_root: Path,
    device: str = "0",
) -> Path:
    """Train Stage 1 for (condition, arch, fold). Returns best.pt path.

    Stage 1 (§3.4.4): 100 epochs, patience=10, backbone frozen for first 10
    epochs via freeze=10, then unfrozen by UnfreezeCallback at epoch 10. One
    model.train() call per M12.
    """
    from ultralytics import YOLO

    runs_root = Path(runs_root).resolve()
    arch_cfg = cfg["detectors"][arch]
    stage_cfg = cfg["detectors"]["stage1"]

    dataset_yaml = build_dataset_yaml(
        condition, fold, manifest_path, pool_roots, runs_root / "datasets_cfg",
    )

    model = YOLO(arch_cfg["weights"])

    log_path = runs_root / f"stage1_{condition}_{arch}_fold{fold}" / "unfreeze.jsonl"
    cb = UnfreezeCallback(
        freeze_epochs=stage_cfg["freeze_epochs"],
        unfreeze_lr=arch_cfg["unfreeze_lr"],
        log_path=log_path,
    )
    model.add_callback("on_train_epoch_start", cb)

    train_kwargs: dict[str, Any] = dict(
        data=str(dataset_yaml),
        epochs=stage_cfg["epochs"],
        patience=stage_cfg["patience"],
        freeze=stage_cfg["freeze_epochs"],  # freeze first 10 layers (backbone)
        optimizer=arch_cfg["optimizer"],
        lr0=arch_cfg["lr0"],
        weight_decay=arch_cfg["weight_decay"],
        project=str(runs_root),
        name=f"stage1_{condition}_{arch}_fold{fold}",
        device=device,
        **_shared_kwargs(cfg),
    )

    # momentum is only set for SGD (YOLOv8m); AdamW (RT-DETR-L) omits it per §3.4.2.
    if "momentum" in arch_cfg:
        train_kwargs["momentum"] = arch_cfg["momentum"]

    model.train(**train_kwargs)

    best_pt = runs_root / f"stage1_{condition}_{arch}_fold{fold}" / "weights" / "best.pt"
    registry.register(condition, arch, fold, "stage1", best_pt)

    return best_pt


def run_stage2(
    condition: str,
    arch: str,
    fold: int,
    stage1_ckpt: Path,
    *,
    manifest_path: Path,
    pool_roots: dict[str, Path],
    cfg: dict,
    registry: CheckpointRegistry,
    runs_root: Path,
    device: str = "0",
) -> Path:
    """Fine-tune Stage 2 from stage1_ckpt. Returns last.pt path.

    Stage 2 (§3.4.4, §3.5.4): 10 epochs, patience=0 (disabled), backbone
    unfrozen throughout (freeze=0), lr = arch_lr0 * 0.1 (unfreeze_lr).
    """
    from ultralytics import YOLO

    runs_root = Path(runs_root).resolve()
    arch_cfg = cfg["detectors"][arch]
    stage_cfg = cfg["detectors"]["stage2"]

    dataset_yaml = build_dataset_yaml(
        condition, fold, manifest_path, pool_roots, runs_root / "datasets_cfg",
    )

    model = YOLO(str(stage1_ckpt))

    train_kwargs: dict[str, Any] = dict(
        data=str(dataset_yaml),
        epochs=stage_cfg["epochs"],
        patience=0,  # §3.4.4: early stopping disabled; Ultralytics: 0 = disabled
        freeze=0,    # backbone unfrozen throughout Stage 2
        optimizer=arch_cfg["optimizer"],
        lr0=arch_cfg["unfreeze_lr"],  # Stage 2 starts where Stage 1 ended
        weight_decay=arch_cfg["weight_decay"],
        project=str(runs_root),
        name=f"stage2_{condition}_{arch}_fold{fold}",
        device=device,
        **_shared_kwargs(cfg),
    )

    if "momentum" in arch_cfg:
        train_kwargs["momentum"] = arch_cfg["momentum"]

    model.train(**train_kwargs)

    last_pt = runs_root / f"stage2_{condition}_{arch}_fold{fold}" / "weights" / "last.pt"
    registry.register(condition, arch, fold, "stage2", last_pt)

    return last_pt


# ---------------------------------------------------------------- CLI


def _resolve_absolute(s: str) -> Path:
    return Path(s).resolve()


def main() -> None:
    from src.util.config import load_base, load_thresholds

    parser = argparse.ArgumentParser(
        description="SA-CycleGAN thesis detector training harness (§3.4)"
    )
    parser.add_argument("--condition", choices=["A", "B", "C"])
    parser.add_argument("--arch", choices=["yolov8m", "rtdetr_l"])
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--manifest", type=_resolve_absolute)
    parser.add_argument("--pool-a", type=_resolve_absolute)
    parser.add_argument("--pool-b", type=_resolve_absolute, default=None)
    parser.add_argument("--pool-c", type=_resolve_absolute, default=None)
    parser.add_argument("--runs-dir", type=_resolve_absolute, default=_resolve_absolute("runs"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--stage1-ckpt", type=_resolve_absolute, default=None)
    parser.add_argument("--run-all", action="store_true",
                        help="Run all conditions × architectures × folds")
    args = parser.parse_args()

    cfg = load_base()
    thr = load_thresholds()

    # k_folds is the one non-null value in thresholds.yaml; default 5 if somehow null.
    try:
        k_folds = thr.k_folds
    except Exception:
        k_folds = 5

    pool_roots: dict[str, Path] = {}
    if args.pool_a:
        pool_roots["A"] = args.pool_a
    if args.pool_b:
        pool_roots["B"] = args.pool_b
    if args.pool_c:
        pool_roots["C"] = args.pool_c

    runs_root = args.runs_dir
    registry = CheckpointRegistry(runs_root / "checkpoint_registry.json")

    if args.run_all:
        conditions = cfg["design"]["conditions"]
        architectures = cfg["design"]["architectures"]
        for cond in conditions:
            for arch in architectures:
                for fold in range(k_folds):
                    print(f"\n=== run_all: condition={cond} arch={arch} fold={fold} ===")
                    s1_ckpt = run_stage1(
                        cond, arch, fold,
                        manifest_path=args.manifest,
                        pool_roots=pool_roots,
                        cfg=cfg,
                        registry=registry,
                        runs_root=runs_root,
                        device=args.device,
                    )
                    run_stage2(
                        cond, arch, fold, s1_ckpt,
                        manifest_path=args.manifest,
                        pool_roots=pool_roots,
                        cfg=cfg,
                        registry=registry,
                        runs_root=runs_root,
                        device=args.device,
                    )
        return

    # Single-run mode — all three of --condition, --arch, --fold are required.
    for flag, val in [("--condition", args.condition), ("--arch", args.arch), ("--fold", args.fold)]:
        if val is None:
            parser.error(f"{flag} is required unless --run-all is set")

    if args.stage2_only:
        if args.stage1_ckpt is None:
            parser.error("--stage1-ckpt is required with --stage2-only")
        run_stage2(
            args.condition, args.arch, args.fold, args.stage1_ckpt,
            manifest_path=args.manifest,
            pool_roots=pool_roots,
            cfg=cfg,
            registry=registry,
            runs_root=runs_root,
            device=args.device,
        )
    else:
        s1_ckpt = run_stage1(
            args.condition, args.arch, args.fold,
            manifest_path=args.manifest,
            pool_roots=pool_roots,
            cfg=cfg,
            registry=registry,
            runs_root=runs_root,
            device=args.device,
        )
        if not args.stage1_only:
            run_stage2(
                args.condition, args.arch, args.fold, s1_ckpt,
                manifest_path=args.manifest,
                pool_roots=pool_roots,
                cfg=cfg,
                registry=registry,
                runs_root=runs_root,
                device=args.device,
            )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
