"""Bounding-box transfer validity audit (§3.6.1, lines 744–750).

Three direct measurements per generator:

1. Localisation audit   — class-agnostic IoU between transferred bbox and the
                           nearest vehicle prediction from an auditing detector.
2. Edge-preservation    — Canny edge IoU inside transferred bbox interiors,
                           comparing source and translated frames.
3. Manual validity CSV  — 200 frames stratified by class, rendered with bboxes
                           overlaid, written for human scoring and Fleiss' kappa.

Controls for the localisation audit (§3.6.1 §3 paragraph):
  - clear-weather source frames vs their existing annotations  (detector baseline)
  - real annotated adverse-weather frames at matched severity  (weather degradation)

Agreement between the synthetic audit and the real-weather control indicates
that transferred annotations behave comparably to authentic annotations under
authentic weather (§3.6.1 line 746).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

__all__ = [
    "load_yolo_boxes",
    "run_auditing_detector",
    "iou",
    "localisation_audit",
    "edge_preservation",
    "render_audit_sample",
]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# YOLO label helpers
# ---------------------------------------------------------------------------

def load_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Read YOLO .txt, return list of {cls, x1, y1, x2, y2} in pixel coords."""
    boxes: list[dict] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append({"cls": cls_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def _image_stem_to_label(img_path: Path, label_dir: Path) -> Path:
    return label_dir / (img_path.stem + ".txt")


# ---------------------------------------------------------------------------
# Auditing detector (class-agnostic)
# ---------------------------------------------------------------------------

def run_auditing_detector(
    img_bgr: np.ndarray,
    weights: str = "yolov8n.pt",
    conf: float = 0.25,
    *,
    device: str = "cuda",
) -> list[dict]:
    """Run an external YOLOv8 detector class-agnostically.

    Returns list of {x1, y1, x2, y2, conf} in pixel coords.
    The detector is used only to verify spatial localisation, not class labels.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. pip install ultralytics") from exc

    model = YOLO(weights)
    results = model.predict(img_bgr, conf=conf, device=device, verbose=False)[0]
    preds: list[dict] = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        preds.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": float(box.conf[0].cpu()),
        })
    return preds


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def iou(a: dict, b: dict) -> float:
    """Compute IoU between two {x1, y1, x2, y2} dicts."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _best_iou(gt_box: dict, preds: list[dict]) -> float:
    """Return highest IoU of gt_box against any prediction."""
    if not preds:
        return 0.0
    return max(iou(gt_box, p) for p in preds)


# ---------------------------------------------------------------------------
# Localisation audit (§3.6.1 line 746)
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Car", "Motorcycle", "Tricycle", "Van", "Bus", "Truck"]


def localisation_audit(
    image_paths: list[Path],
    label_dir: Path,
    *,
    detector_weights: str = "yolov8n.pt",
    device: str = "cuda",
    conf: float = 0.25,
) -> pd.DataFrame:
    """Compute per-box IoU between transferred gt bbox and nearest prediction.

    Returns DataFrame with columns:
        frame, cls_id, cls_name, gt_iou, above_half
    """
    rows: list[dict] = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        label_path = _image_stem_to_label(img_path, label_dir)
        gt_boxes = load_yolo_boxes(label_path, w, h)
        if not gt_boxes:
            continue

        preds = run_auditing_detector(img_bgr, weights=detector_weights,
                                      conf=conf, device=device)
        for gt in gt_boxes:
            best = _best_iou(gt, preds)
            cls_name = CLASS_NAMES[gt["cls"]] if gt["cls"] < len(CLASS_NAMES) else str(gt["cls"])
            rows.append({
                "frame":    img_path.name,
                "cls_id":   gt["cls"],
                "cls_name": cls_name,
                "gt_iou":   best,
                "above_half": best >= 0.5,
            })
    return pd.DataFrame(rows)


def _summary_stats(df: pd.DataFrame) -> dict:
    iou_col = df["gt_iou"]
    return {
        "n_boxes":          len(df),
        "median_iou":       float(iou_col.median()),
        "q25_iou":          float(iou_col.quantile(0.25)),
        "q75_iou":          float(iou_col.quantile(0.75)),
        "pct_below_0.5":    float((~df["above_half"]).mean() * 100),
        "mean_iou":         float(iou_col.mean()),
    }


# ---------------------------------------------------------------------------
# Edge-preservation measure (§3.6.1 line 748)
# ---------------------------------------------------------------------------

def _edge_iou_in_box(src: np.ndarray, trn: np.ndarray, box: dict) -> float | None:
    """Canny edge IoU inside the bbox interior.

    Both src and trn are uint8 BGR images.
    Returns None if the box is empty or degenerate.
    """
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(src.shape[1], x2), min(src.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None

    src_crop = cv2.cvtColor(src[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    trn_crop = cv2.cvtColor(trn[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    # Adaptive thresholds based on median intensity (Canny auto-threshold heuristic).
    def _canny(gray: np.ndarray) -> np.ndarray:
        med = int(np.median(gray))
        lo, hi = max(0, int(0.66 * med)), min(255, int(1.33 * med))
        return cv2.Canny(gray, lo, hi)

    src_edges = _canny(src_crop).astype(bool)
    trn_edges = _canny(trn_crop).astype(bool)

    inter = (src_edges & trn_edges).sum()
    union = (src_edges | trn_edges).sum()
    return float(inter / union) if union > 0 else None


def edge_preservation(
    source_paths: list[Path],
    translated_paths: list[Path],
    label_dir: Path,
) -> pd.DataFrame:
    """Compute Canny edge IoU per box for each source→translated pair.

    source_paths and translated_paths must be paired by index (same filename
    convention) and reference the same set of YOLO labels in label_dir.

    Returns DataFrame with columns:
        frame, cls_id, cls_name, edge_iou
    """
    src_by_name = {p.name: p for p in source_paths}
    trn_by_name = {p.name: p for p in translated_paths}
    common = sorted(src_by_name.keys() & trn_by_name.keys())

    rows: list[dict] = []
    for name in common:
        src_bgr = cv2.imread(str(src_by_name[name]))
        trn_bgr = cv2.imread(str(trn_by_name[name]))
        if src_bgr is None or trn_bgr is None:
            continue
        h, w = src_bgr.shape[:2]
        label_path = label_dir / (Path(name).stem + ".txt")
        gt_boxes = load_yolo_boxes(label_path, w, h)
        for gt in gt_boxes:
            eiou = _edge_iou_in_box(src_bgr, trn_bgr, gt)
            if eiou is None:
                continue
            cls_name = CLASS_NAMES[gt["cls"]] if gt["cls"] < len(CLASS_NAMES) else str(gt["cls"])
            rows.append({
                "frame":    name,
                "cls_id":  gt["cls"],
                "cls_name": cls_name,
                "edge_iou": eiou,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Manual validity audit — render 200 stratified frames (§3.6.1 line 750)
# ---------------------------------------------------------------------------

def render_audit_sample(
    image_paths: list[Path],
    label_dir: Path,
    out_dir: Path,
    n_sample: int = 200,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Write n_sample frames with bbox overlays for manual three-author scoring.

    Stratifies by class: each class contributes floor(n_sample/n_classes) frames,
    remainder distributed to classes with most instances. Returns a DataFrame
    with columns frame, cls_id, cls_name, out_path, valid (empty, to be filled
    by raters) for each rendered image — one row per frame.
    """
    rng = np.random.default_rng(seed)

    # Collect (img_path, cls_ids) pairs.
    frame_cls: list[tuple[Path, list[int]]] = []
    for img_path in image_paths:
        h_dummy, w_dummy = 1, 1
        tmp = cv2.imread(str(img_path))
        if tmp is None:
            continue
        h_dummy, w_dummy = tmp.shape[:2]
        label_path = _image_stem_to_label(img_path, label_dir)
        boxes = load_yolo_boxes(label_path, w_dummy, h_dummy)
        cls_ids = [b["cls"] for b in boxes]
        if cls_ids:
            frame_cls.append((img_path, cls_ids))

    # Build per-class index.
    n_classes = len(CLASS_NAMES)
    per_class: list[list[int]] = [[] for _ in range(n_classes)]
    for idx, (_, cls_ids) in enumerate(frame_cls):
        for c in set(cls_ids):
            if c < n_classes:
                per_class[c].append(idx)

    # Allocate quota per class.
    base_quota, remainder = divmod(n_sample, n_classes)
    class_sizes = [len(pc) for pc in per_class]
    quotas = [base_quota] * n_classes
    sorted_by_size = sorted(range(n_classes), key=lambda c: class_sizes[c], reverse=True)
    for i in range(remainder):
        quotas[sorted_by_size[i]] += 1

    selected: set[int] = set()
    for c, q in enumerate(quotas):
        candidates = [i for i in per_class[c] if i not in selected]
        if not candidates:
            continue
        k = min(q, len(candidates))
        chosen = rng.choice(candidates, k, replace=False).tolist()
        selected.update(chosen)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for idx in sorted(selected):
        img_path, _ = frame_cls[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        label_path = _image_stem_to_label(img_path, label_dir)
        boxes = load_yolo_boxes(label_path, w, h)

        vis = img_bgr.copy()
        for b in boxes:
            cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
            cls_name = CLASS_NAMES[b["cls"]] if b["cls"] < len(CLASS_NAMES) else str(b["cls"])
            cv2.putText(vis, cls_name, (b["x1"], max(b["y1"] - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), vis)
        rows.append({
            "frame":    img_path.name,
            "out_path": str(out_path),
            "valid":    "",  # to be filled by each rater
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fleiss' kappa (for 3-rater manual audit)
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings: pd.DataFrame) -> float:
    """Compute Fleiss' kappa for binary ratings.

    ratings: DataFrame with one column per rater, values 0/1 (invalid/valid).
    Returns kappa in [-1, 1].
    """
    n, k = ratings.shape   # items × raters
    n_cat = 2  # binary

    # Proportion of ratings per category per item.
    counts = np.zeros((n, n_cat), dtype=float)
    arr = ratings.values
    for i in range(n):
        for j in range(k):
            counts[i, int(arr[i, j])] += 1

    p_j = counts.sum(axis=0) / (n * k)   # marginal proportions
    P_bar = ((counts ** 2).sum(axis=1) - k) / (k * (k - 1))  # per-item agreement
    P_bar_mean = P_bar.mean()
    P_e = (p_j ** 2).sum()
    if abs(1 - P_e) < 1e-12:
        return 1.0
    return float((P_bar_mean - P_e) / (1 - P_e))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bounding-box transfer audit (§3.6.1, lines 744–750)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- localisation -------------------------------------------------------
    loc = sub.add_parser("localise", help="Localisation IoU audit")
    loc.add_argument("--images",  required=True, type=Path, help="Directory of synthetic frames")
    loc.add_argument("--labels",  required=True, type=Path, help="Directory of YOLO .txt labels")
    loc.add_argument("--out",     required=True, type=Path, help="Output CSV path")
    loc.add_argument("--weights", default="yolov8n.pt",     help="Auditing detector weights")
    loc.add_argument("--device",  default="cuda")
    loc.add_argument("--conf",    default=0.25, type=float)

    # ---- edge ---------------------------------------------------------------
    edg = sub.add_parser("edge", help="Edge-preservation measure")
    edg.add_argument("--source",   required=True, type=Path, help="Source (clear) frame directory")
    edg.add_argument("--translated", required=True, type=Path, help="Translated frame directory")
    edg.add_argument("--labels",   required=True, type=Path, help="Directory of YOLO .txt labels")
    edg.add_argument("--out",      required=True, type=Path, help="Output CSV path")

    # ---- render (manual audit) ----------------------------------------------
    ren = sub.add_parser("render", help="Render frames for manual validity audit")
    ren.add_argument("--images",  required=True, type=Path)
    ren.add_argument("--labels",  required=True, type=Path)
    ren.add_argument("--out-dir", required=True, type=Path)
    ren.add_argument("--n",       default=200, type=int,
                     help="Number of frames to sample (default 200)")
    ren.add_argument("--seed",    default=42,  type=int)

    # ---- kappa --------------------------------------------------------------
    kap = sub.add_parser("kappa", help="Compute Fleiss kappa from completed rating CSV")
    kap.add_argument("--ratings", required=True, type=Path,
                     help="CSV with one rater column per author (binary 0/1)")
    kap.add_argument("--rater-cols", nargs="+", default=["rater1", "rater2", "rater3"])

    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()

    if args.cmd == "localise":
        exts = _IMG_EXTS
        img_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in exts)
        df = localisation_audit(img_paths, args.labels,
                                detector_weights=args.weights,
                                device=args.device, conf=args.conf)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)

        stats = _summary_stats(df)
        print(json.dumps(stats, indent=2))
        print(f"\nPer-class breakdown:")
        for cls_name, grp in df.groupby("cls_name"):
            s = _summary_stats(grp)
            print(f"  {cls_name:12s}  n={s['n_boxes']:4d}  "
                  f"median_iou={s['median_iou']:.3f}  "
                  f"pct_below_0.5={s['pct_below_0.5']:.1f}%")
        print(f"\nSaved → {args.out}")

    elif args.cmd == "edge":
        exts = _IMG_EXTS
        src_paths = sorted(p for p in args.source.iterdir() if p.suffix.lower() in exts)
        trn_paths = sorted(p for p in args.translated.iterdir() if p.suffix.lower() in exts)
        df = edge_preservation(src_paths, trn_paths, args.labels)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Edge IoU  median={df['edge_iou'].median():.4f}"
              f"  IQR=[{df['edge_iou'].quantile(0.25):.4f}, "
              f"{df['edge_iou'].quantile(0.75):.4f}]"
              f"  n={len(df)}")
        print(f"Saved → {args.out}")

    elif args.cmd == "render":
        exts = _IMG_EXTS
        img_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in exts)
        df = render_audit_sample(img_paths, args.labels, args.out_dir,
                                  n_sample=args.n, seed=args.seed)
        roster_path = args.out_dir / "audit_roster.csv"
        df.to_csv(roster_path, index=False)
        print(f"Rendered {len(df)} frames → {args.out_dir}")
        print(f"Roster (fill 'valid' column 0/1 per rater) → {roster_path}")

    elif args.cmd == "kappa":
        ratings = pd.read_csv(args.ratings)
        rater_data = ratings[args.rater_cols].astype(int)
        k = fleiss_kappa(rater_data)
        valid_pct = rater_data.mean().mean() * 100
        print(f"Fleiss kappa: {k:.4f}")
        print(f"Mean valid %: {valid_pct:.1f}")
        print(json.dumps({
            "fleiss_kappa": k,
            "mean_valid_pct": valid_pct,
            "n_frames": len(rater_data),
            "n_raters": len(args.rater_cols),
        }, indent=2))


if __name__ == "__main__":
    main()
