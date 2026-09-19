"""Model complexity for Table 3.11 — parameters and GFLOPs. Section 3.6.3.

Section 3.6.3 specifies: thop, applied to the UNFUSED model, at 640x640,
batch size 1, and states these are architecture properties independent of the
executing device. So this runs on CPU — no GPU needed, and the numbers are
reproducible on any machine.

Three things this script is careful about, each one a trap already hit:

1. BOTH architectures are measured by the same method. Reporting thop for one
   and Ultralytics' get_flops for the other puts two different measurements in
   one table.

2. The model is deepcopied before profiling. thop registers `total_ops` and
   `total_params` buffers on every module and attaches forward hooks; if those
   are left on a live model, the next forward pass dies with
   `AttributeError: 'ReLU' object has no attribute 'total_ops'`, and the extra
   buffers also corrupt the pretrained-weight transfer count.

3. Parameters are counted directly with sum(p.numel()). In Ultralytics
   8.4.154 `model.info()` returns None unless verbose=True.

Conventions, because the two differ by a factor of two:
  MACs   = what thop returns.
  GFLOPs = 2 x MACs / 1e9. This is the Ultralytics convention and is what the
           model summary line prints. Table 3.11 should use this, and say so.

Usage:  python scripts/model_complexity.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import thop
import ultralytics
from ultralytics import YOLO, RTDETR

ROOT = Path(__file__).resolve().parents[1]
IMGSZ = 640

# Expected unfused parameter counts, from the manuscript. Used as an assertion
# that we are profiling the unfused network: RT-DETR-L fused is 32,148,140, so
# if we silently got a fused model this check fails loudly instead of quietly
# reporting the wrong figure.
MODELS = [
    ("YOLOv8m",   YOLO,   "yolov8m.pt",  25_902_640),
    ("RT-DETR-L", RTDETR, "rtdetr-l.pt", 32_970_476),
]


def measure(loader, weights: str, imgsz: int = IMGSZ):
    """Return (params, macs, gflops) for the unfused model at batch size 1."""
    wrapper = loader(weights)
    net = wrapper.model                      # the raw nn.Module, not yet fused
    net = net.to("cpu").eval()

    params = sum(p.numel() for p in net.parameters())

    # Profile a COPY so no thop buffers or hooks survive on anything we keep.
    probe = copy.deepcopy(net)
    im = torch.zeros(1, 3, imgsz, imgsz)
    macs, _thop_params = thop.profile(probe, inputs=[im], verbose=False)
    del probe

    return params, float(macs), 2.0 * float(macs) / 1e9


def measure_fused(loader, weights: str, imgsz: int = IMGSZ):
    """Same, on the fused model. Reported separately, never mixed into the
    unfused column — Section 3.6.3 requires stating which is which."""
    net = loader(weights).model.to("cpu").eval()
    net = net.fuse() if hasattr(net, "fuse") else net
    params = sum(p.numel() for p in net.parameters())
    probe = copy.deepcopy(net)
    macs, _ = thop.profile(probe, inputs=[torch.zeros(1, 3, imgsz, imgsz)], verbose=False)
    del probe
    return params, float(macs), 2.0 * float(macs) / 1e9


def main() -> int:
    print(f"torch       {torch.__version__}")
    print(f"ultralytics {ultralytics.__version__}")
    print(f"thop        {thop.__version__}  ({getattr(thop, '__file__', '?')})")
    print(f"imgsz       {IMGSZ}, batch 1, CPU, unfused\n")

    if not thop.__version__.startswith("2."):
        print("WARNING: thop 2.x (ultralytics-thop) expected. A 0.1.x version is "
              "the original package and gives different figures.\n", file=sys.stderr)

    rows, ok = [], True
    for name, loader, weights, expect_params in MODELS:
        params, macs, gflops = measure(loader, weights)
        f_params, _f_macs, f_gflops = measure_fused(loader, weights)

        match = params == expect_params
        ok &= match
        flag = "" if match else f"  <-- MISMATCH, manuscript says {expect_params:,}"
        print(f"{name}")
        print(f"  parameters (unfused) : {params:,}{flag}")
        print(f"  MACs                 : {macs/1e9:.4f} G")
        print(f"  GFLOPs (2 x MACs)    : {gflops:.1f}      <-- Table 3.11")
        print(f"  parameters (fused)   : {f_params:,}")
        print(f"  GFLOPs (fused)       : {f_gflops:.1f}      (report separately, if at all)")
        print()
        rows.append({"model": name, "params_unfused": params, "macs_g": macs / 1e9,
                     "gflops_unfused": round(gflops, 1), "params_fused": f_params,
                     "gflops_fused": round(f_gflops, 1)})

    y, r = rows[0], rows[1]
    print("Ratios for Section 3.4.2 (RT-DETR-L relative to YOLOv8m), unfused:")
    print(f"  parameters : {r['params_unfused']/y['params_unfused']:.3f}")
    print(f"  GFLOPs     : {r['gflops_unfused']/y['gflops_unfused']:.3f}")
    print("  Manuscript line 547 currently states 1.27 and 1.37.\n")

    out = ROOT / "runs" / "model_complexity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"imgsz": IMGSZ, "batch": 1, "device": "cpu", "fused": False,
         "torch": torch.__version__, "ultralytics": ultralytics.__version__,
         "thop": thop.__version__, "models": rows}, indent=2), encoding="utf-8")
    print(f"written: {out}")

    if not ok:
        print("\nParameter counts differ from the manuscript. Investigate before "
              "updating Table 3.11 — a changed parameter count is a different "
              "model, not a different profiler.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console is cp932
    raise SystemExit(main())
