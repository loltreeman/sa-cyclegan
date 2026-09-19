# Project state — implementation

Updated 2026-09-19. `CLAUDE.md` at the repo root carries the durable rules and
traps and is loaded automatically; this file is the longer state record. Read
it when a task touches methodology, scheduling, or what is and isn't done.

---

## 1 · Manuscript state

**MANUSCRIPT 5** is the current version. Frozen, but twelve edits are open —
all specified with exact find/replace in `docs/fix-sheet-manuscript-5.md`.

Decided as of 2026-09-19:

| | |
|---|---|
| M1 | §3.5.1 switches to ⟦t1⟧/⟦t2⟧ at lines 632, 635, 636. **Blocks the `lum_band` column and therefore Step 10.** |
| M3 | RTX 5060 Ti exists, access uncertain. Keep the row; change its Role to overflow capacity so the design does not depend on it. |
| M4 | RX 9060 XT is a friend's machine. **Remove it**, plus the ROCm equivalence-check promise and §3.7.3's hardware-conditional paragraph. |
| M5 | 2,000 frames per generator target pool, 1,000 minimum. |
| M12 | Freeze-then-unfreeze via a `on_train_epoch_start` **callback**, not two `.train()` calls, so there is one warmup and one decay ramp as §3.4.3 describes. |
| B3 | λ_cyc = 10 and λ_id = 5, read from the §3.3.5 line 497 equation images. Settled. |

131 placeholders remain and **every one is measurement-gated** — CSMO footage,
the µY histogram, the IR fraction, or the timing pilot. 46 are blank `[[ ]]`
Chapter IV result cells and are supposed to be empty. Nothing left is a
decision being avoided.

`check_manuscript.py` (11 checks) is the verifier. It is a loose local script,
not in this repo — locate it and commit it.

## 2 · Environment

Windows, `.venv-cuda`, Python 3.12.10. Verified 2026-09-19:

```
torch 2.11.0+cu128    CUDA True    RTX 4050 Laptop GPU 6 GB
ultralytics 8.4.154   albumentations 2.0.8   ultralytics-thop 2.1.6
```

RT-DETR-L smoke test passed: one epoch on coco8, batch 4, imgsz 640,
**3.4 GB peak**, 941/941 pretrained items transferred.

### Compute, as it actually stands

| Device | Reality |
|---|---|
| RTX 4050 Laptop 6 GB | The machine work happens on. Continuous. |
| Lab RTX 3060 12 GB | 7 h/day by reservation. §1.4 fixes it as the **sole** latency-benchmark device. |
| Lab RTX 5060 Ti 16 GB | Exists; active access unconfirmed. Treat as overflow. |
| RX 9060 XT 16 GB | Friend's machine, never tested, friend may not be able to operate it. Being removed from §3.7.1. |

Three §3.7.1 claims now rest on the timing pilot rather than on measurement:
that 256×256 batch 1 fits "every device listed above" (which must now name a
6 GB card), that distributing across devices is necessary, and that the
checkpoint intervals bound loss to ~2 h and <1 h.

## 3 · What is built

```
configs/base.yaml         every §3.3/§3.4 hyperparameter, each citing its section
configs/thresholds.yaml   pending measurements as nulls
src/util/config.py        loader that RAISES PendingMeasurement
src/stats/nadeau_bengio.py
src/stats/holm.py
tests/test_stats.py       16 tests, all passing
scripts/model_complexity.py
```

`src/data/`, `src/gan/` and `src/detect/` are empty.

### Verified statistics values — regression tests, not guesses

| | |
|---|---|
| NB variance factor, k=5 | `1/5 + 1/4 = 0.45` (naive 0.20) |
| SE ratio | 1.5 |
| Detectable d_z, α=.05 | **1.86** (power there is ~0.57, not 0.50) |
| d_z for 80% power, α=.05 | **2.40** |
| Comparison family | **70** = 5 × 7 bins × 2 checkpoints |
| Detectable d_z, α=.05/70 | **6.30** |
| d_z for 80% power, α=.05/70 | **7.8** |
| Wilcoxon n=5 two-sided floor | **0.0625** — cannot reach α=.05 |

## 4 · Open items

### Time-critical

**The CSMO footage request.** `claude/csmo-footage-request.md` in the claude.ai
project, written 2026-09-07, notes that Sept–Nov 2025 footage is closest to
rolling off under ~12-month retention. That window is expiring now. Two blanks
remain before sending: the camera IDs used by the prior ALIVE group (source and
target domains must share viewpoints, or the generator learns the camera rather
than the weather), and the PAGASA candidate-date list.

Size the ask realistically. At §3.5.5's fixed 1-frame-per-15-seconds and a
2,000-frame target, **each Pool B subset needs ~8.3 hours of footage, ~33 hours
across the four.** §3.2.1 already carries the fallback if a subset comes up
short.

### Also open

- **Table 3.11 GFLOPs.** The recorded 108.3 unfused / 110.2 fused for RT-DETR-L
  came from the broken thop install and the labels are reversed. Under
  ultralytics-thop 2.1.6 the summary reports 110.2 unfused and 105.7 fused.
  Re-measure **both** architectures with `scripts/model_complexity.py` and
  update Table 3.11 and the ratios at line 547 (currently 1.27× and 1.37×).
  Note the script prints MACs and GFLOPs separately: GFLOPs = 2 × MACs, which
  is the Ultralytics convention. §3.6.3 calls it "multiply–accumulate
  complexity… in GFLOPs", which is loose by a factor of two — fix the wording.
- **The null-result email.** Never sent. Frame it as a *power* question: the
  confirmatory comparison detects only d_z ≥ 1.86 and has 80% power only at
  2.40, so a non-significant result is likely even if SA-CycleGAN helps
  moderately. Ask whether the thesis passes on the comparative analysis alone.
- **RT-DETR-L latency.** A throwaway figure from the smoke test showed 64.6 ms
  per image on the 4050 — roughly 2× the 33.3 ms budget, though measured on 4
  images with no warmup and not under §3.6.3's protocol. §3.6.3 line 815
  anticipates both architectures clearing the threshold. The pilot settles it.

## 5 · Next steps

1. **Timing pilot (Step 29).** YOLOv8m and RT-DETR-L, one epoch each on ~500
   ALIVE frames at 640. Record sec/epoch and peak memory. Produces the run
   schedule and substantiates §3.7.1's checkpoint-interval claim, which is
   currently written as though measured.
2. **GAN memory probe.** `src/gan/` is empty — there is no CycleGAN to time.
   Instantiate two ResNet-9 generators, two spectral-norm PatchGANs and a
   frozen VGG-19, run ~20 optimizer steps at 256×256 batch 1, read peak memory.
   Answers "does it fit in 6 GB" weeks before the real generator exists.
3. **Step 10, the manifest schema.** All three authors, ~1 hour, needs no CSMO
   footage — define it and test against Pool A plus synthetic rows. Freeze
   before Phase 2 or 3 starts; it is the interface all three code against.
4. Split: D → Phase 2 (steps 10–17), G → Phase 3 (18–26), E → Phase 4 (27–30).

The full 43-step plan is `claude/build-plan-step-by-step.md` in the claude.ai
project, along with the per-track handoffs `handoff-track-D / E / G.md`.

## 6 · Work division

Three owners. Phases 2 and 3 run in parallel. ~15 person-days, 2–4 weeks
calendar. See `claude/work-division-three-owners.md`.
