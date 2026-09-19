# Project state — implementation

Updated 2026-09-19. `CLAUDE.md` at the repo root carries the durable rules and
traps and is loaded automatically; this file is the longer state record. Read
it when a task touches methodology, scheduling, or what is and isn't done.

---

## 1 · Manuscript state

**MANUSCRIPT 6** is the current version. Fix-sheet edits M2–M12 applied
(M1 blocked on µY histogram; M2/M5/M7/M8/M9/M10 were already in v6).
Ratios at line 549 corrected to 1.273×/1.390×. Remaining open items below.

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

`scripts/check_manuscript.py` (7 checks + placeholder inventory) is the
verifier. Committed to repo 2026-09-19.

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
configs/base.yaml            every §3.3/§3.4 hyperparameter, each citing its section
configs/thresholds.yaml      pending measurements as nulls
src/util/config.py           loader that RAISES PendingMeasurement
src/stats/nadeau_bengio.py
src/stats/holm.py
src/data/manifest.py         manifest schema + Pool-A builder + validator (Step 10)
tests/test_stats.py          16 tests, all passing
tests/test_manifest.py       56 tests, all passing  (72 total)
scripts/model_complexity.py  Table 3.11 — YOLOv8m 79.3 / RT-DETR-L 110.2 GFLOPs
scripts/timing_pilot.py      §3.7.1 epoch-timing + peak-GPU-memory probe (Step 29)
scripts/gan_memory_probe.py  SA-CycleGAN 6-GB fit probe (no dataloader, ~20 steps)
scripts/check_manuscript.py  manuscript verifier (7 checks + placeholder inventory)
```

`src/gan/` and `src/detect/` are empty — generators and detectors not yet written.

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

- **Table 3.11 GFLOPs.** ✅ Corrected. YOLOv8m 79.3 / RT-DETR-L 110.2 GFLOPs
  unfused. Ratios updated to 1.273× params / 1.390× GFLOPs at manuscript line 549.
- **The null-result email.** Never sent. Frame it as a *power* question: the
  confirmatory comparison detects only d_z ≥ 1.86 and has 80% power only at
  2.40, so a non-significant result is likely even if SA-CycleGAN helps
  moderately. Ask whether the thesis passes on the comparative analysis alone.
- **RT-DETR-L latency.** A throwaway figure from the smoke test showed 64.6 ms
  per image on the 4050 — roughly 2× the 33.3 ms budget, though measured on 4
  images with no warmup and not under §3.6.3's protocol. §3.6.3 line 815
  anticipates both architectures clearing the threshold. The pilot settles it.

## 5 · Next steps

Steps 29 (timing pilot), GAN memory probe, and Step 10 (manifest) are all
done as of 2026-09-19. Next:

**Split by track — runs in parallel once CSMO footage arrives:**
- D → Phase 2 (steps 10–17): detector training harness, fold runner, eval
- G → Phase 3 (steps 18–26): SA-CycleGAN training loop, `src/gan/`
- E → Phase 4 (steps 27–30): latency benchmarking on lab RTX 3060

**Immediate unblocked work (no footage needed):**
- Run `scripts/timing_pilot.py` on the actual g29_demo.mp4 and fill in
  §3.7.1's checkpoint-interval claim with measured numbers.
- Run `scripts/gan_memory_probe.py` to confirm 6 GB fit, record result in
  `runs/gan_memory_probe.json`.
- Resolve the CSMO footage request (see §4 above) — it gates Phase 2 and 3.

The full 43-step plan is `claude/build-plan-step-by-step.md` in the claude.ai
project, along with the per-track handoffs `handoff-track-D / E / G.md`.

## 6 · Work division

Three owners. Phases 2 and 3 run in parallel. ~15 person-days, 2–4 weeks
calendar. See `claude/work-division-three-owners.md`.
