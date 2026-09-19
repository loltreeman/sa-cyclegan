# Fix sheet — MANUSCRIPT 5 and the Wave 0 scaffold

Date: 2026-09-19. Line numbers refer to
`CSCI 199.2 Biason - Jaso - Naguio - MANUSCRIPT 5.md` as uploaded.

Two parts. **Part A** is manuscript edits — you apply these by hand, exact
find/replace given. **Part B** is scaffold code — already applied, shipped as
`thesis-scaffold-patched.zip`, listed here so you can account for every change
in a defence.

Three edits (M3, M4, M5) need a decision from you first. They are marked
**DECISION** and both branches are written out.

---

# Part A — manuscript edits

## M1 — Luminance thresholds are hardcoded in §3.5.1 and pending in §3.2.4.2

**Severity: highest.** §3.2.4.2 says t1 and t2 are derived from the corpus's
own µY distribution and writes them as ⟦t1⟧/⟦t2⟧. §3.5.1 prints 50 and 100 as
settled numbers. `thresholds.yaml` has both null and `config.py` raises on
access. The code cannot be written both ways, and Step 10's `lum_band` column
is blocked until this resolves.

Keep §3.2.4.2 — its empirical-derivation argument is the stronger one, it is
already written, and importing cuts from another corpus is exactly what it
argues against. Change §3.5.1 in three places.

**Line 632** — find:

> Low-light severity is held at normal (µY ≥ 100\) during Phase 1\.

replace:

> Low-light severity is held at normal (µY ≥ ⟦t2⟧) during Phase 1, the threshold being as defined in Section 3.2.4.2\.

**Line 635** — find:

> across three luminance bins: severe (µY \< 50), moderate (50 ≤ µY \< 100), and normal (µY ≥ 100)

replace:

> across the three luminance bins defined in Section 3.2.4.2: severe (µY \< ⟦t1⟧), moderate (⟦t1⟧ ≤ µY \< ⟦t2⟧), and normal (µY ≥ ⟦t2⟧)

**Line 636** — find:

> combined with Severe Low-Light (µY \< 50)

replace:

> combined with Severe Low-Light (µY \< ⟦t1⟧)

This adds 4 placeholder instances to the count, all gated on the µY histogram
you were already waiting for. Nothing new is blocked.

---

## M2 — RT-DETR warmup contradicts the §3.4.3 table and the framework

Line 546 (§3.4.2) specifies an iteration-based warmup. The §3.4.3 table
(line 559) puts "3 warmup epochs" **above** the divider, i.e. equalized across
architectures. Verified against the Ultralytics configuration reference:
`warmup_epochs` defaults to 3.0 and its unit is epochs; no iteration-based
warmup setting is documented. Line 546 describes something the framework
cannot express without patching the trainer.

**Line 546** — find:

> The initial learning rate is 1×10⁻⁴, with a linear warmup applied over the first 1,000 training iterations to stabilize early Transformer weight updates.

replace:

> The initial learning rate is 1×10⁻⁴, with linear warmup applied over the first 3 epochs to stabilize early Transformer weight updates. The warmup length is equalized with YOLOv8m under the alignment rule of Section 3.4.3 and is the framework default for both architectures.

---

## M3 — **DECISION** — the RTX 5060 Ti

§3.7.1's compute table lists an "NVIDIA RTX 5060 Ti (ALIVE laboratory), 16 GB
GDDR7, shared, subject to adviser approval" and assigns it the generative
training runs. §3.7.2 (line 896) builds the CUDA 12.8 requirement around it.
Your implementation handoff does not mention this device at all. Tell me which
document is right; it decides where the six generator runs execute.

**If the 5060 Ti exists and you have a realistic path to it:** no edit. Add it
to the handoff's hardware table so the next context has it, and confirm with
the adviser before Phase 3 planning depends on it.

**If it does not exist:** delete this row from the §3.7.1 table —

> | NVIDIA RTX 5060 Ti (ALIVE laboratory) | 16 GB GDDR7 | Shared, subject to adviser approval | Generative training runs, where memory demand is highest |

and in §3.7.2 (line 896) find:

> GPU acceleration will be managed via CUDA 12.x. The RTX 5060 Ti (Blackwell architecture) requires CUDA 12.8 or newer; the primary RTX 3060 (Ampere architecture) is fully supported under CUDA 12.x. Project workloads accordingly target the CUDA 12.x toolchain to maintain a single, GPU-portable driver and runtime stack across the laboratory hardware listed in Section 3.7.1.

replace:

> GPU acceleration is managed via CUDA 12.x. Both NVIDIA devices in Section 3.7.1, the RTX 3060 (Ampere) and the RTX 4050 Laptop (Ada Lovelace), are fully supported under CUDA 12.x, and project workloads target that toolchain to maintain a single driver and runtime stack across them.

Then reassign generative training in the §3.7.1 table. §3.7.1 already states
the GAN trains at 256×256, batch 1, and argues this fits "every device listed
above, including the 12 GB RTX 3060" — so on that argument the 3060 and the
4050 can both carry generator runs, subject to the timing pilot.

---

## M4 — **DECISION** — the RX 9060 XT

§3.7.1 lists it as "personal workstation, Continuous". Your handoff says
friend's machine, access uncertain, never tested. §3.7.2 builds an entire
heterogeneous-vendor/ROCm argument on it, and §3.7.1 promises a functional
equivalence run whose "outcome is reported". If the machine never materialises
you will be asked, in the defence, for the outcome of a verification you could
not have run.

**If you will actually have it:** no edit, but run the equivalence check early
— the promise is in print.

**If you will not:** three passages go.

1. §3.7.1 table — delete the row:

   > | AMD RX 9060 XT (personal workstation) | 16 GB GDDR6  | Continuous | Generative and detector training |

2. §3.7.1 — delete the whole **Heterogeneous vendor support** paragraph, *except*
   the two sentences about `grid_sample`, which are load-bearing for §3.7.3's
   determinism claim. Keep:

   > Both architectures under study are implemented using framework-level PyTorch operations without custom compiled kernels. In particular, RT-DETR-L's multi-scale deformable attention is implemented through torch.nn.functional.grid\_sample rather than through the custom CUDA extension used by the original Deformable DETR implementation. Consequently, no vendor-specific kernel compilation is required for either architecture.

3. §3.7.1 — delete the paragraph beginning "Functional equivalence on the ROCm
   device is verified before production training…" entirely.

Also check §3.7.3 for any sentence conditioning determinism on the ROCm device;
if one exists it goes with these.

---

## M5 — **DECISION** — §3.3.7 target-pool numbers are inverted

Verbatim in §3.3.7:

> A target of 200 frames per pool is adopted, with 10000 frames treated as the minimum below which the risk of the generator memorising its target pool becomes material.

The stated minimum is 50× the stated target, which cannot be what you meant,
and §3.2.1 cites CycleGAN's ~1,000–6,000 images per domain. My reading is that
the digits were transposed and it should be 2,000 target / 1,000 minimum.
Confirm before I write it into Track G's config — this number sizes the CSMO
footage request.

Suggested replacement:

> A target of 2,000 frames per pool is adopted, with 1,000 frames treated as the minimum below which the risk of the generator memorising its target pool becomes material.

---

## M6 — §3.4.3 augmentation row contradicts itself

**Line 559** — find:

> | Augmentation pipeline | Mosaic, MixUp (held constant); MixUp and CutMix disabled | Mosaic, MixUp (held constant); MixUp and CutMix disabled |

replace:

> | Augmentation pipeline | Mosaic (held constant); MixUp and CutMix disabled | Mosaic (held constant); MixUp and CutMix disabled |

---

## M7 — one empty citation remains

MANUSCRIPT 5 closed two of the three the earlier audit found. One is left.

**Line 824** — find:

> given the result of Bengio and Grandvalet \[ \] no exact correction is available

replace:

> given the result of Bengio and Grandvalet \[55\] no exact correction is available

---

## M8 — five references still uncited

Confirmed by literal search: zero in-text occurrences of `\[2\]`, `\[3\]`,
`\[30\]`, `\[31\]`, `\[34\]`.

**Cite rather than remove.** The reference list is numbered and cited by number
throughout. Deleting entry 2 renumbers every later entry, which breaks all 14
occurrences of `\[27\]`, both `\[54\]`/`\[55\]` in §3.6.4, and everything
between. Adding five in-text citations is a five-line edit; removing one is a
whole-document renumber.

| Ref | Reflist line | Insert at | Section |
|---|---|---|---|
| 2 | 915 | 537 | §3.4.1 YOLOv8m Configuration |
| 3 | 916 | 537 | §3.4.1, same sentence as ref 2 |
| 30 | 943 | 407 | §3.2.6 Class Imbalance Handling |
| 31 | 944 | 764 | §3.6.1 per-image structural measures |
| 34 | 947 | 483 | §3.3.5 Feature consistency loss |

### Refs 2 and 3 — line 537, §3.4.1

Find (second sentence of the paragraph):

> The medium-scale variant (YOLOv8m) is selected as a balance between computational tractability on the ALIVE GPU and sufficient parameter capacity to learn adverse-weather feature distributions across six vehicle classes.

Append after it:

> Comparative evaluations of YOLOv8 against more recent releases of the family on vehicle detection \[2\], and its continued use as the base architecture in current real-time vehicle detection work \[3\], support its selection as a representative and still-current convolutional detector for this task.

**Check both papers actually support that sentence before pasting it.** I have
not read either; I am matching them by title. Geetha et al. \[2\] is a YOLOv8
vs YOLOv10 vehicle-detection comparison and Zhao et al. \[3\] builds a
real-time vehicle detector on YOLOv8, so the claim is the natural one, but
you are the one defending it.

### Ref 30 — line 407, §3.2.6

Weakest fit of the five. Few-shot detection is not a class-imbalance method,
so cite it as an acknowledged-but-not-adopted alternative rather than as
support. Append to the end of the "At the algorithmic level…" paragraph:

> The extreme-minority classes identified above, below 1% of instances, approach the low-shot regime for which dedicated fine-tuning procedures have been proposed \[30\]. Such procedures are not adopted here: mitigation is confined to the data-level oversampling and loss-level reweighting described above, and the resulting minority-class performance is reported rather than corrected for.

That last clause is worth having anyway — it pre-empts "why didn't you use a
few-shot method?" in the defence.

### Ref 31 — line 764, §3.6.1

Demšar is the standard citation for preferring non-parametric tests when
comparing classifiers, and line 764 already adopts exactly that. Find:

> Inferential comparison between configurations is conducted by the Wilcoxon signed-rank test over clip-level aggregates, the non-parametric paired form being adopted as these distributions are not expected to be symmetric.

replace:

> Inferential comparison between configurations is conducted by the Wilcoxon signed-rank test over clip-level aggregates, the non-parametric paired form being adopted as these distributions are not expected to be symmetric, following the recommendation of Demšar \[31\] for paired comparison of learning systems.

Best of the five — the argument is already there, it just lacks its citation.

### Ref 34 — line 483, §3.3.5

Gatys is the origin of the CNN-feature-space formulation Johnson \[35\] made
feed-forward, so it belongs beside it. Find:

> a feature consistency loss computed on intermediate activations of a VGG-19 network \[40\] pretrained on ImageNet, adopting the perceptual loss formulation of Johnson et al. \[35\].

replace:

> a feature consistency loss computed on intermediate activations of a VGG-19 network \[40\] pretrained on ImageNet, adopting the perceptual loss formulation of Johnson et al. \[35\], which builds on the CNN feature-space content representation introduced by Gatys et al. \[34\].

---

## M9 — Table 3.7 row 6 has an empty Status cell

**Line 657** — rows 1–5 read "Trained"; row 6 is blank. D4 puts the
encoder-only ablation in scope, so:

> | 6 | Encoder-only SA-CycleGAN | Encoder only | Rain — heavy | Attention-placement ablation | Trained |

---

## M10 — wrong table cross-reference

**Line 610** — the four production generators are enumerated in **Table 3.7**;
Table 3.6 is the Condition B/C allocation table. Find:

> enumerated in Table 3.6 of Section 3.5.3

replace:

> enumerated in Table 3.7 of Section 3.5.3

---

## M11 — the λ_fc sweep is promised in §3.3.5 and descoped in §3.7.1

§3.3.5: "a one-dimensional sensitivity sweep over {0.5, 1.0, 2.0} will be
conducted during pilot training." §3.7.1's **Scope reductions** placeholder
lists "the λ\_fc sensitivity sweep of Section 3.3.5" among what the compute
budget excludes. Both cannot stand.

The sweep is three short pilot runs on one severity, not three full production
generators — it is cheap, and §3.3.5 leans on it (it is what replaces per-layer
weight tuning, and it is the stated trigger for dropping relu1\_2). Keep the
sweep and strike it from the scope-reduction list. If you descope it instead,
§3.3.5's justification for static equal per-layer weighting has to be rewritten.

---

## M12 — decide now how freeze-then-unfreeze is implemented

Not a contradiction in the text, but it will force one in Phase 4 if left.

Ultralytics' `freeze=N` freezes the first N **layers** for the entire run.
There is no built-in "freeze for 10 epochs, then unfreeze at 10× lower lr".
§3.4.4 requires exactly that. Two implementations, and they differ in a way
§3.4.3 can detect:

- **Two `.train()` calls** (10 epochs frozen, then 90 unfrozen). Simple,
  defensible, resumable. But each call runs its own 3-epoch warmup and its own
  linear decay to lr0×lrf, so you get two warmups and two decay ramps — which
  is not §3.4.3's "decays linearly over the training schedule".
- **One `.train()` call with an `on_train_epoch_start` callback** that unfreezes
  at epoch 10 and rewrites the optimizer lr. One warmup, one decay ramp,
  matches the text. More code to defend, and you have to show the callback
  actually fired.

My recommendation is the callback, with the epoch-10 unfreeze and the resulting
lr logged per run so the schedule is inspectable from the run artifacts — §3.4.4
already promises training curves are reported, and that is where a grader would
look. If you take the two-call route instead, add one sentence to §3.4.3 saying
the schedule restarts at the unfreeze boundary. Decide before Track E writes the
harness; retrofitting changes every detector run.

---

# Part B — scaffold changes (already applied)

Shipped as `thesis-scaffold-patched.zip`. 16 tests, all passing.

## B1 — `src/stats/nadeau_bengio.py`: real bug, silently wrong answer

`dz_for_power` bisected directly on `scipy.stats.nct`. At the Bonferroni alpha
it returned **6.17** — below the 6.31 that reaches only ~57% power — i.e. it
claimed 80% power at an effect size that cannot deliver it.

Cause, precisely: the NaN is in the **lower-tail** term `nct.cdf(-c, df, ncp)`,
whose true value there is ~1e-16 — numerically irrelevant, but NaN poisons the
sum. And it is sensitive to the exact float of `c`, not confined to a clean
region: at df=4, α=0.05/70, `c = 14.097400007074867` gives NaN for ncp ≥ 14.1,
while the same call at `c = 14.0974` returns 5.5e-16. The bisection compared
`NaN < power`, which is False, so `hi` was driven down and it converged to a
wrong answer without raising. Intermittent, silent, and in the one module whose
whole job is to be trustworthy.

Fix: new public `power_at_dz()` integrates the chi-square denominator out by
hand instead of calling `nct` —

```
P(|t| > c | V=v) = Φ(ncp − c·√(v/df)) + Φ(−ncp − c·√(v/df))
```

averaged over the chi²(df) density of V. Verified: agrees with `nct` to ~1e-11
everywhere `nct` is finite, and with Monte Carlo (200k–400k draws) where it is
not. `dz_for_power` now bisects on that and raises if the requested power is
unreachable.

Results after the fix, all matching the manuscript:

| Quantity | Value | Manuscript |
|---|---|---|
| detectable d_z, α=.05 | 1.862 | 1.86 ✓ |
| d_z for 80% power, α=.05 | 2.398 | 2.40 ✓ |
| detectable d_z, α=.05/70 | 6.305 | > 6.30 ✓ |
| d_z for 80% power, α=.05/70 | 7.75 | ≈ 7.8 ✓ |
| power at d_z = 1.0 / 1.5 | 0.168 / 0.387 | ~0.17 / ~0.39 ✓ |

Also corrected a wrong docstring claim: `detectable_dz` is **not** the 50%-power
point. Power there is ~0.57 at k=5, because E[√(V/df)] < 1 for finite df puts
the boundary slightly below the centre of the alternative distribution.

## B2 — `tests/test_stats.py`: 10 → 16 tests

Six added, all pinning the regime the old code got wrong:

- `test_scipy_nct_is_nan_in_the_regime_we_care_about` — documents *why* the
  quadrature exists, using the exact critical value the code computes. If scipy
  ever fixes this, the test fails and you revisit the workaround deliberately.
- `test_bonferroni_detectable_dz_matches_manuscript` — 6.30
- `test_bonferroni_dz_for_power_matches_manuscript` — 7.8, and asserts it
  exceeds the detectable threshold, which the old 6.17 did not
- `test_power_is_monotone_and_ordered_against_the_threshold`
- `test_power_at_dz_reproduces_manuscript_power_curve` — 0.17 / 0.39 / 0.80
- `test_power_at_dz_agrees_with_monte_carlo_at_bonferroni_alpha`

(The handoff says 11 tests; the original file had 10.)

## B3 — `configs/base.yaml`: rewritten, was missing most of what it claimed

Previously held ~12 values and claimed "every §3.4 hyperparameter, exactly
once". Corrections and additions:

**Corrected**

- `epochs.stage2: 30` → `10`. §3.4.4 says "all Stage 2 runs therefore execute
  the full 10 epochs". The old value would have run every Stage 2 fine-tune
  three times too long, across all 30 pipelines.
- `versions.torch: "2.9.1+rocm7.2.1"` → `null`, with a comment that it is
  recorded *after* installation from the actual build string. A pinned ROCm
  version in a CUDA project is how someone ends up installing the wrong wheel.

**Added — detectors**

- `patience: 10` — §3.4.3 states early stopping at patience 10. The Ultralytics
  default is **100**. Without this set explicitly the manuscript's claim is
  false at runtime. This was the most consequential omission.
- `lrf: 0.01` — needed for "lr0 × lrf" to be a number
- `freeze_epochs: 10`, `unfreeze_lr_factor: 0.1`, and per-architecture
  `unfreeze_lr` (1e-3 YOLOv8m / 1e-5 RT-DETR-L) — §3.4.4
- Stage 2 block: 10 epochs, `patience: null` (early stopping disabled),
  `freeze_epochs: 0`, `lr_factor: 0.1` — §3.4.4

**Added — generators (§3.3, was entirely absent)**

Adam β 0.5/0.999, lr 2e-4, 200 epochs as 100 constant + 100 linear-to-zero,
batch 1, 286→256 random crop, hflip only, `save_period: 10`; ResNet-9 generator
with instance norm and tanh; 70×70 PatchGAN with 4 stride-2 layers, LeakyReLU
0.2, spectral norm on the discriminator only; dual attention with
`gamma_init: 0.0` at the 32×32 bottleneck; LSGAN adversarial, λ_fc 1.0 with the
{0.5, 1.0, 2.0} sweep, VGG-19 relu1_2/relu2_2/relu3_3/relu4_3, L1, equal static
weights, target `G(x)`.

**Added — Condition B allocation** (Table 3.6, 5 × 20%) and a **design** block
asserting the 7 bins / 30 runs / 60 states / 420 evaluations / 6 generator runs,
so a design change breaks a test rather than quietly changing Chapter IV.

**Four values marked `CONFIRM`**, not asserted: `ngf`, `ndf`, `lambda_cyc`,
`lambda_id`. §3.3.5 says the cycle and identity weights "follow the original
CycleGAN formulation \[32\]" but renders them as equation images I cannot read.
I put the canonical CycleGAN values (10 and 5) with a CONFIRM comment rather
than state them as read from your text. Check the rendered equation. Same for
the 64/64 channel widths, which §3.3.2 and §3.3.4 never state.

## B4 — `src/stats/holm.py`: docstring warning

Added a header making explicit that §3.6.4 does **not** correct across the
family of 70 — it designates one confirmatory comparison (family of one,
α=.05) and labels everything else exploratory and uncorrected. `family_size()`
and `holm_threshold()` exist only to reproduce §3.6.4's *argument* that
correcting across 70 would demand d_z > 6.30. As written, the module reads like
Holm is the analysis plan. It is not, and applying it to the 70 would contradict
the frozen methodology. The code itself is correct — the step-down logic
verifies.

## B5 — `README.md`: rewritten CUDA-first

The old one led with the ROCm ordering, so anyone following it installs the
wrong torch. Now: CUDA path first with the `+cu` verification and the ~124 MB
tell, ROCm demoted to a clearly-labelled contingency, and an instruction to get
the index URL from the live selector rather than copying it from a chat log.
Also documents the two deliberate design choices (`pyproject` omitting torch,
`config.py` raising) so nobody "fixes" them.

## B6 — `pyproject.toml`: comment only

Dependency list unchanged. The comment block explaining the torch/ultralytics
omission was ROCm-only; it now leads with the CUDA path and keeps ROCm as the
contingency. The omission itself is deliberate and stays.

## B7 — `configs/thresholds.yaml`: one addition

Added `alive_extraction_interval_s: null`. §3.5.5 fixes your extraction interval
at 15 s and flags the inherited ALIVE corpus's interval as pending confirmation
from the source group, with any discrepancy to be reported. That is a pending
measurement like any other and belongs in the file that raises on nulls.

---

# Part C — install sequence

Order matters. torch by hand first, everything else after.

```powershell
py -3.12 -m venv .venv-cuda
.venv-cuda\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

Get `cuXXX` from the selector at <https://pytorch.org/get-started/locally/>
(Stable / Windows / Pip / Python / your CUDA version). **I could not verify the
current value from here** — pypi.org is unreachable from this container and the
selector is JavaScript-driven; the static HTML I got back advertised PyTorch
2.7.0 and cu118/cu126/cu128, which looks stale for September 2026. Read it off
the live page. Do not take `cu128` from the old handoff.

Verify immediately:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print a version ending `+cuXXX`, `True`, and the RTX 4050. No `+cu` suffix
means you got the CPU wheel. A ~124 MB download means the index URL was wrong.

Then record the actual build string in `configs/base.yaml` under
`versions.torch`, and install the rest:

```powershell
pip install ultralytics==8.4.154 albumentations==2.0.8
pip install -e ".[dev]"
pytest -q          # expect 16 passed
```

Not verified from here: that `ultralytics==8.4.154` and `albumentations==2.0.8`
resolve on PyPI. Both are pinned in §3.7.2 and were presumably installed from a
working machine already, but pip will tell you in one line if not.

The Ultralytics defaults I *did* verify against the live configuration
reference — `warmup_epochs` 3.0 in epochs, `cos_lr` False, `lrf` 0.01,
`patience` **100**, `freeze` None, `optimizer` 'auto', `save_period` -1,
`mosaic` 1.0, `mixup` 0, `cutmix` 0 — came from today's docs page, not a
8.4.154-pinned snapshot. These are stable settings, but if a reviewer presses
on "framework default" claims, `check_ultralytics_defaults.py` run in the
pinned venv is the citable evidence, not the docs page.

---

# What I need from you

1. **M1** — confirm §3.5.1 switches to ⟦t1⟧/⟦t2⟧. This unblocks the `lum_band`
   column and therefore Step 10.
2. **M3** — does the RTX 5060 Ti exist?
3. **M4** — will you actually have the RX 9060 XT?
4. **M5** — 2,000 target / 1,000 minimum, or different numbers?
5. **M12** — callback or two `.train()` calls for freeze-then-unfreeze?
6. **B3** — check the four `CONFIRM` values against the rendered equations.

1 and 4 block Step 10 and the CSMO request respectively. The rest can follow.
