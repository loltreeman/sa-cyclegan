#!/usr/bin/env python3
"""
Manuscript verification checks — Blocks 1-5 of the edit sheet.

Usage:   python check_manuscript.py MANUSCRIPT.md

Works on Windows, macOS and Linux. No grep, no Git Bash, no WSL needed.
Handles both the escaped Google-Docs export (\\[\\[3.x\\]\\]) and cleaned text.
"""

import re
import sys
from collections import Counter

# Windows consoles default to a legacy codepage (cp1252, cp932...) that
# cannot print the corner-bracket placeholders. Force UTF-8 on stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

if len(sys.argv) < 2:
    sys.exit("usage: python check_manuscript.py MANUSCRIPT.md")

path = sys.argv[1]
raw = open(path, encoding="utf-8").read()
lines = raw.split("\n")

# Strip the backslash-escapes Google Docs sprinkles through markdown exports,
# so every pattern below matches whether or not you have cleaned them up.
clean = [re.sub(r"\\(.)", r"\1", ln) for ln in lines]

PASS, FAIL = "PASS", "FAIL"
results = []


def report(name, hits, expect_zero=True, note=""):
    ok = (len(hits) == 0) if expect_zero else (len(hits) > 0)
    results.append((name, PASS if ok else FAIL))
    print(f"\n{'='*70}\n{name}  [{PASS if ok else FAIL}]")
    if note:
        print(f"  {note}")
    if not hits:
        print("  (no matches)")
    for n, text in hits[:40]:
        print(f"  {n:5}  {text[:150]}")
    if len(hits) > 40:
        print(f"  ... and {len(hits)-40} more")


def scan(pattern, start=1, end=None, flags=re.I):
    rx = re.compile(pattern, flags)
    end = end or len(clean)
    return [(i, clean[i-1].strip())
            for i in range(start, min(end, len(clean)) + 1)
            if rx.search(clean[i-1])]


# ---------------------------------------------------------------- locate Ch I
# Take the LAST match, not the first — the first is the table of contents.
_hits = [i for i, l in enumerate(clean, 1)
         if re.search(r"1\.5\s+Significance", l)]
ch1_end = _hits[-1] if _hits else 140

# ------------------------------------------------------------------ CHECK 1
survivors = scan(r"mixup|cutmix|cosine anneal|randaugment|1000 iterations")
expected = [h for h in survivors
            if re.search(r"disabled|left at their default", h[1], re.I)]
unexpected = [h for h in survivors if h not in expected]
report("1. Route B — stale framework claims",
       unexpected,
       note=("Expected survivors are the sentences that name MixUp/CutMix as "
             f"deliberately disabled. Found {len(expected)} of those."))

# ------------------------------------------------------------------ CHECK 2
report("2. Cross-reference placeholders",
       scan(r"\[\[3[.0-9a-z]*\]\]"),
       note="Every [[3.x]] / [[3.a]] / [[3.3.7]] should be a real number now.")

# ------------------------------------------------------------------ CHECK 3
report("3. Chapter I — wrong metric / test / modality",
       scan(r"mAP@50|paired t-test|glare", end=ch1_end),
       note=f"Scanning lines 1-{ch1_end} (up to '1.5 Significance').")

# ------------------------------------------------------------------ CHECK 4
report("4. [PLACEHOLDER] tokens", scan(r"\[PLACEHOLDER\]"))

# ------------------------------------------------------------------ CHECK 5
report("5. Decisions that should be closed",
       scan(r"⟦0\.90⟧|⟦m⟧|⟦1 or 3⟧|⟦N_min⟧|\[\[N\]\]|\[\[10,000\]\]|"
            r"⟦gap⟧|\[\[5–10\]\]|\[\[four\]\]|Confirm with Track|"
            r"Track G wording|Confirm this matches",
            flags=0),   # case-SENSITIVE: [[N]] is decision 5, [[n]] is a live count
       note="These are the twelve decisions. All should be gone.")

# ------------------------------------------------------------------ CHECK 6
report("6. Typos introduced in the last revision",
       scan(r"requiresix|\baand\b|camera ,|⟦N\\+_C⟧ x|improves \.|"
            r"five-to-ten-minute"),
       note="Block 2 of the edit sheet.")

# ------------------------------------------------------------------ CHECK 7
blockquotes = [(i, clean[i-1].strip())
               for i in range(200, len(clean) + 1)
               if clean[i-1].startswith("> ")]
report("7. Leaked markdown blockquotes in Ch. III", blockquotes,
       note="My '>' quoting should not appear in the manuscript body.")

# ------------------------------------------------------------- INVENTORY
print(f"\n{'='*70}\nREMAINING PLACEHOLDERS — everything here must be "
      f"measurement-gated\n{'='*70}")

corner = Counter(re.findall(r"⟦[^⟧]{0,90}⟧", "\n".join(clean)))
blank = sum(1 for m in re.findall(r"\[\[[^\]]{0,90}\]\]", "\n".join(clean))
            if not m.strip("[] "))
named = Counter(m for m in re.findall(r"\[\[[^\]]{0,90}\]\]", "\n".join(clean))
                if m.strip("[] "))

DECIDABLE = {"⟦tool⟧", "⟦cite⟧", "⟦institutional storage location⟧",
             "⟦x.y.z⟧", "⟦a.b.c⟧"}

for token, n in (corner + named).most_common():
    flag = "  <-- DECIDABLE TODAY" if token in DECIDABLE else ""
    print(f"  {n:4}  {token}{flag}")
print(f"  {blank:4}  [[ ]]  (blank table cells — Chapter IV results)")

total = sum(corner.values()) + sum(named.values()) + blank
print(f"\n  TOTAL MARKERS: {total}")

# ------------------------------------------------------------------ SUMMARY
print(f"\n{'='*70}")
failed = [n for n, s in results if s == FAIL]
for name, status in results:
    print(f"  [{status}]  {name}")
print(f"\n  {len(results)-len(failed)}/{len(results)} checks passed.")
sys.exit(1 if failed else 0)
