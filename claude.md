# MX Systolic Array PE — Accumulator Design Notes

## Context

Building a RISC-V coprocessor with an MX-format systolic array (ACCL lab, KAUST). BF16toMX encoder is done and verified end-to-end against a microxcaling-derived golden model (4 blocks × 32 elements of fp8_e4m3, all match on Verilator). Next up: the PE — specifically the accumulator strategy.

## The design space

Two accumulator families dominate. Both are viable for MX-quantized operands after dequant-then-multiply:

- **Kulisch / wide fixed-point**: exact, no rounding inside the array. Cheaper for narrow-range formats (INT8-ish). Wider accumulator register per PE, ~30–40 bits for MX-E4M3 with k=32.
- **Narrow floating-point (BF16/FP12/FP16)**: cheaper per PE (fits existing BF16 datapath), but suffers swamping on long dot products. Needs mitigation: chunking, stochastic rounding (SR), or both.

Three orthogonal design axes:

| Axis | Options |
|---|---|
| Accumulator numeric type | Kulisch, narrow-FP, wide-FP |
| Rounding policy | RNE, stochastic (SR), dithered |
| Reduction structure | Sequential, chunked, tree-of-partial-sums |

Project decision so far: array outputs BF16/FP32, **no MX re-quant on output**.

## Reading list (in priority order)

1. **`accl-kaust/mx-systolic-fpga` (GitHub)** — own lab's artifact. Has Exact and BF16 PE variants, multiple pipeline depths. Start here before any paper.
2. **Ben Ali et al., arXiv:2404.14010** — FP8 multiplier + FP12 SR accumulator. Details eager SR (walked through in this doc).
3. **Ben Ali PhD thesis (HAL tel-05452584)** — expanded version, includes far/close path adder details.
4. **Wang et al., arXiv:1812.08011** — seminal chunked accumulation for FP8 training. Orthogonal to SR.
5. **arXiv:2401.14110** — chunked accumulation, systolic-array-aware chunk sizing.
6. **arXiv:2511.06313** — precision-scalable MX datapaths with reduction tree (2025).
7. **Uguen & de Dinechin (HAL hal-01488916v2)** — canonical Kulisch design-space study.
8. **"Exact Dot Product Accumulate Operators for 8-bit FP DL"** — Kulisch for FP8/E4M3, most direct map to our problem.

## Ben Ali FP12 SR MAC — algorithm summary

Reference architecture: FP8 E5M2 × FP8 E5M2 → FP12 E6M5 accumulator, eager SR, 13 random bits, no subnormal support. Achieves ~50% area/delay/energy vs FP32, ~13–29% vs FP16, near-baseline accuracy.

### Three MAC blocks

1. **Multiplier** — exact, no rounding. Outputs the full 2·p_m-bit product.
2. **Accumulator (SR adder)** — where all rounding happens.
3. **PRNG** — Galois LFSR, r-bit output, runs in parallel with the multiplier. One LFSR per PE, uniquely seeded.

### Why SR

Swamping: on long reductions, `S + t` with `|t| << ulp(S)` collapses to `S` under RNE, silently discarding the small term. SR randomizes the round-up decision with probability equal to the fractional distance from the lower grid point. Errors become zero-mean → total error grows as O(√N) instead of O(N).

### Eager SR adder — 10-block pipeline

1. **Exponent difference / Swap** — compares e_x, e_y. Swaps operands so m_x is the larger, m_y the smaller. Outputs shift amount `e_x − e_y`, close/far path selector `c/f`, and the `op` flag (effective add vs subtract).
2. **Shift** — right-shifts m_y by `e_x − e_y`. Output width `p − 1 + r` bits (holds mantissa + guard + deep tail for Sticky Round).
3. **2's Complement** — conditionally negates m_y for effective subtraction. Placed *after* shift to avoid sign-extension mess. Physically: XOR gates + carry-in of `op` to the main adder.
4. **Fanout** — top `p+1` bits → main adder, guard bit `G` → Round Correction, bottom `r−2` bits → Sticky Round. PRNG bits fan out similarly.
5. **Main adder** — `p+2`-bit unnormalized sum. Runs in parallel with Sticky Round.
6. **Sticky Round** — adds `r−2` LSBs of shifted m_y to `r−2` PRNG bits. Outputs the top 2 bits of that sum: `S'_1, S'_2` — two possible random-carry candidates for the two possible normalization outcomes.
7. **LZD/Shift** (close path) + **Normalization** (far path) — run in parallel. Normalization is trivial (0-bit or 1-bit right shift). LZD/Shift counts leading zeros and barrel-shifts left (cancellation case). LZD/Shift forces `S' = 0,0` because cancellation pulls tail bits up into mantissa territory.
8. **Trapezoid mux** — selects close vs far path based on `c/f`.
9. **Round Correction** (the money block):
   - Left adder: `R,S + S'` → candidate for **no-shift** case, uses S'_1
   - Right adder: `R,G + S'` → candidate for **shift** case, uses S'_2 (G was promoted into round position by the right-shift)
   - CarrySelect mux: picks between the two based on whether Normalization actually shifted
   - Bottom adder: applies selected carry to mantissa LSB region
10. **Increment** — finalizes mantissa + exponent → output `z`.

### Why eager wins over lazy

Lazy SR does random-bit addition *after* normalization, forcing LZD/Shift and Normalization to be `p + r` bits wide (~19 bits for FP12 with r=13). Eager SR computes both possible random carries *speculatively in parallel* with normalization, so those blocks stay `p + 1` bits wide (~7 bits). Speculation cost is tiny (two 3-bit adders + one mux, ~20 gates). Wins are big: 26.6% latency, 18.5% area vs lazy.

### Critical path

Main `+` → LZD/Shift or Normalization → Trapezoid mux → Round Correction → Increment. Sticky Round + PRNG never on the critical path.

### The elegant part

SR probability = fractional distance from floor. This is NOT computed explicitly. Adding uniform random bits to the tail and checking for carry-out into the round position is arithmetically equivalent to comparing the tail against a uniform random threshold. Value-dependent probability emerges naturally from an adder. No divider, no comparator.

### Key knob: r (number of random bits)

Paper explores r ∈ {4, 7, 9, 11, 13}. r=4 tanks accuracy (43% on ResNet20/CIFAR10). r=13 matches baseline (~91.4% vs FP32's 91.5%). r=13 costs modestly more than r=9 in area/delay (~10% each) — worth it.

### Subnormal support

Denormals fill the gap between 0 and smallest normal. Handling them in the adder adds special-case logic (alignment, normalization, corner cases). Dropping them ("Sub OFF") flushes denormals to 0. Accuracy hit negligible with r=13. Area savings modest but real (~4-5% for FP12).

## Decisions to make before RTL

1. **Accumulator format** — FP12 E6M5 (paper's pick, most efficient), BF16 (matches lab baseline, simpler), or Kulisch (exact but wider register per PE)?
2. **Rounding** — RNE (simple, deterministic, works for short reductions), SR eager (paper's pick, needs LFSR + Round Correction), or SR + chunking (belt-and-suspenders, more area)?
3. **Subnormal support** — Sub ON (spec-compliant, ~5% area cost) or Sub OFF (matches Ben Ali's optimal config)?
4. **PRNG width** — r=13 for matched-to-baseline accuracy, or smaller for area savings if accuracy budget allows.
5. **Where does the accumulator live?** — inside each PE (weight-stationary dataflow), or centralized reduction after the array (output-stationary)?

## Verification strategy (carry forward from BF16toMX work)

- Golden model in Python using microxcaling primitives (already have this pattern). **Use `round='even'` not `round='nearest'`** — microxcaling's "nearest" is round-half-away-from-zero, not RNE.
- Hex vector I/O for `$readmemh`, one flat file per port/signal.
- SV testbench with self-checking `!==` comparisons.
- Verilator flow: `verilator --binary --timing -Wall --Wno-WIDTHEXPAND --Wno-WIDTHTRUNC --Wno-GENUNNAMED --Wno-DECLFILENAME`. Nuke `obj_dir/` between builds.
- Predict expected output before running — every mismatch is either an RTL bug OR a golden model bug; the reflex to blame RTL first is a trap.

## Open questions worth asking Fahmy

1. Which accumulator mode does the current `mx-systolic-fpga` repo default to, and why?
2. Was SR considered in the TRETS work? If not, what killed it?
3. What's the target dot-product length (K)? Determines whether swamping is even the dominant error source vs. per-element quantization error.
4. Any measurement of the actual close-path activation frequency in real workloads? (Determines whether LZD/Shift dominates critical path in practice or is dead silicon most of the time.)

## Immediate next steps

1. Clone and read `accl-kaust/mx-systolic-fpga` — both PE variants side by side.
2. Decide accumulator format based on measurements from that repo, not from papers.
3. If going SR route: prototype Sticky Round + Round Correction as a standalone SV module, verify against a Python SR reference (build on existing `bf16_mx_golden.py` pattern).
4. Once single-PE MAC is verified, plumb it into an array — start with a small (4×4) systolic to keep debug tractable.