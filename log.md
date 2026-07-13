# Project Decision Log

Running record of decisions taken on the MX systolic-array PE. Newest entries
at the bottom of each section. Keep this current whenever a design choice is
made or reversed.

---

## Accumulator design (locked in)

- **Accumulator format: FP12 E6M5.** Chosen over BF16 (lab baseline, simpler)
  and Kulisch (exact but wider per-PE register). Follows Ben Ali et al.
  (arXiv:2404.14010) as the most area/energy-efficient point that still hits
  near-baseline accuracy.
- **Rounding: eager stochastic rounding (SR), r=13.** Eager (not lazy) keeps
  the LZD/Shift + Normalization blocks narrow (~7 bits vs ~19) by computing
  both possible random carries speculatively in parallel. r=13 matches
  FP32-baseline accuracy; r≤4 tanks it.
- **Subnormals: Sub-OFF.** Denormal operands flushed to 0 at decode. Matches
  Ben Ali's optimal config; accuracy hit negligible at r=13; saves ~4-5% area.
- **Accumulator placement: inside each PE (weight-stationary).** One shared,
  time-multiplexed `sr_adder_fp12` per PE with a round-robin lane register file.
- **Output: BF16, no MX re-quant.** Final FP12 lane value widened to BF16 with
  the block's two shared E8M0 MX scale codes folded in (`conv_fp12_2bf16.sv`).

## Microarchitecture

- **NUM_LANES = 7 (= L+1), not 6.** `sr_adder_fp12` has 6 register stages, and
  writing the result into a lane register is itself a 7th synchronous stage, so
  a lane isn't safely reusable until 7 cycles after its dispatch is captured.
  This corrects the plan's flagged §1.3 margin caveat.
- **Combine tree is serial + valid-gated.** The drain FSM waits for each combine
  add's `valid_out` before issuing the next, rather than assuming a fixed cycle
  count (each combine add depends on the previous add's actual result).
- **Systolic stagger = 3 cycles/hop.** Row i / col j feeds are offset by 3·i /
  3·j to match `pe_fp12sr`'s own 3-deep pass-through pipeline, so element n of
  row i and col j reach PE[i][j] on the same cycle.

## Verification methodology

- Golden model in Python (`fp12sr_golden.py`), bit-exact replica of each RTL
  module, built bottom-up: LFSR → S3 bridge → SR adder → conv → single-PE →
  array. Six self-checking Verilator TBs, one per stage.
- microxcaling golden uses `round='even'` (its "nearest" is round-half-away, not
  RNE) — carried over from the BF16toMX work.
- Hex vector I/O via `$readmemh` (bare tokens, one per line), self-checking `!==`
  compares, `.meta.txt` sidecar per target.
- Every mismatch is *either* an RTL bug *or* a golden-model bug — do not
  reflexively blame RTL.

## Session log

### 2026-07-12 — FP12-SR PE + 4×4 array bring-up complete

- **Bug found & fixed: dispatch-gate the per-PE PRNG.** The 4×4 array TB failed
  35/64 PE results (all 4 formats), each off by one low-order mantissa bit (the
  SR-draw fingerprint). Cause: the LFSR was free-running (`enable(1'b1)`), so the
  draw an add consumes depended on absolute cycle time; the systolic stagger
  phase-shifted every off-diagonal PE's draw sequence away from the golden
  model's cycle-1 assumption. **Fix:** `enable(adder_valid_in)` — LFSR advances
  once per dispatched add, so draw index tracks *local* dispatch order (intake
  element i → draw i, combine step j → draw k+j), never wall-clock. Each PE is
  now a pure function of its local element stream. See memory
  `gate-prng-on-dispatch`.
- **Consequence:** deleted all cycle-accurate timing modeling from
  `pe_fp12sr_single_block` in the golden (the `ADDER_LAT`/`LANE_WRITE_LAT`/
  `LANE_REUSE_GAP` constants and `t_in`/`n_dispatch`/`combine_start` machinery) —
  replaced by a plain sequential draw walk. One block-level replay now verifies
  a PE at any array position.
- **Status:** all six TBs pass on Verilator across e5m2/e4m3/e3m2/e2m3. The FP12
  eager-SR PE path is complete and bit-exact against the golden model.
- **Docs:** wrote a from-scratch project `README.md` at the repo root — full
  walkthrough of MX format, systolic arrays, the accumulator design question, the
  end-to-end dataflow, every fp12sr module (S1-S9 + combine), the array top,
  output conversion, the verification methodology, build/run instructions, a
  decision table, and a glossary. Structured for a first-time reader.

### 2026-07-12 — Prepared repo for public GitHub publishing

- **Artifact cleanup:** deleted all build/cache artifacts (root `obj_dir` 49M,
  `tb/fp12sr_tb/obj_dir` 70M, `__pycache__`, `.DS_Store`). Repo went 124M → ~5M.
- **Nested repos folded into one umbrella repo (user decision):**
  `mx-systolic-fpga/` was a separate clone of the public `accl-kaust/mx-systolic-fpga`
  with our `fp12sr_accum`/`fp12sr_tb` additions untracked on top. To publish the
  FP12-SR work as one repo, its nested `.git` was relocated out of the tree
  (moved, not deleted — recoverable backup in the session scratchpad). Its MIT
  LICENSE + README are retained for attribution.
- **microxcaling treated as an external dependency (default):** Microsoft's MIT
  library is *not* redistributed — it's gitignored and documented (requirements.txt,
  NOTICE, README). Only the legacy encoder golden (`golden-model.py`) needs it;
  `fp12sr_golden.py` is pure stdlib.
- **License (user decision): MIT © 2026 KAUST ACCL Lab.** Matches both upstream deps.
- **Repo hygiene added:** `.gitignore` (artifacts, `*.hex`/`*.meta.txt` generated
  vectors, personal `.claude/settings.local.json`, `*.code-workspace`, microxcaling),
  `.gitattributes` (LF normalization, mark vectors linguist-generated), `NOTICE`
  (third-party attribution), `requirements.txt`.
- **Generated test vectors are NOT committed** — regenerate with
  `python3 fp12sr_golden.py` (fp12sr) / `python3 golden-model.py` (encoder).
- **git init on `main`, one clean initial commit** (315 files). NOT pushed and no
  remote created — that's left to the user (needs their GitHub account).
- **Re-verified after all changes:** vectors regenerate, the 4×4 array TB passes
  all four formats, and `git status` stays clean after a build (ignore rules work).

### 2026-07-13 — Chisel-compatible packed arrays + Chipyard integration (in progress)

- **SV ports converted from unpacked to packed arrays for Chisel BlackBox
  compatibility.** Chisel's `Vec`/`UInt` flattens to individual numbered wires
  (`data_in_west_0`, `data_in_west_1`, ...), which can't connect to SV unpacked
  arrays (`data_in_west [N]`). Changed `top_fp12sr_systolic_mx.sv` ports from
  unpacked `[N]` suffix to packed `[N-1:0]` prefix form (e.g.,
  `input logic [N-1:0][bit_width-1:0] data_in_west`). Packed 2D arrays are
  bit-identical to Chisel's flat `UInt((n * bitWidth).W)` and still support
  per-element indexing (`port[i]`), so internal generate-loop logic is unchanged.
  Testbench signal declarations updated to match. All 4 formats × 16 PEs still
  pass on Verilator 5.050.
- **Chipyard integration started (remote `chipyard` machine, separate repo).**
  Goal: instantiate the FP12-SR systolic array as a RoCC accelerator inside a
  Rocket SoC, elaborated and simulated via Chipyard's Verilator flow.
  - **Approach: RoCC (not MMIO).** RoCC accelerators are plugged in entirely via
    config fragments — no `DigitalTop.scala` changes, no `CanHavePeriphery`
    traits. The config fragment tells Rocket to instantiate the accelerator inside
    the tile. This is the standard Chipyard pattern (Gemmini, etc.).
  - **File layout:**
    - `generators/fp12_systolic/src/main/scala/FP12_Systolic.scala` —
      `TopFp12srSystolicMxBlackBox` (Chisel BlackBox wrapping the SV module, with
      `addResource` for all 6 SV files), `SystolicArrayAccelerator` (LazyRoCC
      wrapper, currently a minimal stub with tied-off data ports),
      `WithFP12SystolicArray` (config fragment adding the accelerator via
      `BuildRoCC`).
    - `generators/chipyard/src/main/scala/config/FP12Configs.scala` —
      `FP12RocketConfig` (extends `WithFP12SystolicArray ++ WithNBigCores(1) ++
      AbstractConfig`). Config lives in the chipyard project to avoid a circular
      sbt dependency (fp12_systolic depends on rocketchip; chipyard depends on
      fp12_systolic; config needs chipyard's `AbstractConfig`).
    - `generators/fp12_systolic/src/main/resources/vsrc/` — all 6 SV source files
      (`top_fp12sr_systolic_mx.sv`, `pe_fp12sr.sv`, `sr_adder_fp12.sv`,
      `mx_product_to_fp_operand.sv`, `lfsr_galois.sv`, `conv_fp12_2bf16.sv`).
    - `build.sbt` — `lazy val fp12systolic` project definition (depends on
      `rocketchip`, needs `commonSettings` + chisel library deps), added to
      chipyard's `.dependsOn(...)` list.
  - **Build command:** `make CONFIG=chipyard.FP12RocketConfig -j8` from
    `sims/verilator/`.
  - **Errors resolved so far:**
    1. `DigitalTop is already defined` — user had duplicated the class while
       adding a `CanHavePeriphery` trait; reverted to original (not needed for
       RoCC approach).
    2. `not found: value fp12_systolic` — sbt project wasn't registered in
       `build.sbt`; added `lazy val` + `.dependsOn()`.
    3. sbt identifier mismatch (`fp12systolic` vs `fp12Systolic`) — names must
       match exactly between the `lazy val` definition and `.dependsOn()`.
  - **Remaining blocker:** `object util is not a member of package chisel3` —
    the fp12systolic sbt project is missing chisel3 library dependencies. Fix:
    copy the `.settings(libraryDependencies ++= ...)` chain from an existing
    generator definition (e.g., `testchipip`) in `build.sbt`. The exact variable
    name (`rocketLibDeps`, `chiselLibDeps`, etc.) is Chipyard-version-dependent.
  - **Current state:** RoCC stub compiles Scala but fails on missing chisel3
    classpath. Data path is tied off (no functional DMA yet) — first goal is to
    get the skeleton elaborating into the SoC, then wire actual data movement.
  - **Lesson (early MMIO vs RoCC):** started with the MMIO `CanHavePeriphery`
    approach, which requires modifying `DigitalTop.scala` and plumbing TileLink.
    Switched to RoCC after hitting compilation issues — RoCC is config-only, no
    `DigitalTop` changes, simpler to get a first build. MMIO may be revisited
    later when adding a proper data-movement interface.
