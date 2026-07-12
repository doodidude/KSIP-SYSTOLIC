"""
fp12sr_golden.py -- golden-model generator for the FP12 eager-SR accumulator
PE family (mx-systolic-fpga/src/fp12sr_accum/), verified bottom-up per the
project's approved plan (see AGENTS/plan notes):

  §4a PRNG            : lfsr_galois.sv            -- bit-exact Galois LFSR replica
  §S3 bridge stage    : mx_product_to_fp_operand.sv
  §4b SR adder        : sr_adder_fp12.sv          -- Ben-Ali eager-SR, blocks 1-10
  §4c single-PE/block : pe_fp12sr.sv              -- shared adder + 7-lane RR dispatch
  §conv FP12 -> BF16  : conv_fp12_2bf16.sv        -- folds in the two MX scale codes
  §4d full array      : top_fp12sr_systolic_mx.sv -- N x N, 3-cycle systolic stagger

The PE's LFSR is dispatch-gated (advances once per add issued to the shared
adder), so a PE's result is a pure function of its local element stream --
one block-level replay (pe_fp12sr_single_block) verifies a PE at any array
position, with no cycle-accurate timing model anywhere in this file.

Same conventions as golden-model.py: $readmemh-safe flat hex files (bare hex
tokens, one per line), a paired human-readable .meta.txt per target, and
self-checking SV testbenches using !== (4-state) compares.
"""

# ============================================================== §4a: LFSR
# Galois-form 13-bit LFSR, polynomial x^13 + x^4 + x^3 + x^1 + 1
# (taps {13,4,3,1}, canonical maximal-length polynomial per Xilinx XAPP052).
# poly_mask has bit (p-1) set for every tap p, including the top tap p=width.
LFSR_WIDTH = 13
LFSR_POLY_MASK = 0x100D  # bits 12, 3, 2, 0 set (taps 13, 4, 3, 1)


def lfsr_galois_step(reg: int, width: int = LFSR_WIDTH,
                      poly_mask: int = LFSR_POLY_MASK) -> int:
    """One Galois-LFSR update, bit-exact with lfsr_galois.sv's always_ff block:
        fb  = reg[0]
        reg = reg >> 1
        if (fb) reg = reg ^ poly_mask
    """
    fb = reg & 1
    reg >>= 1
    if fb:
        reg ^= poly_mask
    return reg & ((1 << width) - 1)


def lfsr_galois_sequence(seed: int, num_cycles: int, width: int = LFSR_WIDTH,
                          poly_mask: int = LFSR_POLY_MASK) -> list:
    """Returns [seed, step(seed), step^2(seed), ..., step^num_cycles(seed)] --
    length num_cycles+1, index i is the value BEFORE the i-th update (index 0
    is the raw seed itself, matching rand_out immediately after reset with
    enable held low, before any clocked update)."""
    assert seed != 0, "LFSR seed must be nonzero (all-zero state never escapes)"
    seq = [seed & ((1 << width) - 1)]
    reg = seq[0]
    for _ in range(num_cycles):
        reg = lfsr_galois_step(reg, width, poly_mask)
        seq.append(reg)
    return seq


# --------------------------------------------------------------- hex writer
def write_hex(path, values, hex_digits):
    """$readmemh-safe: bare hex tokens only, one per line."""
    with open(path, 'w') as fh:
        for v in values:
            if v is None:
                v = 0
            fh.write(f"{v:0{hex_digits}x}\n")


# ================================================================ TARGET §4a
def gen_lfsr_vectors(seed=0x1ACE, num_cycles=256, width=LFSR_WIDTH,
                      poly_mask=LFSR_POLY_MASK, outdir='.'):
    """LFSR target: seed -> num_cycles+1 sequential rand_out golden values.
    Writes: fp12sr_lfsr_seed.hex (1 line), fp12sr_lfsr_gold.hex (num_cycles+1
    lines, index 0 == seed, index i == state after i clocked updates)."""
    seq = lfsr_galois_sequence(seed, num_cycles, width, poly_mask)
    hex_digits = (width + 3) // 4

    write_hex(f"{outdir}/fp12sr_lfsr_seed.hex", [seed], hex_digits)
    write_hex(f"{outdir}/fp12sr_lfsr_gold.hex", seq, hex_digits)

    with open(f"{outdir}/fp12sr_lfsr_vectors.meta.txt", 'w') as fh:
        fh.write(f"LFSR target  width={width} poly_mask=0x{poly_mask:X} "
                  f"seed=0x{seed:X} num_cycles={num_cycles}\n")
        fh.write("fp12sr_lfsr_seed.hex : 1 line, "
                  f"{hex_digits}b hex (lfsr_galois.seed_in)\n")
        fh.write(f"fp12sr_lfsr_gold.hex : {len(seq)} lines, {hex_digits}b hex -- "
                  "index 0 is rand_out immediately after reset (== seed), "
                  "index i is rand_out after i clocked (enable=1) updates\n")

    print(f"[lfsr] wrote fp12sr_lfsr_seed.hex (1), "
          f"fp12sr_lfsr_gold.hex ({len(seq)})")
    return dict(seed=seed, seq=seq, width=width, poly_mask=poly_mask,
                num_cycles=num_cycles)


# ========================================================== §S3: bridge stage
# Bit-exact Python replica of mx_product_to_fp_operand.sv -- turns a raw
# mantissa-only exact product (u_prd, from the reused exact multiply) into
# a normalized (sign, FP12-bias exponent field, lossless fraction) operand.
FP12_EXP_W = 6
FP12_BIAS = 31


def mx_product_to_fp_operand(exp_width: int, man_width: int, prd_sign: int,
                              u_prd: int, exp0_field: int, exp1_field: int):
    """Bit-exact with mx_product_to_fp_operand.sv. Returns (sign, exp_field,
    frac) -- exp_field is the saturated FP12-bias (31) exponent field,
    frac is frac_width = 2*man_width+1 bits wide."""
    M = man_width
    exp_adjust = 33 - (1 << exp_width)
    exp_max = (1 << FP12_EXP_W) - 1

    if u_prd == 0:
        return 0, 0, 0

    shift_bit = (u_prd >> (2 * M + 1)) & 1
    if shift_bit:
        frac = u_prd & ((1 << (2 * M + 1)) - 1)          # u_prd[2M:0]
    else:
        frac = (u_prd & ((1 << (2 * M)) - 1)) << 1        # {u_prd[2M-1:0],1'b0}

    exp_wide = exp0_field + exp1_field + exp_adjust + shift_bit
    if exp_wide > exp_max:
        exp_out = exp_max
    elif exp_wide < 0:
        exp_out = 0
    else:
        exp_out = exp_wide

    return prd_sign, exp_out, frac


def gen_mx_product_to_fp_operand_vectors(exp_width, man_width, num_cases=200,
                                          seed=0, outdir='.', tag=''):
    """S3 standalone target: random (prd_sign, u_prd, exp0_field, exp1_field)
    -> golden (sign, exp_field, frac). u_prd is synthesized as F0*F1 from
    normalized man_width+1-bit mantissas (matching what the reused exact
    multiply actually produces post Sub-OFF-flush), plus a batch of forced
    F=0 cases to exercise the zero-product special case.
    Writes fp12sr_s3_{tag}_{sign,uprd,exp0,exp1}_in.hex and
    fp12sr_s3_{tag}_{sign,exp,frac}_gold.hex."""
    import random
    rng = random.Random(seed)
    M = man_width
    fi_width = M + 2
    fi_prd_width = 2 * fi_width
    frac_width = 2 * M + 1
    f_lo, f_hi = (1 << M), (1 << (M + 1)) - 1
    exp_field_max = (1 << exp_width) - 1

    cases = []
    for _ in range(max(4, num_cases // 20)):
        cases.append((0, rng.randint(f_lo, f_hi), rng.randint(0, 1),
                      rng.randint(0, exp_field_max), rng.randint(0, exp_field_max)))
    while len(cases) < num_cases:
        cases.append((rng.randint(f_lo, f_hi), rng.randint(f_lo, f_hi),
                      rng.randint(0, 1),
                      rng.randint(0, exp_field_max), rng.randint(0, exp_field_max)))

    in_sign, in_uprd, in_e0, in_e1 = [], [], [], []
    gold_sign, gold_exp, gold_frac = [], [], []
    for F0, F1, prd_sign, e0, e1 in cases:
        u_prd = F0 * F1
        s_out, exp_out, frac_out = mx_product_to_fp_operand(
            exp_width, man_width, prd_sign, u_prd, e0, e1)
        in_sign.append(prd_sign)
        in_uprd.append(u_prd)
        in_e0.append(e0)
        in_e1.append(e1)
        gold_sign.append(s_out)
        gold_exp.append(exp_out)
        gold_frac.append(frac_out)

    p = f"{outdir}/fp12sr_s3_{tag}"
    write_hex(f"{p}_sign_in.hex", in_sign, 1)
    write_hex(f"{p}_uprd_in.hex", in_uprd, (fi_prd_width + 3) // 4)
    write_hex(f"{p}_exp0_in.hex", in_e0, (exp_width + 3) // 4)
    write_hex(f"{p}_exp1_in.hex", in_e1, (exp_width + 3) // 4)
    write_hex(f"{p}_sign_gold.hex", gold_sign, 1)
    write_hex(f"{p}_exp_gold.hex", gold_exp, (FP12_EXP_W + 3) // 4)
    write_hex(f"{p}_frac_gold.hex", gold_frac, (frac_width + 3) // 4)

    print(f"[s3:{tag}] exp_width={exp_width} man_width={man_width} "
          f"wrote {len(cases)} cases")
    return dict(exp_width=exp_width, man_width=man_width, cases=cases,
                fi_prd_width=fi_prd_width, frac_width=frac_width)


# ============================================================ §S4-S9: SR adder
# Bit-exact Python replica of sr_adder_fp12.sv -- Ben-Ali's eager-SR adder
# (blocks 1-10, plan stages S4-S9). Operand A is the FP12 E6M5 lane register
# (mant_a: 5 explicit bits, hidden bit = 1 iff exp_a != 0). Operand B is the
# S3 bridge-stage "increment" (frac_b: frac_width = 2*man_width+1 explicit
# bits -- 5 for man_width=2, 7 for man_width=3 -- hidden bit = 1 iff
# exp_b != 0; S3 guarantees frac_b == 0 whenever exp_b == 0). rand13 is one
# 13-bit LFSR draw consumed by this single add.
#
# Widths: P = FP12_SIG_W = 6 (hidden+5) is FP12's own precision -- the main
# adder/normalizer always operates on a (CW+1)-bit window (CW = B's
# frac_width for this format), so that BOTH operands' full real precision
# participates in the sum and in the eager sticky-round carry candidates
# (S'_1/S'_2). SR_TAIL_W=11 (=r-2) random bits, mixed with whatever of Y's
# own bits get shifted below the window, produce those carry candidates.
# A second, much smaller round-off (S9) then trims the (CW+1)-bit corrected
# significand down to FP12's native 5-bit mantissa; for man_width=3 formats
# (e4m3/e2m3, EXTRA=2) this consumes the 2 LFSR bits left over from the
# 13-bit draw (13-11=2) as genuine SR source -- not padding, a full use of
# every drawn bit. For man_width=2 formats (EXTRA=0) this stage is a no-op.
FP12_MANT_W = 5
FP12_SIG_W  = FP12_MANT_W + 1      # 6 = hidden + 5 explicit ("p")
LFSR_DRAW_W = 13                    # r
SR_TAIL_W   = LFSR_DRAW_W - 2        # 11 -- S6 sticky-round tail width (format-independent)


def sr_adder_fp12(sign_a: int, exp_a: int, mant_a: int,
                   sign_b: int, exp_b: int, frac_b: int, man_width: int,
                   rand13: int):
    """Returns (sign, exp, mant) -- the new FP12 lane value after adding B
    (an S3 operand) into A (the current FP12 lane register), using rand13
    as this add's single LFSR draw."""
    CW    = 2 * man_width + 1        # B's frac_width for this format (5 or 7)
    EXTRA = CW - FP12_MANT_W          # 0 or 2 -- real precision beyond FP12's mantissa

    if exp_a == 0 and exp_b == 0:
        return 0, 0, 0
    if exp_b == 0:
        return sign_a, exp_a, mant_a

    a_full = mant_a << EXTRA          # zero-extend A's 5-bit mantissa to B's CW-bit width

    # ---- Block 1: Exponent diff / Swap -- magnitude compare picks X (larger), Y (smaller) ----
    a_key = (exp_a, (1 << CW) | a_full) if exp_a != 0 else (0, 0)
    b_key = (exp_b, (1 << CW) | frac_b)
    if a_key >= b_key:
        x_sign, x_exp = sign_a, exp_a
        x_full = ((1 << CW) | a_full) if exp_a != 0 else 0
        y_sign, y_exp, y_full = sign_b, exp_b, (1 << CW) | frac_b
    else:
        x_sign, x_exp, x_full = sign_b, exp_b, (1 << CW) | frac_b
        y_sign, y_exp = sign_a, exp_a
        y_full = ((1 << CW) | a_full) if exp_a != 0 else 0

    reg_w     = (CW + 1) + SR_TAIL_W    # Y's shift-register width
    shift_amt = min(x_exp - y_exp, reg_w)
    op_sub    = (x_sign != y_sign)

    # ---- Blocks 2/3: Shift, 2's Complement (placed after shift) ----
    y_ext     = (y_full << SR_TAIL_W) & ((1 << reg_w) - 1)
    y_shifted = y_ext >> shift_amt
    if op_sub:
        y_shifted = (~y_shifted + 1) & ((1 << reg_w) - 1)

    # ---- Blocks 4/5/6: Fanout, Main adder, Sticky Round (parallel) ----
    y_main   = (y_shifted >> SR_TAIL_W) & ((1 << (CW + 1)) - 1)
    y_tail   = y_shifted & ((1 << SR_TAIL_W) - 1)
    main_sum = (x_full + y_main) & ((1 << (CW + 2)) - 1)

    rand_tail = rand13 & ((1 << SR_TAIL_W) - 1)
    stick_sum = (y_tail + rand_tail) & ((1 << (SR_TAIL_W + 1)) - 1)
    s1 = (stick_sum >> SR_TAIL_W) & 1          # no-shift carry candidate
    s2 = (stick_sum >> (SR_TAIL_W - 1)) & 1    # shift carry candidate

    # ---- Block 7: LZD/Shift (close path) + Normalization (far path), parallel ----
    close_path = op_sub and shift_amt <= 1
    if close_path:
        win = main_sum & ((1 << (CW + 1)) - 1)
        if win == 0:
            return 0, 0, 0
        lz = 0
        while ((win << lz) & (1 << CW)) == 0:
            lz += 1
        norm_sig = (win << lz) & ((1 << (CW + 1)) - 1)
        norm_exp = x_exp - lz
        sel_s = 0    # forced -- cancellation pulls tail bits up into mantissa territory
    else:
        carry_out = (main_sum >> (CW + 1)) & 1
        if carry_out:
            norm_sig = (main_sum >> 1) & ((1 << (CW + 1)) - 1)
            norm_exp = x_exp + 1
            sel_s = s2
        else:
            norm_sig = main_sum & ((1 << (CW + 1)) - 1)
            norm_exp = x_exp
            sel_s = s1

    # ---- Blocks 8/9: Trapezoid mux + Round Correction ----
    corrected = (norm_sig + sel_s) & ((1 << (CW + 2)) - 1)
    if (corrected >> (CW + 1)) & 1:
        corrected >>= 1
        norm_exp += 1

    # ---- Second-stage round-off: (CW+1)-bit corrected sig -> native P=6 bits ----
    native_top = corrected >> EXTRA
    extra_bits = corrected & ((1 << EXTRA) - 1)
    spare_rand = (rand13 >> SR_TAIL_W) & ((1 << EXTRA) - 1)
    round_sum  = extra_bits + spare_rand
    carry_fin  = (round_sum >> EXTRA) & 1 if EXTRA else 0
    native_sig = (native_top + carry_fin) & ((1 << (FP12_SIG_W + 1)) - 1)
    if (native_sig >> FP12_SIG_W) & 1:
        native_sig >>= 1
        norm_exp += 1

    # ---- Block 10: Increment / finalize (with the fp8_e5m2 saturating clamp) ----
    if norm_exp > 63:
        return x_sign, 63, (1 << FP12_MANT_W) - 1
    if norm_exp < 0:
        return 0, 0, 0    # Sub-OFF: underflow flushes to true zero, not a subnormal
    mant_out = native_sig & ((1 << FP12_MANT_W) - 1)
    return x_sign, norm_exp, mant_out


def gen_sr_adder_fp12_vectors(man_width, num_cases=300, seed=0, outdir='.', tag=''):
    """SR adder standalone target: random (A, B, rand13) triples -> golden
    (sign,exp,mant). Deliberately includes: exact-equality cancellation,
    close-path (small exponent diff, opposite sign), far-path normal adds,
    add-to-zero, both-zero, and (man_width==2 only, since fp8_e5m2 is the
    only format whose worst-case single-add exponent can reach FP12's
    ceiling per plan §1.2) forced saturating-clamp corner cases.
    Writes fp12sr_sr_{tag}_{a_sign,a_exp,a_mant,b_sign,b_exp,b_frac,rand}_in.hex
    and fp12sr_sr_{tag}_{sign,exp,mant}_gold.hex."""
    import random
    rng = random.Random(seed)
    CW = 2 * man_width + 1

    cases = []

    # exact-equality cancellation (a == b in magnitude, opposite sign -> exact zero)
    for _ in range(max(4, num_cases // 15)):
        exp = rng.randint(1, 62)
        mant = rng.randint(0, 31)
        frac = mant << (CW - FP12_MANT_W)
        sign_a = rng.randint(0, 1)
        cases.append((sign_a, exp, mant, 1 - sign_a, exp, frac))

    # close-path: equal or off-by-one exponent, opposite sign
    for _ in range(max(8, num_cases // 8)):
        exp_a = rng.randint(1, 62)
        exp_b = min(63, max(1, exp_a + rng.choice([-1, 0, 1])))
        cases.append((rng.randint(0, 1), exp_a, rng.randint(0, 31),
                      1, exp_b, rng.randint(0, (1 << CW) - 1)))
        cases[-1] = (cases[-1][0], cases[-1][1], cases[-1][2],
                      1 - cases[-1][0], cases[-1][4], cases[-1][5])

    # add-to-zero / both-zero
    for _ in range(max(4, num_cases // 20)):
        cases.append((0, 0, 0, rng.randint(0, 1), rng.randint(1, 62), rng.randint(0, (1 << CW) - 1)))
        cases.append((rng.randint(0, 1), rng.randint(1, 62), rng.randint(0, 31), 0, 0, 0))
    cases.append((0, 0, 0, 0, 0, 0))

    # forced saturating-clamp corner (fp8_e5m2 only, man_width==2's e5m2 instance is
    # distinguished from e3m2 by the caller passing a high exp_a/exp_b deliberately;
    # this generator is format-agnostic, so the caller decides whether to exercise it)
    for _ in range(max(4, num_cases // 20)):
        cases.append((rng.randint(0, 1), 63, 31, rng.randint(0, 1), 63, (1 << CW) - 1))
        cases[-1] = (cases[-1][0], 63, 31, cases[-1][0], 63, (1 << CW) - 1)  # same sign -> overflow, not cancel

    # far-path normal random fill
    while len(cases) < num_cases:
        cases.append((rng.randint(0, 1), rng.randint(1, 63), rng.randint(0, 31),
                      rng.randint(0, 1), rng.randint(1, 63), rng.randint(0, (1 << CW) - 1)))

    in_sa, in_ea, in_ma, in_sb, in_eb, in_fb, in_r = [], [], [], [], [], [], []
    gold_s, gold_e, gold_m = [], [], []
    for sign_a, exp_a, mant_a, sign_b, exp_b, frac_b in cases:
        rand13 = rng.randint(0, (1 << LFSR_DRAW_W) - 1)
        s_out, e_out, m_out = sr_adder_fp12(sign_a, exp_a, mant_a,
                                             sign_b, exp_b, frac_b, man_width, rand13)
        in_sa.append(sign_a); in_ea.append(exp_a); in_ma.append(mant_a)
        in_sb.append(sign_b); in_eb.append(exp_b); in_fb.append(frac_b)
        in_r.append(rand13)
        gold_s.append(s_out); gold_e.append(e_out); gold_m.append(m_out)

    p = f"{outdir}/fp12sr_sr_{tag}"
    write_hex(f"{p}_a_sign_in.hex", in_sa, 1)
    write_hex(f"{p}_a_exp_in.hex", in_ea, (FP12_SIG_W + 3) // 4)
    write_hex(f"{p}_a_mant_in.hex", in_ma, (FP12_MANT_W + 3) // 4)
    write_hex(f"{p}_b_sign_in.hex", in_sb, 1)
    write_hex(f"{p}_b_exp_in.hex", in_eb, (FP12_SIG_W + 3) // 4)
    write_hex(f"{p}_b_frac_in.hex", in_fb, (CW + 3) // 4)
    write_hex(f"{p}_rand_in.hex", in_r, (LFSR_DRAW_W + 3) // 4)
    write_hex(f"{p}_sign_gold.hex", gold_s, 1)
    write_hex(f"{p}_exp_gold.hex", gold_e, (FP12_SIG_W + 3) // 4)
    write_hex(f"{p}_mant_gold.hex", gold_m, (FP12_MANT_W + 3) // 4)

    print(f"[sr:{tag}] man_width={man_width} wrote {len(cases)} cases")
    return dict(man_width=man_width, cases=cases)


# =========================================================== §4c: single-PE/block
# Bit-exact Python replica of pe_fp12sr.sv for one full k-element block
# through one PE instance (intake + serial combine-tree drain). The PE's
# LFSR is dispatch-gated (lfsr_galois.enable = adder_valid_in), advancing
# exactly once per add issued to the shared sr_adder_fp12 -- so the draw an
# add consumes depends only on its position in the PE's own dispatch order
# (intake element i consumes draw i, combine step j consumes draw k+j),
# never on absolute cycle time. Each PE is therefore a pure function of its
# local element stream: this one block-level replay verifies a PE at any
# array position, behind any systolic stagger or feed gap, with no
# cycle-accurate modeling. NUM_LANES = 7 (P = L+1) comes from the
# lane-reuse-gap argument -- a lane isn't safely reusable until 7 cycles
# after its dispatch is captured (sr_adder_fp12's 6 register stages plus
# pe_fp12sr's own lane-regfile write register); see pe_fp12sr.sv's header
# comment for the cycle-by-cycle derivation.
NUM_LANES = 7                        # P = L+1


def mx_element_decode_multiply(exp_width, man_width, code0, code1):
    """S1 (reused exact decode, Sub-OFF mantissa flush replacing the exact
    design's own IEEE-subnormal encoding) + S2 (reused exact multiply),
    bit-exact with pe_fp12sr.sv's S1/S2 always_ff blocks. code0/code1 are
    packed MX element codes: sign<<(exp_width+man_width) | exp_field<<man_width
    | mant_bits. Returns (prd_sign, u_prd, exp0_field, exp1_field) -- S3's
    expected inputs."""
    def decode_one(code):
        sign = (code >> (exp_width + man_width)) & 1
        exp_field = (code >> man_width) & ((1 << exp_width) - 1)
        mant_bits = code & ((1 << man_width) - 1)
        nrm = 1 if exp_field != 0 else 0
        man_ext = ((1 << man_width) | mant_bits) if nrm else 0   # Sub-OFF flush
        return sign, exp_field, man_ext

    s0, e0, u_op0 = decode_one(code0)
    s1, e1, u_op1 = decode_one(code1)
    prd_sign = s0 ^ s1
    u_prd = u_op0 * u_op1
    return prd_sign, u_prd, e0, e1


def pe_fp12sr_single_block(exp_width, man_width, elements, seed, k=32):
    """elements: list of k (code_west, code_north) packed-MX-code pairs, in
    arrival order. Returns (sign, exp, mant) -- the final combined FP12
    lane value after the full intake + (NUM_LANES-1)-add serial
    combine-tree drain, bit-exact with pe_fp12sr.sv. LFSR draw d is the
    register state after d dispatch-gated updates (draw 0 == the seed,
    which the enable-gated LFSR still holds when the first dispatch's
    posedge captures rand_in and steps it)."""
    assert len(elements) == k
    CW = 2 * man_width + 1
    EXTRA = CW - FP12_MANT_W

    draws = lfsr_galois_sequence(seed, k + NUM_LANES - 2)   # k intake + NUM_LANES-1 combine draws

    s3_outputs = []
    for code0, code1 in elements:
        prd_sign, u_prd, e0, e1 = mx_element_decode_multiply(exp_width, man_width, code0, code1)
        s, e, f = mx_product_to_fp_operand(exp_width, man_width, prd_sign, u_prd, e0, e1)
        s3_outputs.append((s, e, f))

    lane_state = [(0, 0, 0)] * NUM_LANES
    for i in range(k):
        lane = i % NUM_LANES
        sa, ea, ma = lane_state[lane]
        sb, eb, fb = s3_outputs[i]
        lane_state[lane] = sr_adder_fp12(sa, ea, ma, sb, eb, fb, man_width, draws[i])

    for j in range(NUM_LANES - 1):
        sa, ea, ma = lane_state[0]
        sb, eb, fb_native = lane_state[j + 1]
        fb = fb_native << EXTRA   # zero-extend FP12's native 5-bit mantissa to CW bits
        lane_state[0] = sr_adder_fp12(sa, ea, ma, sb, eb, fb, man_width, draws[k + j])

    return lane_state[0]


def random_mx_code(exp_width, man_width, rng):
    """Random packed MX element code, weighted to exercise Sub-OFF flush
    (denormal codes) and true zero alongside ordinary normals."""
    roll = rng.random()
    sign = rng.randint(0, 1)
    if roll < 0.10:
        exp_field, mant_bits = 0, 0                       # true zero
    elif roll < 0.25:
        exp_field, mant_bits = 0, rng.randint(1, (1 << man_width) - 1)  # denormal -> Sub-OFF flush
    else:
        exp_field = rng.randint(1, (1 << exp_width) - 1)
        mant_bits = rng.randint(0, (1 << man_width) - 1)
    return (sign << (exp_width + man_width)) | (exp_field << man_width) | mant_bits


def gen_pe_fp12sr_single_block_vectors(exp_width, man_width, num_blocks=20, k=32,
                                        pe_id=0, seed_base=0x1ACE, seed=0,
                                        outdir='.', tag=''):
    """Single-PE/single-block target (plan §4c): num_blocks independent
    k-element blocks, each driven through one pe_fp12sr instance (with an
    intervening reset pulse between blocks), checked against
    pe_fp12sr_single_block. Writes fp12sr_pe_{tag}_{west,north}_in.hex
    (num_blocks*k lines, bit_width hex each) and
    fp12sr_pe_{tag}_{sign,exp,mant}_gold.hex (num_blocks lines each)."""
    import random
    rng = random.Random(seed)
    bit_width = 1 + exp_width + man_width
    lfsr_seed = (seed_base ^ pe_id) & 0x1FFF
    lfsr_seed |= 1   # sr_adder's rand_in feed never needs a zero seed guard beyond this

    west_codes, north_codes = [], []
    gold_sign, gold_exp, gold_mant = [], [], []
    for _ in range(num_blocks):
        elements = [(random_mx_code(exp_width, man_width, rng),
                     random_mx_code(exp_width, man_width, rng)) for _ in range(k)]
        for c0, c1 in elements:
            west_codes.append(c0)
            north_codes.append(c1)
        s, e, m = pe_fp12sr_single_block(exp_width, man_width, elements, lfsr_seed, k=k)
        gold_sign.append(s)
        gold_exp.append(e)
        gold_mant.append(m)

    p = f"{outdir}/fp12sr_pe_{tag}"
    write_hex(f"{p}_west_in.hex", west_codes, (bit_width + 3) // 4)
    write_hex(f"{p}_north_in.hex", north_codes, (bit_width + 3) // 4)
    write_hex(f"{p}_sign_gold.hex", gold_sign, 1)
    write_hex(f"{p}_exp_gold.hex", gold_exp, (FP12_SIG_W + 3) // 4)
    write_hex(f"{p}_mant_gold.hex", gold_mant, (FP12_MANT_W + 3) // 4)

    print(f"[pe:{tag}] exp_width={exp_width} man_width={man_width} "
          f"wrote {num_blocks} blocks x {k} elements")
    return dict(exp_width=exp_width, man_width=man_width, num_blocks=num_blocks, k=k,
                lfsr_seed=lfsr_seed)


# ============================================================ §4d: N x N array
def gen_array_vectors(exp_width, man_width, N=4, k=32, seed_base=0x1ACE, seed=0,
                       outdir='.', tag=''):
    """Array target (plan §4d): N x N pe_fp12sr grid, one flat k-element MX
    block per row (west) and per column (north), one shared E8M0-style scale
    byte per row/column. Each PE[i][j]'s pe_id = i*N+j (matching
    top_fp12sr_systolic_mx.sv's genvar-indexed .pe_id(i*N+j) parameter), and
    its own final (sign,exp,mant) is computed via the already-verified
    pe_fp12sr_single_block -- no global-cycle modeling needed here: each
    PE's internal FSM only reacts to its OWN local valid_in_left&&
    valid_in_top events, so it's insensitive to when, in absolute time, its
    own local block actually starts arriving. That timing correctness is
    entirely the SV testbench's job (systolic stagger on the RTL side).
    Writes fp12sr_arr_{tag}_{west,north}_in.hex (N*k lines each, row-major
    for west / column-major for north), fp12sr_arr_{tag}_scale_{west,north}
    _in.hex (N lines each), fp12sr_arr_{tag}_bf16_gold.hex (N*N lines,
    index i*N+j matching top_fp12sr_systolic_mx.sv's bf16_result[i*N+j])."""
    import random
    rng = random.Random(seed)
    bit_width = 1 + exp_width + man_width

    west_rows = [[random_mx_code(exp_width, man_width, rng) for _ in range(k)] for _ in range(N)]
    north_cols = [[random_mx_code(exp_width, man_width, rng) for _ in range(k)] for _ in range(N)]
    scale_west = [rng.randint(0, 254) for _ in range(N)]
    scale_north = [rng.randint(0, 254) for _ in range(N)]

    gold_bf16 = [0] * (N * N)
    for i in range(N):
        for j in range(N):
            pe_id = i * N + j
            lfsr_seed = ((seed_base ^ pe_id) & 0x1FFF) | 1
            elements = list(zip(west_rows[i], north_cols[j]))
            s, e, m = pe_fp12sr_single_block(exp_width, man_width, elements, lfsr_seed, k=k)
            gold_bf16[i * N + j] = conv_fp12_2bf16(s, e, m, scale_north[j], scale_west[i])

    west_flat = [c for row in west_rows for c in row]
    north_flat = [c for col in north_cols for c in col]

    p = f"{outdir}/fp12sr_arr_{tag}"
    write_hex(f"{p}_west_in.hex", west_flat, (bit_width + 3) // 4)
    write_hex(f"{p}_north_in.hex", north_flat, (bit_width + 3) // 4)
    write_hex(f"{p}_scale_west_in.hex", scale_west, 2)
    write_hex(f"{p}_scale_north_in.hex", scale_north, 2)
    write_hex(f"{p}_bf16_gold.hex", gold_bf16, 4)

    print(f"[arr:{tag}] exp_width={exp_width} man_width={man_width} N={N} "
          f"wrote {N*N} PE results")
    return dict(exp_width=exp_width, man_width=man_width, N=N, k=k)


# ===================================================== §conv: FP12 -> BF16
# Bit-exact Python replica of conv_fp12_2bf16.sv -- widens an FP12 E6M5
# lane/result value to BF16, folding in the block's two shared MX scale
# codes (E8M0, biased 127). No CLZ/renormalize needed (FP12 input is
# already normalized); no saturate/clamp (matches convert_fixed2bf16.sv's
# own unclamped exponent assignment).
def conv_fp12_2bf16(sign_in: int, exp_in: int, mant_in: int,
                     shared_scale_1: int, shared_scale_2: int):
    """Returns packed 16-bit BF16 code (sign<<15 | exp<<7 | mant)."""
    if exp_in == 0:
        bf16_exp = 0
        bf16_man = 0
    else:
        bf16_man = (mant_in << 2) & 0x7F
        exp_wide = exp_in + shared_scale_1 + shared_scale_2 - 158
        bf16_exp = exp_wide & 0xFF
    return (sign_in << 15) | (bf16_exp << 7) | bf16_man


def gen_conv_fp12_2bf16_vectors(num_cases=200, seed=0, outdir='.'):
    """conv target: random (sign_in, exp_in, mant_in, shared_scale_1,
    shared_scale_2) -> golden packed BF16 code. Includes a batch of forced
    exp_in==0 cases to exercise the exact-zero special case.
    Writes fp12sr_conv_{sign,exp,mant,ss1,ss2}_in.hex and
    fp12sr_conv_bf16_gold.hex."""
    import random
    rng = random.Random(seed)

    cases = []
    for _ in range(max(4, num_cases // 20)):
        cases.append((rng.randint(0, 1), 0, 0,
                       rng.randint(0, 254), rng.randint(0, 254)))
    while len(cases) < num_cases:
        cases.append((rng.randint(0, 1), rng.randint(1, 63), rng.randint(0, 31),
                       rng.randint(0, 254), rng.randint(0, 254)))

    in_sign, in_exp, in_mant, in_ss1, in_ss2, gold_bf16 = [], [], [], [], [], []
    for sign_in, exp_in, mant_in, ss1, ss2 in cases:
        bf16 = conv_fp12_2bf16(sign_in, exp_in, mant_in, ss1, ss2)
        in_sign.append(sign_in)
        in_exp.append(exp_in)
        in_mant.append(mant_in)
        in_ss1.append(ss1)
        in_ss2.append(ss2)
        gold_bf16.append(bf16)

    p = f"{outdir}/fp12sr_conv"
    write_hex(f"{p}_sign_in.hex", in_sign, 1)
    write_hex(f"{p}_exp_in.hex", in_exp, 2)
    write_hex(f"{p}_mant_in.hex", in_mant, 2)
    write_hex(f"{p}_ss1_in.hex", in_ss1, 2)
    write_hex(f"{p}_ss2_in.hex", in_ss2, 2)
    write_hex(f"{p}_bf16_gold.hex", gold_bf16, 4)

    print(f"[conv] wrote {len(cases)} cases")
    return dict(cases=cases)


# =================================================================== MAIN
if __name__ == '__main__':
    gen_lfsr_vectors(outdir='mx-systolic-fpga/tb/fp12sr_tb')
    for tag, ew, mw in [('e5m2', 5, 2), ('e4m3', 4, 3),
                        ('e3m2', 3, 2), ('e2m3', 2, 3)]:
        gen_mx_product_to_fp_operand_vectors(
            ew, mw, outdir='mx-systolic-fpga/tb/fp12sr_tb', tag=tag)
    for tag, mw in [('mw2', 2), ('mw3', 3)]:
        gen_sr_adder_fp12_vectors(mw, outdir='mx-systolic-fpga/tb/fp12sr_tb', tag=tag)
    for tag, ew, mw in [('e5m2', 5, 2), ('e4m3', 4, 3),
                        ('e3m2', 3, 2), ('e2m3', 2, 3)]:
        gen_pe_fp12sr_single_block_vectors(
            ew, mw, outdir='mx-systolic-fpga/tb/fp12sr_tb', tag=tag)
    gen_conv_fp12_2bf16_vectors(outdir='mx-systolic-fpga/tb/fp12sr_tb')
    for tag, ew, mw in [('e5m2', 5, 2), ('e4m3', 4, 3),
                        ('e3m2', 3, 2), ('e2m3', 2, 3)]:
        gen_array_vectors(ew, mw, N=4, outdir='mx-systolic-fpga/tb/fp12sr_tb', tag=tag)
