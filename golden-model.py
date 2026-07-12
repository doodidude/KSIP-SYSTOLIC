"""
bf16_mx_golden.py -- standalone golden-model generator for BF16<->MX RTL
verification. Three independent test targets, each producing $readmemh hex files:

  ENCODER : BF16 in            -> MX elem codes + MXscale out  (BF16toMX)
  DECODER : MX elem codes+scale -> BF16 out                    (MXtoBF16, future)
            (codes always chained from the encoder's golden output --
             decoder is tested against what a CORRECT encoder would produce,
             not hand-crafted/arbitrary codes)
  MATMUL  : BF16 A, BF16 B     -> BF16 out (systolic array reference)
            accumulate dtype = BF16-rounded FP32, per project decision that
            the array outputs BF16/FP32, no MX re-quant stage on output.

Element rounding (RNE, denormals, saturation) is delegated to Microsoft
microxcaling's _quantize_elemwise_core for reference-exactness -- this is
the one piece NOT reinvented, since hand-rolling RNE-with-denorms is where
subtle bugs hide.

All hex files are $readmemh-safe: bare hex tokens only, one per line, no
comments/labels in the numeric files themselves (a separate .meta.txt is
written per run for human-readable context, never fed to $readmemh).

Input BF16 stimulus (A, and B for matmul) is FIXED/seeded by default so
encoder/decoder/matmul runs are directly comparable -- override via seed/
shape args when you're ready to vary it.
"""
import math
import torch
from dataclasses import dataclass
from mx.formats import ElemFormat, _get_format_params, FP32_MIN_NORMAL
from mx.elemwise_ops import _quantize_elemwise_core

SCALE_BITS = 8
SCALE_EMAX = 2 ** (SCALE_BITS - 1) - 1  # 127

# Fixed default stimulus knobs -- change these, not the call sites, to vary
# the "same input across targets" data in one place.
DEFAULT_SEED = 0
DEFAULT_BLOCK_SIZE = 32
DEFAULT_NUM_BLOCKS = 4
DEFAULT_ELEM_FORMAT = 'fp8_e4m3'


# --------------------------------------------------------------- format params
def fmt_params(elem_format: str):
    ef = ElemFormat.from_str(elem_format)
    ebits, mbits, emax, max_norm, _ = _get_format_params(ef)
    E = ebits
    M = mbits - 2
    bias = 2 ** (E - 1) - 1 if E > 0 else 0
    return dict(name=elem_format, ef=ef, E=E, M=M, bias=bias, emax=emax,
                max_norm=max_norm, ebits=ebits, mbits=mbits, is_int=(E == 0))


# ------------------------------------------------------------------ BF16 utils
def to_bf16_fp32(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 -> BF16 (RNE) and widen back to FP32. This FP32-valued
    tensor numerically IS the BF16 value (BF16 -> FP32 widening is exact,
    since BF16 is FP32's top 16 bits)."""
    return x.float().bfloat16().float()


def bf16_hex(x: torch.Tensor) -> list:
    """Pack each element's BF16 bit pattern -> list of 16-bit ints."""
    bf = x.bfloat16()
    try:
        raw = bf.view(torch.uint16)
    except (RuntimeError, TypeError):
        # older torch: no uint16 view support, go via int16 + mask
        raw = bf.view(torch.int16).to(torch.int32) & 0xFFFF
    return [int(v) for v in raw.reshape(-1).tolist()]


def bf16_from_bits(bits: list) -> torch.Tensor:
    """Inverse of bf16_hex: 16-bit int codes -> FP32 tensor (widened)."""
    import numpy as np
    arr = np.array(bits, dtype=np.uint16)
    bf = torch.from_numpy(arr.copy()).view(torch.bfloat16)
    return bf.float()


# ------------------------------------------------------------- element codec
def elem_encode(v: float, P: dict):
    """on-grid element value -> (sign, exp_field, mant, packed_uint)."""
    if P['is_int']:
        code = int(round(v * 64.0))
        code = max(-127, min(127, code))
        return (1 if code < 0 else 0), None, abs(code), code & 0xFF
    E, M, bias = P['E'], P['M'], P['bias']
    s = 1 if math.copysign(1.0, v) < 0 else 0
    a = abs(v)
    if a == 0.0:
        ef, mant = 0, 0
    else:
        m, e = math.frexp(a)
        mant_norm, exp_unb = m * 2.0, e - 1
        ef = exp_unb + bias
        if ef >= 1:
            mant = round((mant_norm - 1.0) * (1 << M))
            if mant == (1 << M):
                ef += 1
                mant = 0
        else:
            emin = 1 - bias
            ef, mant = 0, round(a / (2.0 ** (emin - M)))
    packed = (s << (E + M)) | (ef << M) | mant
    return s, ef, mant, packed


def elem_decode(s: int, ef: int, mant: int, P: dict) -> float:
    if P['is_int']:
        return mant * (1 if s == 0 else -1) / 64.0
    E, M, bias = P['E'], P['M'], P['bias']
    if ef == 0:
        val = mant * (2.0 ** ((1 - bias) - M))
    else:
        val = (1.0 + mant / (1 << M)) * (2.0 ** (ef - bias))
    return -val if s else val


def elem_decode_from_code(packed: int, P: dict) -> float:
    if P['is_int']:
        code = packed if packed < 128 else packed - 256  # two's complement
        return code / 64.0
    E, M = P['E'], P['M']
    s = (packed >> (E + M)) & 0x1
    ef = (packed >> M) & ((1 << E) - 1)
    mant = packed & ((1 << M) - 1)
    return elem_decode(s, ef, mant, P)


# -------------------------------------------------------------- block codec
@dataclass
class MXBlock:
    scale_code: int          # E8M0 code, 0..255 (255 = NaN)
    shared_exp: int           # decoded exponent (scale = 2**shared_exp), None if NaN
    elem_codes: list          # packed ints, one per element
    decoded: list              # fp32 values = elem_decode * scale


def _shared_exp(block_max: float, emax: int):
    if not math.isfinite(block_max):
        return None
    guard = FP32_MIN_NORMAL if block_max == 0 else 0.0
    s_raw = math.floor(math.log2(block_max + guard))
    s = s_raw - emax
    if s > SCALE_EMAX:
        return None
    if s < -SCALE_EMAX:
        s = -SCALE_EMAX
    return s


def encode_block(vec_bf16: torch.Tensor, P: dict) -> MXBlock:
    """Encode one BF16-valued block (len <= block_size) to MX. Input MUST
    already be BF16-rounded FP32 (i.e. what the RTL's BF16 port carries)."""
    vec = vec_bf16.float()
    bmax = float(vec.abs().max())
    s = _shared_exp(bmax, P['emax'])
    if s is None:
        n = len(vec)
        return MXBlock(255, None, [0] * n, [float('nan')] * n)
    scale = 2.0 ** s
    scaled = vec / scale
    q = _quantize_elemwise_core(scaled, P['mbits'], P['ebits'], P['max_norm'],
                                 round='even', allow_denorm=True,
                                 saturate_normals=True, custom_cuda=False)
    codes, dec = [], []
    for ev in q.tolist():
        sg, ef, mant, packed = elem_encode(ev, P)
        codes.append(packed)
        dec.append(elem_decode(sg, ef, mant, P) * scale)
    return MXBlock(s + 127, s, codes, dec)


def decode_block(blk: MXBlock, P: dict) -> list:
    """Decode an MXBlock back to BF16-rounded FP32 values (what MXtoBF16
    should output). Uses the STORED codes, not blk.decoded, so this is a
    genuine independent decode step (mirrors what decoder RTL does: read
    codes + scale, reconstruct value)."""
    if blk.scale_code == 255:
        return [float('nan')] * len(blk.elem_codes)
    scale = 2.0 ** blk.shared_exp
    vals = [elem_decode_from_code(c, P) * scale for c in blk.elem_codes]
    return [float(v) for v in to_bf16_fp32(torch.tensor(vals))]


# --------------------------------------------------------- stimulus (fixed)
def gen_fixed_bf16(shape, seed=DEFAULT_SEED, value_scale_range=(0.01, 40.0)):
    """Deterministic BF16-rounded FP32 tensor of given shape. Same seed +
    shape always reproduces the same stimulus -- this is the 'fixed input
    across test targets' generator. Change seed/value_scale_range to vary
    it later; call sites don't need to change."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = value_scale_range
    span = torch.empty(1).uniform_(lo, hi, generator=g).item()
    raw = torch.randn(shape, generator=g) * span
    return to_bf16_fp32(raw)


# --------------------------------------------------------------- hex writer
def write_hex(path, values, hex_digits):
    """$readmemh-safe: bare hex tokens only, one per line."""
    with open(path, 'w') as fh:
        for v in values:
            if v is None:
                v = 0
            fh.write(f"{v:0{hex_digits}x}\n")


def elem_width_bits(P: dict) -> int:
    return 8 if P['is_int'] else (1 + P['E'] + P['M'])


# ================================================================ TARGET 1
def gen_encoder_vectors(elem_format=DEFAULT_ELEM_FORMAT,
                         block_size=DEFAULT_BLOCK_SIZE,
                         num_blocks=DEFAULT_NUM_BLOCKS,
                         seed=DEFAULT_SEED, outdir='.'):
    """ENCODER target: BF16 in -> MX codes + MXscale golden out.
    Writes: enc_bf16_in.hex, enc_mxout_gold.hex, enc_mxscale_gold.hex
    Returns the block list too (consumed directly by gen_decoder_vectors --
    no file round-trip needed to chain them)."""
    P = fmt_params(elem_format)
    bf16_in = gen_fixed_bf16((num_blocks, block_size), seed=seed)

    bits_flat, codes_flat, scale_flat, blocks = [], [], [], []
    for i in range(num_blocks):
        row = bf16_in[i]
        blk = encode_block(row, P)
        bits_flat.extend(bf16_hex(row))
        codes_flat.extend(blk.elem_codes)
        scale_flat.append(blk.scale_code)
        blocks.append(blk)

    ew = elem_width_bits(P)
    write_hex(f"{outdir}/enc_bf16_in.hex", bits_flat, 4)
    write_hex(f"{outdir}/enc_mxout_gold.hex", codes_flat, (ew + 3) // 4)
    write_hex(f"{outdir}/enc_mxscale_gold.hex", scale_flat, 2)

    with open(f"{outdir}/enc_vectors.meta.txt", 'w') as fh:
        fh.write(f"ENCODER target  format={elem_format} block_size={block_size} "
                  f"num_blocks={num_blocks} seed={seed} elem_width_bits={ew}\n")
        fh.write("enc_bf16_in.hex      : num_blocks*block_size lines, 16b hex "
                  "(BF16toMX.BF16 input)\n")
        fh.write("enc_mxout_gold.hex   : num_blocks*block_size lines, "
                  f"{ew}b hex (BF16toMX.MXout expected)\n")
        fh.write("enc_mxscale_gold.hex : num_blocks lines, 8b hex "
                  "(BF16toMX.MXscale expected, one per block)\n")

    print(f"[encoder] wrote enc_bf16_in.hex ({len(bits_flat)}), "
          f"enc_mxout_gold.hex ({len(codes_flat)}), "
          f"enc_mxscale_gold.hex ({len(scale_flat)})")
    return dict(P=P, bf16_in=bf16_in, blocks=blocks, elem_format=elem_format,
                block_size=block_size, num_blocks=num_blocks)


# ================================================================ TARGET 2
def gen_decoder_vectors(encoder_result: dict, outdir='.'):
    """DECODER target: MX codes+scale (chained from encoder_result's golden
    blocks) -> BF16 golden out.
    Writes: dec_mxin.hex, dec_mxscale_in.hex, dec_bf16_gold.hex
    encoder_result MUST come from gen_encoder_vectors() -- decoder input is
    always what a correct encoder would have produced, per design decision."""
    P = encoder_result['P']
    blocks = encoder_result['blocks']
    ew = elem_width_bits(P)

    codes_flat, scale_flat, bf16_gold_flat = [], [], []
    for blk in blocks:
        codes_flat.extend(blk.elem_codes)
        scale_flat.append(blk.scale_code)
        bf16_gold_flat.extend(decode_block(blk, P))

    write_hex(f"{outdir}/dec_mxin.hex", codes_flat, (ew + 3) // 4)
    write_hex(f"{outdir}/dec_mxscale_in.hex", scale_flat, 2)
    write_hex(f"{outdir}/dec_bf16_gold.hex", bf16_hex(torch.tensor(bf16_gold_flat)), 4)

    with open(f"{outdir}/dec_vectors.meta.txt", 'w') as fh:
        fh.write(f"DECODER target  format={encoder_result['elem_format']} "
                  f"block_size={encoder_result['block_size']} "
                  f"num_blocks={encoder_result['num_blocks']} "
                  f"elem_width_bits={ew}\n")
        fh.write("dec_mxin.hex       : num_blocks*block_size lines, "
                  f"{ew}b hex (MXtoBF16.MXin, == enc_mxout_gold.hex)\n")
        fh.write("dec_mxscale_in.hex : num_blocks lines, 8b hex "
                  "(MXtoBF16.MXscale in, == enc_mxscale_gold.hex)\n")
        fh.write("dec_bf16_gold.hex  : num_blocks*block_size lines, 16b hex "
                  "(MXtoBF16.BF16out expected)\n")

    print(f"[decoder] wrote dec_mxin.hex ({len(codes_flat)}), "
          f"dec_mxscale_in.hex ({len(scale_flat)}), "
          f"dec_bf16_gold.hex ({len(bf16_gold_flat)})")
    return dict(codes=codes_flat, scales=scale_flat, bf16_gold=bf16_gold_flat)


# ================================================================ TARGET 3
def gen_matmul_vectors(elem_format=DEFAULT_ELEM_FORMAT,
                        block_size=DEFAULT_BLOCK_SIZE,
                        M=4, K=None, N=3, seed=DEFAULT_SEED, outdir='.'):
    """MATMUL target: BF16 A (M x K), BF16 B (K x N) -> BF16 golden out
    (M x N), where K defaults to block_size so A has exactly one MX block
    per row (mirrors the systolic array's per-block contraction). Accumulate
    dtype = BF16-rounded FP32 (array outputs BF16/FP32, no output MX
    re-quant, per project decision).

    Writes: mm_bf16_A.hex, mm_bf16_B.hex, mm_bf16_out_gold.hex
    Also writes mm_bf16_out_lossless_gold.hex: the NO-MX-loss BF16 matmul
    (A @ B directly in BF16-accumulate), for isolating "array logic bug"
    from "expected MX quantization error" when you diff against RTL later.
    """
    if K is None:
        K = block_size
    P = fmt_params(elem_format)

    g = torch.Generator().manual_seed(seed)
    A = to_bf16_fp32(torch.randn(M, K, generator=g) * 5)
    B = to_bf16_fp32(torch.randn(K, N, generator=g) * 5)

    # lossless BF16 reference: no MX quantization at all
    out_lossless = to_bf16_fp32(A.double() @ B.double())

    # MX-quantized reference: encode A row-wise, B column-wise (K = contraction
    # axis on both), decode, then BF16-accumulate matmul -- this is what the
    # systolic array is actually expected to produce.
    A_blocks = [encode_block(A[m, :], P) for m in range(M)]
    A_dec = torch.tensor([blk.decoded for blk in A_blocks])          # M x K
    Bt = B.transpose(0, 1).contiguous()                                # N x K
    B_blocks = [encode_block(Bt[n, :], P) for n in range(N)]
    B_dec = torch.tensor([blk.decoded for blk in B_blocks]).transpose(0, 1)  # K x N

    out_mx = to_bf16_fp32(A_dec.double() @ B_dec.double())

    write_hex(f"{outdir}/mm_bf16_A.hex", bf16_hex(A), 4)
    write_hex(f"{outdir}/mm_bf16_B.hex", bf16_hex(B), 4)
    write_hex(f"{outdir}/mm_bf16_out_gold.hex", bf16_hex(out_mx), 4)
    write_hex(f"{outdir}/mm_bf16_out_lossless_gold.hex", bf16_hex(out_lossless), 4)

    with open(f"{outdir}/mm_vectors.meta.txt", 'w') as fh:
        fh.write(f"MATMUL target  format={elem_format}  M={M} K={K} N={N} "
                  f"seed={seed}\n")
        fh.write("mm_bf16_A.hex                 : M*K lines, 16b hex, row-major "
                  "(systolic array A input)\n")
        fh.write("mm_bf16_B.hex                 : K*N lines, 16b hex, row-major "
                  "(systolic array B input)\n")
        fh.write("mm_bf16_out_gold.hex          : M*N lines, 16b hex, row-major "
                  "-- EXPECTED array output (includes MX quant error)\n")
        fh.write("mm_bf16_out_lossless_gold.hex : M*N lines, 16b hex, row-major "
                  "-- no-MX-loss reference, for error isolation only, NOT what "
                  "the array should match\n")

    print(f"[matmul] wrote mm_bf16_A.hex ({M*K}), mm_bf16_B.hex ({K*N}), "
          f"mm_bf16_out_gold.hex ({M*N}), mm_bf16_out_lossless_gold.hex ({M*N})")
    rel_err = float((out_mx - out_lossless).abs().max() / out_lossless.abs().max())
    print(f"[matmul] max rel err from MX quantization: {rel_err:.4e}")
    return dict(A=A, B=B, out_mx=out_mx, out_lossless=out_lossless)


# =================================================================== MAIN
if __name__ == '__main__':
    enc = gen_encoder_vectors()
    gen_decoder_vectors(enc)
    gen_matmul_vectors()