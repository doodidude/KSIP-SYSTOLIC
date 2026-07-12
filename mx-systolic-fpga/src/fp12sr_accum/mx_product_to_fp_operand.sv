`timescale 1ns / 1ps

// S3 "bridge" stage (plan §1.0/§1.1): turns the exact multiplier's raw
// mantissa-only product (S2 of pe_exact_4s.sv's reused decode+multiply) into
// a normalized floating operand the eager-SR adder (S4-S9) can consume.
// Not a numbered Ben-Ali block -- Ben-Ali's own MAC assumes it's already
// handed floating operands; this is the work that gets it there.
//
// Caller contract: exp0_field/exp1_field are the raw (biased, per-format)
// MX exponent fields forwarded from S1; u_prd is the unsigned mantissa-only
// product from S2 (fi_prd_width bits, of which only the low 2*man_width+2
// bits carry real data -- see pe_exact_4s.sv's fi_width padding). Sub-OFF
// denormal flushing must already have happened upstream in S1 (mantissa
// forced to 0 whenever an operand's exponent field is 0), so this module
// only has to special-case u_prd == 0 (an exact zero product) -- it does
// not need the nrm flags directly.
//
// Mantissa renormalization: two operand mantissas are each in [1,2), so
// their product lands in [1,4) -- a single conditional bit-select (not a
// barrel shift) picks out the correctly-aligned fraction with zero
// precision loss, using one extra bit of headroom in the shifted case:
//   shift_bit = 0  ->  frac = {u_prd[2M-1:0], 1'b0}   (2M real bits, 1 pad)
//   shift_bit = 1  ->  frac = u_prd[2M:0]             (2M+1 real bits)
// (M = man_width). Output frac_out is therefore lossless and wider than
// FP12's own 5-bit mantissa for man_width > 2 -- the extra bits are genuine
// product precision for the SR adder's sticky-round stage (S6), not random
// padding.
//
// Exponent: combine the two per-element unbiased exponents (using each
// format's own bias = 2^(exp_width-1)-1, confirmed against
// golden-model.py/microxcaling's _get_format_params convention) directly
// into FP12 E6M5's bias (31), folding format bias removal and FP12's own
// bias into one constant per format: exp_adjust = 33 - 2^exp_width.
// Overflow resolution (plan §1.2): only fp8_e5m2's absolute worst case can
// reach FP12's exponent ceiling, and only after further growth across the
// k=32 accumulation (not from a single product alone) -- but this module
// still saturates defensively, using exp_guard extra bits of headroom so
// the add can't wrap before the clamp fires. The engaging clamp for that
// corner lives in sr_adder_fp12.sv's S9 (post-accumulate); this one is a
// no-op for all four target formats at the single-product level.
module mx_product_to_fp_operand #(
    parameter exp_width = 4,
    parameter man_width = 3,
    localparam fi_width      = man_width + 2,
    localparam fi_prd_width  = 2 * fi_width,
    localparam frac_width    = 2 * man_width + 1,
    localparam fp12_exp_w    = 6,
    localparam exp_guard     = 2,
    localparam exp_int_w     = fp12_exp_w + exp_guard,
    localparam signed [exp_int_w:0] exp_adjust = 33 - (1 << exp_width),
    localparam signed [exp_int_w:0] exp_max    = (1 << fp12_exp_w) - 1
)(
    input  logic clk,
    input  logic rst,
    input  logic valid_in,               // caller ANDs both systolic-path valids
    input  logic prd_sign,
    input  logic unsigned [fi_prd_width-1:0] u_prd,
    input  logic [exp_width-1:0] exp0_field,
    input  logic [exp_width-1:0] exp1_field,

    output logic valid_out,
    output logic sign_out,
    output logic [fp12_exp_w-1:0] exp_out,   // saturated FP12-bias exponent field
    output logic [frac_width-1:0] frac_out   // lossless fraction, >= FP12 mantissa width
);

    logic shift_bit;
    logic [frac_width-1:0] frac_comb;
    logic signed [exp_int_w:0] exp_wide;
    logic is_zero;

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_out <= 1'b0;
            sign_out  <= 1'b0;
            exp_out   <= '0;
            frac_out  <= '0;
        end else begin
            valid_out <= valid_in;

            is_zero   = (u_prd == '0);
            shift_bit = u_prd[2*man_width+1];
            frac_comb = shift_bit ? u_prd[2*man_width:0]
                                   : {u_prd[2*man_width-1:0], 1'b0};
            exp_wide  = $signed({1'b0, exp0_field}) + $signed({1'b0, exp1_field})
                        + exp_adjust + $signed({7'b0, shift_bit});

            if (is_zero) begin
                sign_out <= 1'b0;
                exp_out  <= '0;
                frac_out <= '0;
            end else begin
                sign_out <= prd_sign;
                frac_out <= frac_comb;
                if (exp_wide > exp_max)
                    exp_out <= exp_max[fp12_exp_w-1:0];
                else if (exp_wide < 0)
                    exp_out <= '0;
                else
                    exp_out <= exp_wide[fp12_exp_w-1:0];
            end
        end
    end

endmodule
