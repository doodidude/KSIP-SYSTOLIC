`timescale 1ns / 1ps

// FP12 E6M5 (sign, exp[5:0] biased 31, mant[4:0]) -> BF16 (sign, exp[7:0]
// biased 127, mant[6:0]), folding in the block's two shared MX scale codes
// (E8M0, biased 127, per golden-model.py::encode_block's scale_code) the
// same way convert_fixed2bf16.sv folds shared_scale_1/2 into in_bias --
// simpler here since the FP12 input is already normalized floating point
// (no CLZ/renormalize needed, unlike the wide-Kulisch case that module
// handles).
//
//   bf16_unbiased_exp = fp12_unbiased_exp   + shared_exp_1        + shared_exp_2
//                     = (exp_in - 31)       + (shared_scale_1-127) + (shared_scale_2-127)
//   bf16_biased_exp   = bf16_unbiased_exp + 127
//                     = exp_in + shared_scale_1 + shared_scale_2 - 158
//
// No saturate/clamp here, matching convert_fixed2bf16.sv's own unclamped
// exponent assignment -- overflow protection for the accumulate path
// already lives upstream (mx_product_to_fp_operand.sv's S3 clamp and
// sr_adder_fp12.sv's S9 clamp, plan §1.2); this stage assumes exp_in and
// the shared scales are within the range those upstream clamps and a
// non-NaN (!=255) scale code guarantee.
module conv_fp12_2bf16 (
    input  logic        sign_in,
    input  logic [5:0]  exp_in,        // FP12 biased-31 field; 0 == exact zero
    input  logic [4:0]  mant_in,       // FP12 explicit mantissa
    input  logic [7:0]  shared_scale_1,
    input  logic [7:0]  shared_scale_2,
    output logic [15:0] o_bf16
);

    localparam signed [9:0] EXP_ADJUST = -10'sd158; // 127 - 31 - 127 - 127

    logic bf16_sgn;
    logic [7:0] bf16_exp;
    logic [6:0] bf16_man;
    logic signed [9:0] exp_wide;

    assign bf16_sgn = sign_in;
    assign bf16_man = (exp_in == 6'd0) ? 7'd0 : {mant_in, 2'b00};

    assign exp_wide = $signed({4'b0, exp_in}) + $signed({2'b0, shared_scale_1}) +
                       $signed({2'b0, shared_scale_2}) + EXP_ADJUST;

    assign bf16_exp = (exp_in == 6'd0) ? 8'd0 : exp_wide[7:0];

    assign o_bf16 = {bf16_sgn, bf16_exp, bf16_man};

endmodule
