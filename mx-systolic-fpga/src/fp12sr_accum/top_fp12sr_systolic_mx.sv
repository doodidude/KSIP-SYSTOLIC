`timescale 1ns / 1ps

// N x N systolic array of pe_fp12sr PEs (plan §3), mirroring
// top_exact_systolic_mx.sv's generate-loop wiring (west/north edges,
// pe_data_right/bottom internal fabric, per-PE BF16 conversion) but with
// the exact design's wide-Kulisch accumulate + conv_fixed2bf16_adjusted
// replaced by the FP12 eager-SR accumulate (pe_fp12sr) + conv_fp12_2bf16.
//
// Completion signal: unlike the exact design's pass-through valid_right/
// valid_bottom (which are LEVEL signals, safely AND-able across both
// directions at the bottom-right corner), pe_fp12sr's own result_valid is
// a single-cycle PULSE fired once per finished block reduction. Because
// every PE has identical internal latency once its own local element 0
// arrives, and entry delay into PE[i][j] is 3*(i+j) cycles (3 = this PE's
// systolic pass-through latency, data_right/bottom_s1..s3), the
// bottom-right corner PE[N-1][N-1] has the unique maximum entry delay and
// therefore always finishes strictly last -- every other PE's result_sign/
// exp/mant (which hold their final value once written, they are not reset
// after the pulse) is already stable by the time the corner's result_valid
// pulses. So result_valid_out is simply that corner pulse, not an AND.


module top_fp12sr_systolic_mx #(
    parameter exp_width = 4,
    parameter man_width = 3,
    parameter bit_width = 1 + exp_width + man_width,
    parameter k = 32,
    parameter N = 32,
    parameter logic [12:0] SEED_BASE = 13'h1ACE,
    localparam FP12_MANT_W = 5
)(
    input  logic clk,
    input  logic rst,
    input  logic [bit_width-1:0] data_in_west [N],
    input  logic [bit_width-1:0] data_in_north [N],
    input  logic data_valid_west [N],
    input  logic data_valid_north [N],
    input  logic [7:0] shared_scale_west [N],
    input  logic [7:0] shared_scale_north [N],
    output logic [15:0] bf16_result [N*N],
    output logic result_valid_out
);
    logic [bit_width-1:0] pe_data_right [N][N];
    logic [bit_width-1:0] pe_data_bottom [N][N];
    logic pe_valid_right [N][N];
    logic pe_valid_bottom [N][N];

    logic pe_result_valid [N][N];
    logic pe_result_sign  [N][N];
    logic [5:0] pe_result_exp [N][N];
    logic [FP12_MANT_W-1:0] pe_result_mant [N][N];

    genvar i, j;
    generate
        for (i = 0; i < N; i++) begin : row
            for (j = 0; j < N; j++) begin : col
                pe_fp12sr #(
                    .exp_width(exp_width),
                    .man_width(man_width),
                    .k(k),
                    .pe_id(i*N + j),
                    .SEED_BASE(SEED_BASE)
                ) pe (
                    .clk(clk),
                    .rst(rst),
                    .data_in_left((j == 0) ? data_in_west[i] : pe_data_right[i][j-1]),
                    .data_in_top((i == 0) ? data_in_north[j] : pe_data_bottom[i-1][j]),
                    .valid_in_left((j == 0) ? data_valid_west[i] : pe_valid_right[i][j-1]),
                    .valid_in_top((i == 0) ? data_valid_north[j] : pe_valid_bottom[i-1][j]),
                    .data_out_right(pe_data_right[i][j]),
                    .data_out_bottom(pe_data_bottom[i][j]),
                    .valid_out_right(pe_valid_right[i][j]),
                    .valid_out_bottom(pe_valid_bottom[i][j]),
                    .result_valid(pe_result_valid[i][j]),
                    .result_sign(pe_result_sign[i][j]),
                    .result_exp(pe_result_exp[i][j]),
                    .result_mant(pe_result_mant[i][j])
                );

                conv_fp12_2bf16 bf16_conv (
                    .sign_in(pe_result_sign[i][j]),
                    .exp_in(pe_result_exp[i][j]),
                    .mant_in(pe_result_mant[i][j]),
                    .shared_scale_1(shared_scale_north[j]),
                    .shared_scale_2(shared_scale_west[i]),
                    .o_bf16(bf16_result[i*N + j])
                );
            end
        end
    endgenerate

    assign result_valid_out = pe_result_valid[N-1][N-1];

endmodule
