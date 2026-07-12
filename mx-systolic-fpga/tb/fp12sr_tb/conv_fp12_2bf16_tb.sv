`timescale 1ns / 1ps

// Self-checking standalone testbench for conv_fp12_2bf16.sv -- the final
// FP12->BF16 widening stage. Purely combinational DUT, so vectors are
// driven and checked in the same cycle (one #1 settle delay after each
// assignment, no pipeline latency to account for). Vectors come from
// fp12sr_golden.py::gen_conv_fp12_2bf16_vectors, covering both the general
// (exp_in != 0) case across random shared-scale pairs and the exact-zero
// (exp_in == 0) special case.
module conv_fp12_2bf16_tb;

    localparam NUM_CASES = 200;

    logic sign_in;
    logic [5:0] exp_in;
    logic [4:0] mant_in;
    logic [7:0] shared_scale_1, shared_scale_2;
    logic [15:0] o_bf16;

    conv_fp12_2bf16 dut (
        .sign_in(sign_in),
        .exp_in(exp_in),
        .mant_in(mant_in),
        .shared_scale_1(shared_scale_1),
        .shared_scale_2(shared_scale_2),
        .o_bf16(o_bf16)
    );

    logic sign_mem [NUM_CASES];
    logic [5:0] exp_mem [NUM_CASES];
    logic [4:0] mant_mem [NUM_CASES];
    logic [7:0] ss1_mem [NUM_CASES];
    logic [7:0] ss2_mem [NUM_CASES];
    logic [15:0] gold_bf16_mem [NUM_CASES];

    int errors;
    int i;

    initial begin
        errors = 0;

        $readmemh("fp12sr_conv_sign_in.hex", sign_mem);
        $readmemh("fp12sr_conv_exp_in.hex", exp_mem);
        $readmemh("fp12sr_conv_mant_in.hex", mant_mem);
        $readmemh("fp12sr_conv_ss1_in.hex", ss1_mem);
        $readmemh("fp12sr_conv_ss2_in.hex", ss2_mem);
        $readmemh("fp12sr_conv_bf16_gold.hex", gold_bf16_mem);

        for (i = 0; i < NUM_CASES; i++) begin
            sign_in        = sign_mem[i];
            exp_in         = exp_mem[i];
            mant_in        = mant_mem[i];
            shared_scale_1 = ss1_mem[i];
            shared_scale_2 = ss2_mem[i];
            #1;
            if (o_bf16 !== gold_bf16_mem[i]) begin
                $display("[FAIL] case %0d: sign=%b exp=%02h mant=%02h ss1=%02h ss2=%02h o_bf16=%04h, expected %04h",
                          i, sign_in, exp_in, mant_in, shared_scale_1, shared_scale_2,
                          o_bf16, gold_bf16_mem[i]);
                errors++;
            end
        end

        if (errors == 0)
            $display("=== ALL TESTS PASSED (%0d cases) ===", NUM_CASES);
        else
            $display("=== %0d MISMATCHES FOUND ===", errors);
        $finish;
    end

endmodule
