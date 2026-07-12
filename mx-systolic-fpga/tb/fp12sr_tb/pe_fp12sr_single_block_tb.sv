`timescale 1ns / 1ps

// Self-checking standalone testbench for pe_fp12sr.sv (plan §4c) -- the
// hardest verification gate: one PE instance per MX format, each fed
// NUM_BLOCKS independent k=32-element blocks (west/north pairs, one per
// cycle, back-to-back), with a reset pulse between blocks to re-arm the
// intake counter / lane register file / combine FSM. Waits for the DUT's
// own result_valid pulse (event-driven, not a hardcoded cycle count) and
// checks the final combined FP12 value against
// fp12sr_golden.py::pe_fp12sr_single_block, which reproduces the exact
// posedge-by-posedge dispatch/combine timing (and therefore LFSR draw
// order) that pe_fp12sr.sv's RTL implements.
module pe_fp12sr_single_block_tb;

    localparam NUM_BLOCKS = 20;
    localparam K = 32;

    logic clk;
    initial clk = 0;
    always #5 clk = ~clk;

    int e5m2_errors, e4m3_errors, e3m2_errors, e2m3_errors;

    check_one_pe #(.exp_width(5), .man_width(2), .num_blocks(NUM_BLOCKS), .k(K), .tag("e5m2"))
        u_e5m2 (.clk(clk), .errors_out(e5m2_errors));
    check_one_pe #(.exp_width(4), .man_width(3), .num_blocks(NUM_BLOCKS), .k(K), .tag("e4m3"))
        u_e4m3 (.clk(clk), .errors_out(e4m3_errors));
    check_one_pe #(.exp_width(3), .man_width(2), .num_blocks(NUM_BLOCKS), .k(K), .tag("e3m2"))
        u_e3m2 (.clk(clk), .errors_out(e3m2_errors));
    check_one_pe #(.exp_width(2), .man_width(3), .num_blocks(NUM_BLOCKS), .k(K), .tag("e2m3"))
        u_e2m3 (.clk(clk), .errors_out(e2m3_errors));

    initial begin
        wait (u_e5m2.done && u_e4m3.done && u_e3m2.done && u_e2m3.done);
        if (e5m2_errors + e4m3_errors + e3m2_errors + e2m3_errors == 0)
            $display("=== ALL TESTS PASSED (4 formats x %0d blocks x %0d elements) ===", NUM_BLOCKS, K);
        else
            $display("=== %0d MISMATCHES FOUND ===", e5m2_errors + e4m3_errors + e3m2_errors + e2m3_errors);
        $finish;
    end

endmodule


module check_one_pe #(
    parameter exp_width = 4,
    parameter man_width = 3,
    parameter num_blocks = 20,
    parameter k = 32,
    parameter string tag = "e4m3",
    localparam bit_width = 1 + exp_width + man_width,
    localparam TIMEOUT_CYCLES = 500
)(
    input  logic clk,
    output int errors_out
);

    logic rst;
    logic [bit_width-1:0] data_in_left, data_in_top;
    logic valid_in_left, valid_in_top;

    logic result_valid, result_sign;
    logic [5:0] result_exp;
    logic [4:0] result_mant;

    logic [bit_width-1:0] west_mem [num_blocks*k];
    logic [bit_width-1:0] north_mem [num_blocks*k];
    logic gold_sign_mem [num_blocks];
    logic [5:0] gold_exp_mem [num_blocks];
    logic [4:0] gold_mant_mem [num_blocks];

    logic done;
    int errors;
    int b, e, wait_count;

    pe_fp12sr #(
        .exp_width(exp_width),
        .man_width(man_width),
        .k(k),
        .pe_id(0)
    ) dut (
        .clk(clk),
        .rst(rst),
        .data_in_left(data_in_left),
        .data_in_top(data_in_top),
        .valid_in_left(valid_in_left),
        .valid_in_top(valid_in_top),
        .data_out_right(),
        .data_out_bottom(),
        .valid_out_right(),
        .valid_out_bottom(),
        .result_valid(result_valid),
        .result_sign(result_sign),
        .result_exp(result_exp),
        .result_mant(result_mant)
    );

    initial begin
        errors = 0;
        done   = 0;
        rst    = 1;
        valid_in_left = 0; valid_in_top = 0;
        data_in_left = '0; data_in_top = '0;

        $readmemh({"fp12sr_pe_", tag, "_west_in.hex"}, west_mem);
        $readmemh({"fp12sr_pe_", tag, "_north_in.hex"}, north_mem);
        $readmemh({"fp12sr_pe_", tag, "_sign_gold.hex"}, gold_sign_mem);
        $readmemh({"fp12sr_pe_", tag, "_exp_gold.hex"}, gold_exp_mem);
        $readmemh({"fp12sr_pe_", tag, "_mant_gold.hex"}, gold_mant_mem);

        for (b = 0; b < num_blocks; b++) begin
            // reset pulse re-arms intake counter, lane regfile, combine FSM,
            // and the per-PE LFSR (matching seed every block, same as the
            // golden model's per-block re-seed)
            rst = 1;
            @(posedge clk); #1;
            rst = 0;

            for (e = 0; e < k; e++) begin
                data_in_left = west_mem[b*k + e];
                data_in_top  = north_mem[b*k + e];
                valid_in_left = 1;
                valid_in_top  = 1;
                @(posedge clk); #1;
            end
            valid_in_left = 0;
            valid_in_top  = 0;

            wait_count = 0;
            while (!result_valid && wait_count < TIMEOUT_CYCLES) begin
                @(posedge clk); #1;
                wait_count++;
            end

            if (!result_valid) begin
                $display("[FAIL][%s] block %0d: result_valid never asserted within %0d cycles",
                          tag, b, TIMEOUT_CYCLES);
                errors++;
            end else if (result_sign !== gold_sign_mem[b] || result_exp !== gold_exp_mem[b] ||
                         result_mant !== gold_mant_mem[b]) begin
                $display("[FAIL][%s] block %0d: sign=%b exp=%02h mant=%02h, expected sign=%b exp=%02h mant=%02h",
                          tag, b, result_sign, result_exp, result_mant,
                          gold_sign_mem[b], gold_exp_mem[b], gold_mant_mem[b]);
                errors++;
            end
        end

        if (errors == 0)
            $display("[PASS][%s] all %0d blocks matched", tag, num_blocks);
        errors_out = errors;
        done = 1;
    end

endmodule
