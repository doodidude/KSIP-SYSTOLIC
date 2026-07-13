`timescale 1ns / 1ps

// Self-checking 4x4 array testbench for top_fp12sr_systolic_mx.sv (plan
// §4d), covering all four target MX formats in one run. Vectors come from
// fp12sr_golden.py::gen_array_vectors, which computes each PE's expected
// final BF16 result via the already-verified pe_fp12sr_single_block (no
// global-cycle modeling needed there -- each PE's FSM only reacts to its
// own local valid_in_left&&valid_in_top events) followed by
// conv_fp12_2bf16. This testbench's own job is purely to reproduce correct
// systolic stagger timing: row i's west edge feed and column j's north
// edge feed must each be offset by 3*i / 3*j cycles (3 = pe_fp12sr.sv's
// own pass-through latency, data_right/bottom_s1..s3) so that row i's
// n-th element and column j's n-th element arrive at PE[i][j]
// simultaneously, matching the derivation in top_fp12sr_systolic_mx.sv's
// header comment.
module top_fp12sr_array_tb;

    localparam N = 4;
    localparam K = 32;
    localparam L_PASS = 3;   // pe_fp12sr.sv's own systolic pass-through latency

    logic clk;
    initial clk = 0;
    always #5 clk = ~clk;

    int e5m2_errors, e4m3_errors, e3m2_errors, e2m3_errors;

    check_one_array #(.exp_width(5), .man_width(2), .N(N), .k(K), .tag("e5m2"))
        u_e5m2 (.clk(clk), .errors_out(e5m2_errors));
    check_one_array #(.exp_width(4), .man_width(3), .N(N), .k(K), .tag("e4m3"))
        u_e4m3 (.clk(clk), .errors_out(e4m3_errors));
    check_one_array #(.exp_width(3), .man_width(2), .N(N), .k(K), .tag("e3m2"))
        u_e3m2 (.clk(clk), .errors_out(e3m2_errors));
    check_one_array #(.exp_width(2), .man_width(3), .N(N), .k(K), .tag("e2m3"))
        u_e2m3 (.clk(clk), .errors_out(e2m3_errors));

    initial begin
        wait (u_e5m2.done && u_e4m3.done && u_e3m2.done && u_e2m3.done);
        if (e5m2_errors + e4m3_errors + e3m2_errors + e2m3_errors == 0)
            $display("=== ALL TESTS PASSED (4 formats x %0d x %0d array) ===", N, N);
        else
            $display("=== %0d MISMATCHES FOUND ===",
                      e5m2_errors + e4m3_errors + e3m2_errors + e2m3_errors);
        $finish;
    end

endmodule


module check_one_array #(
    parameter exp_width = 4,
    parameter man_width = 3,
    parameter N = 4,
    parameter k = 32,
    parameter string tag = "e4m3",
    localparam bit_width = 1 + exp_width + man_width,
    localparam L_PASS = 3,
    localparam TIMEOUT_CYCLES = 2000
)(
    input  logic clk,
    output int errors_out
);

    logic rst;
    logic [N-1:0][bit_width-1:0] data_in_west;
    logic [N-1:0][bit_width-1:0] data_in_north;
    logic [N-1:0] data_valid_west;
    logic [N-1:0] data_valid_north;
    logic [N-1:0][7:0] shared_scale_west;
    logic [N-1:0][7:0] shared_scale_north;
    logic [N*N-1:0][15:0] bf16_result;
    logic result_valid_out;

    top_fp12sr_systolic_mx #(
        .exp_width(exp_width),
        .man_width(man_width),
        .k(k),
        .N(N)
    ) dut (
        .clk(clk),
        .rst(rst),
        .data_in_west(data_in_west),
        .data_in_north(data_in_north),
        .data_valid_west(data_valid_west),
        .data_valid_north(data_valid_north),
        .shared_scale_west(shared_scale_west),
        .shared_scale_north(shared_scale_north),
        .bf16_result(bf16_result),
        .result_valid_out(result_valid_out)
    );

    logic [bit_width-1:0] west_mem [N*k];
    logic [bit_width-1:0] north_mem [N*k];
    logic [7:0] scale_west_mem [N];
    logic [7:0] scale_north_mem [N];
    logic [15:0] gold_bf16_mem [N*N];

    logic done;
    int errors;
    int wait_count;
    int cycle_count;
    int idx;

    initial begin
        errors = 0;
        done   = 0;
        rst    = 1;
        cycle_count = 0;
        for (idx = 0; idx < N; idx++) begin
            data_valid_west[idx]  = 0;
            data_valid_north[idx] = 0;
            data_in_west[idx]     = '0;
            data_in_north[idx]    = '0;
        end

        $readmemh({"fp12sr_arr_", tag, "_west_in.hex"}, west_mem);
        $readmemh({"fp12sr_arr_", tag, "_north_in.hex"}, north_mem);
        $readmemh({"fp12sr_arr_", tag, "_scale_west_in.hex"}, scale_west_mem);
        $readmemh({"fp12sr_arr_", tag, "_scale_north_in.hex"}, scale_north_mem);
        $readmemh({"fp12sr_arr_", tag, "_bf16_gold.hex"}, gold_bf16_mem);

        for (idx = 0; idx < N; idx++) begin
            shared_scale_west[idx]  = scale_west_mem[idx];
            shared_scale_north[idx] = scale_north_mem[idx];
        end

        @(posedge clk); #1;
        rst = 0;

        // cycle_count = M = number of posedges crossed since the
        // reset-deassert edge. Element e of row/col idx (offset
        // off=L_PASS*idx) must be SET UP when M == off+e (immediately after
        // crossing posedge M, i.e. before the next @(posedge clk)) so that
        // it is CAPTURED at posedge M+1 = off+e+1 -- matching the single-PE
        // testbench's own convention (element 0 set up right after the
        // reset-deassert posedge, with no extra wait first).
        cycle_count = 0;
        wait_count = 0;
        do begin
            for (idx = 0; idx < N; idx++) begin
                data_valid_west[idx]  = (cycle_count >= L_PASS*idx) &&
                                         (cycle_count < k + L_PASS*idx);
                data_valid_north[idx] = (cycle_count >= L_PASS*idx) &&
                                         (cycle_count < k + L_PASS*idx);
                data_in_west[idx]  = data_valid_west[idx]  ? west_mem[idx*k + (cycle_count - L_PASS*idx)]  : '0;
                data_in_north[idx] = data_valid_north[idx] ? north_mem[idx*k + (cycle_count - L_PASS*idx)] : '0;
            end
            @(posedge clk); #1;
            cycle_count++;
            wait_count++;
        end while (!result_valid_out && wait_count < TIMEOUT_CYCLES);

        if (!result_valid_out) begin
            $display("[FAIL][%s] result_valid_out never asserted within %0d cycles",
                      tag, TIMEOUT_CYCLES);
            errors++;
        end else begin
            for (idx = 0; idx < N*N; idx++) begin
                if (bf16_result[idx] !== gold_bf16_mem[idx]) begin
                    $display("[FAIL][%s] PE%0d: bf16=%04h, expected %04h",
                              tag, idx, bf16_result[idx], gold_bf16_mem[idx]);
                    errors++;
                end
            end
        end

        if (errors == 0)
            $display("[PASS][%s] all %0d PE results matched", tag, N*N);
        errors_out = errors;
        done = 1;
    end

endmodule
