`timescale 1ns / 1ps

// Self-checking standalone testbench for sr_adder_fp12.sv (S4-S9 eager-SR
// adder), covering both distinct man_width configurations relevant to the
// four MX target formats: man_width=2 (fp8_e5m2, fp6_e3m2 -- CW=5, EXTRA=0)
// and man_width=3 (fp8_e4m3, fp6_e2m3 -- CW=7, EXTRA=2). Vectors come from
// fp12sr_golden.py::gen_sr_adder_fp12_vectors, covering swamping, exact
// cancellation, close-path, add-to-zero, the fp8_e5m2 saturating-clamp
// corner, and general far-path fill. Since the adder pipeline has L=6
// recurrence latency and each test case is an independent (A,B) pair (not
// a chained reduction), cases are driven back-to-back at one per cycle and
// results are checked L cycles later against the matching gold entry.
module sr_adder_fp12_tb;

    localparam NUM_CASES = 300;

    logic clk;
    initial clk = 0;
    always #5 clk = ~clk;

    int mw2_errors, mw3_errors;

    check_one_mw #(.man_width(2), .num_cases(NUM_CASES), .tag("mw2"))
        u_mw2 (.clk(clk), .errors_out(mw2_errors));
    check_one_mw #(.man_width(3), .num_cases(NUM_CASES), .tag("mw3"))
        u_mw3 (.clk(clk), .errors_out(mw3_errors));

    initial begin
        wait (u_mw2.done && u_mw3.done);
        if (mw2_errors + mw3_errors == 0)
            $display("=== ALL TESTS PASSED (2 configs x %0d cases) ===", NUM_CASES);
        else
            $display("=== %0d MISMATCHES FOUND ===", mw2_errors + mw3_errors);
        $finish;
    end

endmodule


module check_one_mw #(
    parameter man_width = 2,
    parameter num_cases = 300,
    parameter string tag = "mw2",
    localparam CW = 2 * man_width + 1,
    localparam L  = 5   // 6 register stages (v_s4..v_s8, output regs); the
                         // first stage samples the same iteration's input,
                         // so the testbench offset is stages-1, not stages
)(
    input  logic clk,
    output int errors_out
);

    logic rst;
    logic valid_in;
    logic sign_a;
    logic [5:0] exp_a;
    logic [4:0] mant_a;
    logic sign_b;
    logic [5:0] exp_b;
    logic [CW-1:0] frac_b;
    logic [12:0] rand_in;

    logic valid_out, sign_out;
    logic [5:0] exp_out;
    logic [4:0] mant_out;

    logic a_sign_mem [num_cases];
    logic [5:0] a_exp_mem [num_cases];
    logic [4:0] a_mant_mem [num_cases];
    logic b_sign_mem [num_cases];
    logic [5:0] b_exp_mem [num_cases];
    logic [CW-1:0] b_frac_mem [num_cases];
    logic [12:0] rand_mem [num_cases];
    logic gold_sign_mem [num_cases];
    logic [5:0] gold_exp_mem [num_cases];
    logic [4:0] gold_mant_mem [num_cases];

    logic done;
    int errors;
    int i;
    int g;

    sr_adder_fp12 #(
        .man_width(man_width)
    ) dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .sign_a(sign_a),
        .exp_a(exp_a),
        .mant_a(mant_a),
        .sign_b(sign_b),
        .exp_b(exp_b),
        .frac_b(frac_b),
        .rand_in(rand_in),
        .valid_out(valid_out),
        .sign_out(sign_out),
        .exp_out(exp_out),
        .mant_out(mant_out)
    );

    initial begin
        errors = 0;
        done   = 0;
        rst    = 1;
        valid_in = 0;
        sign_a = 0; exp_a = '0; mant_a = '0;
        sign_b = 0; exp_b = '0; frac_b = '0;
        rand_in = '0;

        $readmemh({"fp12sr_sr_", tag, "_a_sign_in.hex"}, a_sign_mem);
        $readmemh({"fp12sr_sr_", tag, "_a_exp_in.hex"}, a_exp_mem);
        $readmemh({"fp12sr_sr_", tag, "_a_mant_in.hex"}, a_mant_mem);
        $readmemh({"fp12sr_sr_", tag, "_b_sign_in.hex"}, b_sign_mem);
        $readmemh({"fp12sr_sr_", tag, "_b_exp_in.hex"}, b_exp_mem);
        $readmemh({"fp12sr_sr_", tag, "_b_frac_in.hex"}, b_frac_mem);
        $readmemh({"fp12sr_sr_", tag, "_rand_in.hex"}, rand_mem);
        $readmemh({"fp12sr_sr_", tag, "_sign_gold.hex"}, gold_sign_mem);
        $readmemh({"fp12sr_sr_", tag, "_exp_gold.hex"}, gold_exp_mem);
        $readmemh({"fp12sr_sr_", tag, "_mant_gold.hex"}, gold_mant_mem);

        @(posedge clk);
        rst = 0;

        for (i = 0; i < num_cases + L; i++) begin
            if (i < num_cases) begin
                sign_a   = a_sign_mem[i];
                exp_a    = a_exp_mem[i];
                mant_a   = a_mant_mem[i];
                sign_b   = b_sign_mem[i];
                exp_b    = b_exp_mem[i];
                frac_b   = b_frac_mem[i];
                rand_in  = rand_mem[i];
                valid_in = 1;
            end else begin
                valid_in = 0;
            end

            @(posedge clk);
            #1;

            if (i >= L) begin
                g = i - L;
                if (!valid_out) begin
                    $display("[FAIL][%s] case %0d: valid_out low, expected high", tag, g);
                    errors++;
                end else if (sign_out !== gold_sign_mem[g] || exp_out !== gold_exp_mem[g] ||
                             mant_out !== gold_mant_mem[g]) begin
                    $display("[FAIL][%s] case %0d: sign=%b exp=%02h mant=%02h, expected sign=%b exp=%02h mant=%02h",
                              tag, g, sign_out, exp_out, mant_out,
                              gold_sign_mem[g], gold_exp_mem[g], gold_mant_mem[g]);
                    errors++;
                end
            end
        end

        if (errors == 0)
            $display("[PASS][%s] all %0d cases matched", tag, num_cases);
        errors_out = errors;
        done = 1;
    end

endmodule
