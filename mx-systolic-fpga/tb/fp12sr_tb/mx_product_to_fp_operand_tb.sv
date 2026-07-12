`timescale 1ns / 1ps

// Self-checking standalone testbench for mx_product_to_fp_operand.sv (S3
// bridge stage), covering all four target MX formats in one run:
// MXFP8_E5M2, MXFP8_E4M3, MXFP6_E3M2, MXFP6_E2M3. Each format gets its own
// DUT instance (parameters differ) driven by vectors from
// fp12sr_golden.py::gen_mx_product_to_fp_operand_vectors.
module mx_product_to_fp_operand_tb;

    localparam NUM_CASES = 200;

    logic clk;
    initial clk = 0;
    always #5 clk = ~clk;

    int total_errors;
    initial total_errors = 0;

    // ------------------------------------------------------------- e5m2
    check_one_format #(.exp_width(5), .man_width(2), .num_cases(NUM_CASES),
                        .tag("e5m2")) u_e5m2 (.clk(clk), .errors_out(e5m2_errors));
    // ------------------------------------------------------------- e4m3
    check_one_format #(.exp_width(4), .man_width(3), .num_cases(NUM_CASES),
                        .tag("e4m3")) u_e4m3 (.clk(clk), .errors_out(e4m3_errors));
    // ------------------------------------------------------------- e3m2
    check_one_format #(.exp_width(3), .man_width(2), .num_cases(NUM_CASES),
                        .tag("e3m2")) u_e3m2 (.clk(clk), .errors_out(e3m2_errors));
    // ------------------------------------------------------------- e2m3
    check_one_format #(.exp_width(2), .man_width(3), .num_cases(NUM_CASES),
                        .tag("e2m3")) u_e2m3 (.clk(clk), .errors_out(e2m3_errors));

    int e5m2_errors, e4m3_errors, e3m2_errors, e2m3_errors;

    initial begin
        wait (u_e5m2.done && u_e4m3.done && u_e3m2.done && u_e2m3.done);
        total_errors = e5m2_errors + e4m3_errors + e3m2_errors + e2m3_errors;
        if (total_errors == 0)
            $display("=== ALL TESTS PASSED (4 formats x %0d cases) ===", NUM_CASES);
        else
            $display("=== %0d MISMATCHES FOUND ===", total_errors);
        $finish;
    end

endmodule


module check_one_format #(
    parameter exp_width = 4,
    parameter man_width = 3,
    parameter num_cases = 200,
    parameter string tag = "e4m3",
    localparam fi_width     = man_width + 2,
    localparam fi_prd_width = 2 * fi_width,
    localparam frac_width   = 2 * man_width + 1,
    localparam fp12_exp_w   = 6
)(
    input logic clk,
    output int errors_out
);

    logic rst, valid_in, prd_sign;
    logic unsigned [fi_prd_width-1:0] u_prd;
    logic [exp_width-1:0] exp0_field, exp1_field;
    logic valid_out, sign_out;
    logic [fp12_exp_w-1:0] exp_out;
    logic [frac_width-1:0] frac_out;

    logic sign_mem [num_cases];
    logic [fi_prd_width-1:0] uprd_mem [num_cases];
    logic [exp_width-1:0] exp0_mem [num_cases];
    logic [exp_width-1:0] exp1_mem [num_cases];
    logic gold_sign_mem [num_cases];
    logic [fp12_exp_w-1:0] gold_exp_mem [num_cases];
    logic [frac_width-1:0] gold_frac_mem [num_cases];

    logic done;
    int errors;
    int i;

    mx_product_to_fp_operand #(
        .exp_width(exp_width),
        .man_width(man_width)
    ) dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .prd_sign(prd_sign),
        .u_prd(u_prd),
        .exp0_field(exp0_field),
        .exp1_field(exp1_field),
        .valid_out(valid_out),
        .sign_out(sign_out),
        .exp_out(exp_out),
        .frac_out(frac_out)
    );

    initial begin
        errors = 0;
        done   = 0;
        rst    = 1;
        valid_in = 0;
        prd_sign = 0;
        u_prd = '0;
        exp0_field = '0;
        exp1_field = '0;

        $readmemh({"fp12sr_s3_", tag, "_sign_in.hex"}, sign_mem);
        $readmemh({"fp12sr_s3_", tag, "_uprd_in.hex"}, uprd_mem);
        $readmemh({"fp12sr_s3_", tag, "_exp0_in.hex"}, exp0_mem);
        $readmemh({"fp12sr_s3_", tag, "_exp1_in.hex"}, exp1_mem);
        $readmemh({"fp12sr_s3_", tag, "_sign_gold.hex"}, gold_sign_mem);
        $readmemh({"fp12sr_s3_", tag, "_exp_gold.hex"}, gold_exp_mem);
        $readmemh({"fp12sr_s3_", tag, "_frac_gold.hex"}, gold_frac_mem);

        @(posedge clk);
        rst = 0;

        for (i = 0; i < num_cases; i++) begin
            prd_sign   = sign_mem[i];
            u_prd      = uprd_mem[i];
            exp0_field = exp0_mem[i];
            exp1_field = exp1_mem[i];
            valid_in   = 1;
            @(posedge clk);
            #1;
            if (sign_out !== gold_sign_mem[i] || exp_out !== gold_exp_mem[i] ||
                frac_out !== gold_frac_mem[i]) begin
                $display("[FAIL][%s] case %0d: sign=%b exp=%02h frac=%h, expected sign=%b exp=%02h frac=%h",
                          tag, i, sign_out, exp_out, frac_out,
                          gold_sign_mem[i], gold_exp_mem[i], gold_frac_mem[i]);
                errors++;
            end
        end

        if (errors == 0)
            $display("[PASS][%s] all %0d cases matched", tag, num_cases);
        errors_out = errors;
        done = 1;
    end

endmodule
