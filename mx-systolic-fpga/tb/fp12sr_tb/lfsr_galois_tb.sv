`timescale 1ns / 1ps

// Self-checking testbench for lfsr_galois.sv (FP12-SR accumulator PRNG, §4a).
// Loads a seed + a cycle-by-cycle golden rand_out sequence produced by the
// bit-exact Python replica (fp12sr_golden.py::lfsr_galois_sequence), then
// clocks the RTL LFSR and checks every cycle with a 4-state !== compare.
module lfsr_galois_tb;

    localparam WIDTH      = 13;
    localparam NUM_CYCLES = 256;
    localparam HEXW       = (WIDTH + 3) / 4;

    logic clk;
    logic rst;
    logic enable;
    logic [WIDTH-1:0] seed_in;
    logic [WIDTH-1:0] rand_out;

    logic [WIDTH-1:0] seed_mem [1];
    logic [WIDTH-1:0] gold_mem [NUM_CYCLES+1];

    lfsr_galois #(
        .width_i(WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .seed_in(seed_in),
        .enable(enable),
        .rand_out(rand_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    int errors;
    int i;

    initial begin
        errors = 0;

        $readmemh("fp12sr_lfsr_seed.hex", seed_mem);
        $readmemh("fp12sr_lfsr_gold.hex", gold_mem);

        seed_in = seed_mem[0];
        rst     = 1;
        enable  = 0;

        @(posedge clk);
        #1;
        if (rand_out !== gold_mem[0]) begin
            $display("[FAIL] cycle 0 (post-reset): rand_out = %04h, expected %04h",
                      rand_out, gold_mem[0]);
            errors++;
        end else begin
            $display("[PASS] cycle 0 (post-reset): rand_out = %04h", rand_out);
        end

        rst    = 0;
        enable = 1;

        for (i = 1; i <= NUM_CYCLES; i++) begin
            @(posedge clk);
            #1;
            if (rand_out !== gold_mem[i]) begin
                $display("[FAIL] cycle %0d: rand_out = %04h, expected %04h",
                          i, rand_out, gold_mem[i]);
                errors++;
            end
        end

        if (errors == 0)
            $display("=== ALL TESTS PASSED (%0d cycles) ===", NUM_CYCLES + 1);
        else
            $display("=== %0d MISMATCHES FOUND ===", errors);

        $finish;
    end

endmodule
