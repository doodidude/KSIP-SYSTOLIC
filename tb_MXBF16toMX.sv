`timescale 1ns/1ps

module tb_MXBF16toMX;

    localparam k        = 32;
    localparam exp_size = 4;
    localparam man_size = 3;
    localparam sat      = 1;
    localparam NUM_BLOCKS = 4;

    logic [15:0]              bf16_mem   [NUM_BLOCKS*k];
    logic [exp_size+man_size:0] mxout_gold [NUM_BLOCKS*k];
    logic [7:0]                mxscale_gold [NUM_BLOCKS];

    logic clk;
    logic [15:0] BF16 [k];
    logic [exp_size+man_size:0] MXout [k];
    logic [7:0] MXscale;

    BF16toMX #(
        .k(k), .exp_size(exp_size), .man_size(man_size), .sat(sat)
    ) u0 (
        .clk(clk),
        .BF16(BF16),
        .MXout(MXout),
        .MXscale(MXscale)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    int errors;
    int blk;
    int i;

    initial begin
        errors = 0;

        $readmemh("enc_bf16_in.hex",      bf16_mem);
        $readmemh("enc_mxout_gold.hex",   mxout_gold);
        $readmemh("enc_mxscale_gold.hex", mxscale_gold);

        @(negedge clk);

        for (blk = 0; blk < NUM_BLOCKS; blk++) begin
            for (i = 0; i < k; i++) begin
                BF16[i] = bf16_mem[blk*k + i];
            end
            @(posedge clk);
            #1; 

            if (MXscale !== mxscale_gold[blk]) begin
                $display("[FAIL] block %0d: MXscale = %02h, expected %02h",
                          blk, MXscale, mxscale_gold[blk]);
                errors++;
            end else begin
                $display("[PASS] block %0d: MXscale = %02h", blk, MXscale);
            end

            for (i = 0; i < k; i++) begin
                if (MXout[i] !== mxout_gold[blk*k + i]) begin
                    $display("[FAIL] block %0d elem %0d: MXout = %02h, expected %02h",
                              blk, i, MXout[i], mxout_gold[blk*k + i]);
                    errors++;
                end
            end

            @(negedge clk);
        end

        if (errors == 0)
            $display("=== ALL TESTS PASSED (%0d blocks, %0d elements each) ===", NUM_BLOCKS, k);
        else
            $display("=== %0d MISMATCHES FOUND ===", errors);

        $finish;
    end

endmodule
