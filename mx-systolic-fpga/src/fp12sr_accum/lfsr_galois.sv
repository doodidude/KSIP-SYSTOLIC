`timescale 1ns / 1ps

// Galois-form LFSR PRNG. Default polynomial x^13 + x^4 + x^3 + x^1 + 1
// (taps {13,4,3,1}, a canonical maximal-length 13-bit polynomial per
// Xilinx XAPP052's LFSR tap table).
//
// Galois update (width_i bits, poly_mask bit (p-1) set for every tap p,
// including the top tap p=width_i):
//   fb  = reg[0];
//   reg = reg >> 1;
//   if (fb) reg = reg ^ poly_mask;
module lfsr_galois #(
    parameter width_i = 13,
    parameter [width_i-1:0] poly_mask = 13'h100D  // taps {13,4,3,1}
)(
    input  logic clk,
    input  logic rst,
    input  logic [width_i-1:0] seed_in,  // must be nonzero
    input  logic enable,
    output logic [width_i-1:0] rand_out
);

    logic [width_i-1:0] reg_q;
    logic fb;

    assign fb = reg_q[0];

    always_ff @(posedge clk) begin
        if (rst) begin
            reg_q <= seed_in;
        end else if (enable) begin
            reg_q <= fb ? ((reg_q >> 1) ^ poly_mask) : (reg_q >> 1);
        end
    end

    assign rand_out = reg_q;

endmodule
