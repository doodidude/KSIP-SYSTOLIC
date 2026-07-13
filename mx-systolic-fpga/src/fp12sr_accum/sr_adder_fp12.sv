`timescale 1ns / 1ps

// S4-S9: Ben-Ali eager-SR adder (plan blocks 1-10, arXiv:2404.14010).
// Six-stage pipeline (L=6 recurrence latency, plan §1.1.1) implementing
// exactly the algorithm bit-exact with fp12sr_golden.py::sr_adder_fp12 --
// see that function's docstring for the full block-by-block derivation.
// One stage per Ben-Ali block group:
//   S4  Block 1        Exponent diff / Swap (magnitude compare -> X,Y)
//   S5  Blocks 2/3     Shift, 2's Complement
//   S6  Blocks 4/5/6   Fanout, Main adder, Sticky Round
//   S7  Block 7        LZD/Shift (close path) + Normalization (far path), parallel
//   S8  Block 8/9      Trapezoid mux + Round Correction
//   S9  Block 10       Second-stage EXTRA-bit round-off + Increment/finalize
//
// Operand A is the FP12 E6M5 lane register (mant_a: 5 explicit bits,
// hidden bit implied 1 iff exp_a != 0). Operand B is the S3 bridge-stage
// "increment" (frac_b: CW = 2*man_width+1 explicit bits -- 5 for
// man_width=2, 7 for man_width=3 -- hidden bit implied 1 iff exp_b != 0;
// S3 guarantees frac_b == 0 whenever exp_b == 0, so exp_b==0 alone is a
// safe "adding zero" bypass test). rand_in is one 13-bit LFSR draw
// consumed by this single add: the low 11 bits (r-2, format-independent)
// feed the S6 sticky-round tail; the top 2 bits (13-11 spare) feed the S9
// second-stage round-off that trims real EXTRA precision (man_width==3
// formats only) down to FP12's native 5-bit mantissa -- a no-op when
// EXTRA==0 (man_width==2 formats, where B's own frac_width already equals
// FP12's mantissa width exactly).


module sr_adder_fp12 #(
    parameter man_width = 2,
    localparam CW          = 2 * man_width + 1,
    localparam FP12_MANT_W = 5,
    localparam FP12_SIG_W  = 6, // significand = hidden bit + 5 mant 
    localparam EXTRA       = CW - FP12_MANT_W, // 0 or 2 
    localparam TAIL_W      = 11, // r - 2
    localparam REG_W       = (CW + 1) + TAIL_W, // total alignment register width 
    localparam SH_W        = $clog2(REG_W + 1), // bits to encode shift amount 
    localparam EXP_W       = 8   // signed working exponent width (guard bits, matches S3's convention)
)(
    input  logic clk,
    input  logic rst,
    input  logic valid_in,
    input  logic sign_a,
    input  logic [5:0] exp_a,
    input  logic [FP12_MANT_W-1:0] mant_a,
    input  logic sign_b,
    input  logic [5:0] exp_b,
    input  logic [CW-1:0] frac_b,
    input  logic [12:0] rand_in,

    output logic valid_out,
    output logic sign_out,
    output logic [5:0] exp_out,
    output logic [FP12_MANT_W-1:0] mant_out
);

    // Signals piped unchanged alongside the arithmetic, one register per stage.
    typedef struct packed {
        logic bypass;                       // exp_b == 0 -> pass A straight through
        logic bypass_sign;                  // A's fields, saved for the bypass path
        logic [5:0] bypass_exp;
        logic [FP12_MANT_W-1:0] bypass_mant;
        logic [12:0] rand13;                // PRNG 
        logic result_sign;                  // = X's sign, fixed at S4, used at S9
    } pass_t;

    // ================================================================ S4
    // Block 1: Exponent diff / Swap -- magnitude compare picks X (larger), Y (smaller)

    logic [CW-1:0] a_full_c;
    logic [CW:0]   a_sig_c, b_sig_c;
    logic signed [EXP_W-1:0] a_exp_c, b_exp_c;
    logic a_ge_b_c;
    logic x_sign_c, y_sign_c;
    logic signed [EXP_W-1:0] x_exp_c, y_exp_c;
    logic [CW:0] x_full_c, y_full_c;
    logic signed [EXP_W-1:0] shift_diff_c;
    logic [SH_W-1:0] shift_amt_c;
    logic op_sub_c;

    always_comb begin
        a_full_c = mant_a << EXTRA; // aligning operand A with B 
        a_sig_c  = (exp_a != 0) ? {1'b1, a_full_c} : '0; // restore hidden bit or flush to zero
        a_exp_c  = (exp_a != 0) ? $signed({2'b0, exp_a}) : '0; // widen exponent for safety 
        b_sig_c  = {1'b1, frac_b}; // doesnt need zero case as bypass covers it 
        b_exp_c  = $signed({2'b0, exp_b});

        a_ge_b_c = (a_exp_c > b_exp_c) || ((a_exp_c == b_exp_c) && (a_sig_c >= b_sig_c)); // magnitude comparison

        if (a_ge_b_c) begin // swapping
            x_sign_c = sign_a; x_exp_c = a_exp_c; x_full_c = a_sig_c;
            y_sign_c = sign_b; y_exp_c = b_exp_c; y_full_c = b_sig_c;
        end else begin
            x_sign_c = sign_b; x_exp_c = b_exp_c; x_full_c = b_sig_c;
            y_sign_c = sign_a; y_exp_c = a_exp_c; y_full_c = a_sig_c;
        end

        shift_diff_c = x_exp_c - y_exp_c;
        shift_amt_c  = (shift_diff_c > REG_W) ? REG_W[SH_W-1:0] : shift_diff_c[SH_W-1:0];  // how far to push right to align with X 
        op_sub_c     = (x_sign_c != y_sign_c); // effective subtract if sign differ 
    end

    logic signed [EXP_W-1:0] x_exp_s4;
    logic [CW:0] x_full_s4, y_full_s4;
    logic [SH_W-1:0] shift_amt_s4;
    logic close_path_s4;
    pass_t pass_s4;

    // ================================================================ pipeline valid chain
    logic v_s4, v_s5, v_s6, v_s7, v_s8;

    always_ff @(posedge clk) begin
        if (rst) begin
            v_s4 <= 1'b0;
            x_exp_s4 <= '0; x_full_s4 <= '0; y_full_s4 <= '0;
            shift_amt_s4 <= '0; close_path_s4 <= 1'b0;
            pass_s4 <= '0;
        end else begin
            v_s4 <= valid_in;
            x_exp_s4  <= x_exp_c;
            x_full_s4 <= x_full_c;
            y_full_s4 <= y_full_c;
            shift_amt_s4  <= shift_amt_c;
            close_path_s4 <= op_sub_c && (shift_amt_c <= 1); // close flag, checks if its subtracting and exponents are within 1 of eachother 
            pass_s4.bypass      <= (exp_b == 0);
            pass_s4.bypass_sign <= sign_a;
            pass_s4.bypass_exp  <= exp_a;
            pass_s4.bypass_mant <= mant_a;
            pass_s4.rand13      <= rand_in;
            pass_s4.result_sign <= x_sign_c;
        end
    end

    // ================================================================ S5
    // Blocks 2/3: Shift, 2's Complement (placed after shift)

    logic [REG_W-1:0] y_ext_c, y_shifted_c;

    always_comb begin
        y_ext_c     = {y_full_s4, {TAIL_W{1'b0}}}; // pad Y to REG_w bits 
        y_shifted_c = y_ext_c >> shift_amt_s4; // right shift to align with X 
    end

    logic op_sub_s4_q;

    always_ff @(posedge clk) begin
        if (!rst) op_sub_s4_q <= op_sub_c;
    end

    logic [REG_W-1:0] y_shifted_final_c;
    always_comb begin
        y_shifted_final_c = op_sub_s4_q ? ((~y_shifted_c) + 1'b1) : y_shifted_c; // 2s compliment if subtracting
    end

    logic signed [EXP_W-1:0] x_exp_s5;
    logic [CW:0] x_full_s5;
    logic [REG_W-1:0] y_shifted_s5;
    logic close_path_s5;
    pass_t pass_s5;

    always_ff @(posedge clk) begin
        if (rst) begin
            v_s5 <= 1'b0;
            x_exp_s5 <= '0; x_full_s5 <= '0; y_shifted_s5 <= '0; close_path_s5 <= 1'b0;
            pass_s5 <= '0;
        end else begin
            v_s5 <= v_s4;
            x_exp_s5     <= x_exp_s4;
            x_full_s5    <= x_full_s4;
            y_shifted_s5 <= y_shifted_final_c;
            close_path_s5 <= close_path_s4;
            pass_s5 <= pass_s4;
        end
    end

    // ================================================================ S6
    // Blocks 4/5/6: Fanout, Main adder, Sticky Round (parallel)

    logic [CW:0]   y_main_c;
    logic [TAIL_W-1:0] y_tail_c, rand_tail_c;
    logic [CW+1:0] main_sum_c;
    logic [TAIL_W:0] stick_sum_c;
    logic s1_c, s2_c;

    always_comb begin
        y_main_c = y_shifted_s5[REG_W-1 -: (CW+1)];
        y_tail_c = y_shifted_s5[TAIL_W-1:0];
        main_sum_c = {1'b0, x_full_s5} + {1'b0, y_main_c}; // add leading zero to catch carry 
        rand_tail_c = pass_s5.rand13[TAIL_W-1:0];
        stick_sum_c = {1'b0, y_tail_c} + {1'b0, rand_tail_c}; 
        s1_c = stick_sum_c[TAIL_W];
        s2_c = stick_sum_c[TAIL_W-1]; // S1 and S2 from MSBs
    end

    logic signed [EXP_W-1:0] x_exp_s6;
    logic [CW+1:0] main_sum_s6;
    logic s1_s6, s2_s6, close_path_s6;
    pass_t pass_s6;

    always_ff @(posedge clk) begin
        if (rst) begin
            v_s6 <= 1'b0;
            x_exp_s6 <= '0; main_sum_s6 <= '0; s1_s6 <= 1'b0; s2_s6 <= 1'b0; close_path_s6 <= 1'b0;
            pass_s6 <= '0;
        end else begin
            v_s6 <= v_s5;
            x_exp_s6 <= x_exp_s5;
            main_sum_s6 <= main_sum_c;
            s1_s6 <= s1_c;
            s2_s6 <= s2_c;
            close_path_s6 <= close_path_s5;
            pass_s6 <= pass_s5;
        end
    end

    // ================================================================ S7
    // Block 7: LZD/Shift (close path) + Normalization (far path), parallel

    logic [CW:0] win_c;
    logic [$clog2(CW+2)-1:0] lz_c;
    logic zero_result_c;
    logic [CW:0] close_norm_sig_c;
    logic signed [EXP_W-1:0] close_norm_exp_c;
    logic carry_out_c;
    logic [CW:0] far_norm_sig_c;
    logic signed [EXP_W-1:0] far_norm_exp_c;
    logic far_sel_s_c;

    always_comb begin
        win_c = main_sum_s6[CW:0]; // strip overflow gaurd bit, keep 8 bits
        zero_result_c = (win_c == '0);
        lz_c = 0;
        for (int b = CW; b >= 0; b--) begin // leading zero counter
            if (win_c[b]) begin
                lz_c = CW - b;
                break;
            end
        end
        close_norm_sig_c = win_c << lz_c; // shift to push the first 1 into MSB (Normalize)
        close_norm_exp_c = x_exp_s6 - $signed({{(EXP_W-$bits(lz_c)){1'b0}}, lz_c}); // decrement exponent to compensate 

        carry_out_c = main_sum_s6[CW+1];
        if (carry_out_c) begin
            far_norm_sig_c = main_sum_s6[CW+1:1]; // right shift by 1
            far_norm_exp_c = x_exp_s6 + 1; // compensate exponent 
            far_sel_s_c    = s2_s6; // shifted round point -> use s2 
        end else begin
            far_norm_sig_c = main_sum_s6[CW:0]; // no shift 
            far_norm_exp_c = x_exp_s6; // exponent unchanged 
            far_sel_s_c    = s1_s6; // normal round point -> use s1 
        end
    end

    logic [CW:0] close_norm_sig_s7, far_norm_sig_s7;
    logic signed [EXP_W-1:0] close_norm_exp_s7, far_norm_exp_s7;
    logic far_sel_s_s7, zero_result_s7, close_path_s7;
    pass_t pass_s7;

    always_ff @(posedge clk) begin
        if (rst) begin
            v_s7 <= 1'b0;
            close_norm_sig_s7 <= '0; far_norm_sig_s7 <= '0;
            close_norm_exp_s7 <= '0; far_norm_exp_s7 <= '0;
            far_sel_s_s7 <= 1'b0; zero_result_s7 <= 1'b0; close_path_s7 <= 1'b0;
            pass_s7 <= '0;
        end else begin
            v_s7 <= v_s6;
            close_norm_sig_s7 <= close_norm_sig_c;
            far_norm_sig_s7   <= far_norm_sig_c;
            close_norm_exp_s7 <= close_norm_exp_c;
            far_norm_exp_s7   <= far_norm_exp_c;
            far_sel_s_s7   <= far_sel_s_c;
            zero_result_s7 <= zero_result_c;
            close_path_s7  <= close_path_s6;
            pass_s7 <= pass_s6;
        end
    end

    // ================================================================ S8
    // Blocks 8/9: Trapezoid mux + Round Correction
    logic [CW+1:0] corrected_s8;
    logic signed [EXP_W-1:0] corrected_exp_s8;
    logic zero_result_s8;
    pass_t pass_s8;

    logic [CW+1:0] corrected_pre_c;
    logic signed [EXP_W-1:0] corrected_exp_pre_c;
    always_comb begin
        if (close_path_s7) begin
            corrected_pre_c     = {1'b0, close_norm_sig_s7};
            corrected_exp_pre_c = close_norm_exp_s7;
        end else begin
            corrected_pre_c     = {1'b0, far_norm_sig_s7} + {{(CW+1){1'b0}}, far_sel_s_s7}; 
            corrected_exp_pre_c = far_norm_exp_s7;
        end
    end

    logic [CW+1:0] corrected_final_c;
    logic signed [EXP_W-1:0] corrected_exp_final_c;
    always_comb begin
        if (corrected_pre_c[CW+1]) begin // after injecting carry, check if overflowed 
            corrected_final_c     = corrected_pre_c >> 1;
            corrected_exp_final_c = corrected_exp_pre_c + 1;
        end else begin
            corrected_final_c     = corrected_pre_c;
            corrected_exp_final_c = corrected_exp_pre_c;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            v_s8 <= 1'b0;
            corrected_s8 <= '0; corrected_exp_s8 <= '0; zero_result_s8 <= 1'b0;
            pass_s8 <= '0;
        end else begin
            v_s8 <= v_s7;
            corrected_s8     <= corrected_final_c;
            corrected_exp_s8 <= corrected_exp_final_c;
            zero_result_s8   <= close_path_s7 && zero_result_s7;
            pass_s8 <= pass_s7;
        end
    end

    // ================================================================ S9
    // Second-stage EXTRA-bit round-off + Block 10: Increment / finalize
    // extra_bits_c/spare_rand_c are sized to EXTRA's max possible value (2,
    // for man_width==3 formats) rather than [EXTRA-1:0] directly, since
    // EXTRA==0 (man_width==2 formats) would otherwise elaborate a
    // zero-/negative-width vector.

    localparam EXTRA_W = (EXTRA > 0) ? EXTRA : 1;
    logic [FP12_SIG_W-1:0] native_top_c;
    logic [EXTRA_W-1:0] extra_bits_c, spare_rand_c;
    logic [EXTRA_W:0] round_sum_c;
    logic carry_fin_c;
    logic [FP12_SIG_W:0] native_sig_c;
    logic signed [EXP_W-1:0] final_exp_c;
    logic [FP12_MANT_W-1:0] mant_final_c;

    always_comb begin
        // corrected_s8 is [CW+1:0] -- bit CW+1 is an always-0 overflow-guard
        // bit (any carry out of the CW+1-bit window was already renormalized
        // away in S8), so the real significand lives in bits [CW:0]. Slicing
        // from CW+1 instead of CW would drop the true LSB and open a gap with
        // extra_bits_c below.

        native_top_c = corrected_s8[CW -: FP12_SIG_W]; // top 6 bits of corrected significand 
        if (EXTRA > 0) begin
            extra_bits_c = corrected_s8[EXTRA_W-1:0]; // bits[1:0] the extra 2 precision bits
            spare_rand_c = pass_s8.rand13[12 -: EXTRA_W]; // top 2 PRNG bits 
        end else begin
            extra_bits_c = '0;
            spare_rand_c = '0;
        end
        round_sum_c = {1'b0, extra_bits_c} + {1'b0, spare_rand_c}; // pad zero to catch carry 
        carry_fin_c = (EXTRA > 0) ? round_sum_c[EXTRA_W] : 1'b0; // second stage SR carry 
        native_sig_c = {1'b0, native_top_c} + carry_fin_c;

        if (native_sig_c[FP12_SIG_W]) begin // bit 6 set -> overflow 
            final_exp_c  = corrected_exp_s8 + 1;
            mant_final_c = native_sig_c[FP12_SIG_W:1][FP12_MANT_W-1:0]; // bits [6:1] then [4:0]
        end else begin
            final_exp_c  = corrected_exp_s8;
            mant_final_c = native_sig_c[FP12_MANT_W-1:0]; // bits [4:0]
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_out <= 1'b0;
            sign_out  <= 1'b0;
            exp_out   <= '0;
            mant_out  <= '0;
        end else begin
            valid_out <= v_s8;
            if (pass_s8.bypass) begin // if bypassed, just output A unchanged 
                sign_out <= pass_s8.bypass_sign;
                exp_out  <= pass_s8.bypass_exp;
                mant_out <= pass_s8.bypass_mant;
            end else if (zero_result_s8) begin // close path subtraction gave zero, output zero 
                sign_out <= 1'b0;
                exp_out  <= '0;
                mant_out <= '0;
            end else if (final_exp_c > 63) begin // exponent too large, saturate 
                sign_out <= pass_s8.result_sign;
                exp_out  <= 6'd63;
                mant_out <= {FP12_MANT_W{1'b1}};
            end else if (final_exp_c < 0) begin // exponent negative -> flush to zero 
                sign_out <= 1'b0;
                exp_out  <= '0;
                mant_out <= '0;
            end else begin // normal operation 
                sign_out <= pass_s8.result_sign;
                exp_out  <= final_exp_c[5:0];
                mant_out <= mant_final_c;
            end
        end
    end

endmodule
