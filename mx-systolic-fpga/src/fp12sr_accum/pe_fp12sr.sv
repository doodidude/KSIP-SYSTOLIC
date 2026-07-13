`timescale 1ns / 1ps

// FP12 eager-SR accumulator PE (plan §1). S1/S2 are the reused exact-design
// decode + exact multiply (pe_exact_4s.sv stages 1-2), modified only to
// flush the mantissa to 0 whenever an operand's exponent field is 0
// (Sub-OFF denormal flush, decided in the plan) instead of the exact
// design's own IEEE-subnormal encoding. S3 is the existing
// mx_product_to_fp_operand bridge module (already verified standalone).
// S4-S9 is a single SHARED sr_adder_fp12 instance, time-multiplexed via
// round-robin dispatch across a NUM_LANES-entry FP12 lane register file,
// followed by a serial combine-tree FSM that reduces the lanes to one
// final FP12 value once all k block elements have landed.
//
// NUM_LANES = 7, not 6: sr_adder_fp12 has 6 register stages (an op
// presented at posedge n produces valid_out/sign_out/... starting at
// posedge n+5). Writing that result into a lane register is ITSELF a
// synchronous register (this module's own always_ff), which can only
// capture the fresh value at posedge n+6 (its input must be stable through
// the interval ending at that edge). So a lane isn't safely reusable until
// posedge n+7. With 1 dispatch/cycle round-robin, consecutive dispatches
// to the same lane are exactly NUM_LANES cycles apart -- so NUM_LANES must
// be >= 7, confirming (and correcting) the plan's own flagged margin
// caveat in §1.3 ("If the dispatch mux itself costs a pipeline cycle, P
// must become 7"). The same 7-cycle-minimum-gap argument applies to the
// combine tree's serial adds (each depends on the previous add's actual
// result, not just lane reuse), so the FSM waits for each combine add's
// valid_out before issuing the next one, rather than assuming a fixed
// cycle count.

module pe_fp12sr #(
    parameter exp_width = 4,
    parameter man_width = 3,
    parameter bit_width = 1 + exp_width + man_width,
    parameter k = 32, // MX Block size
    parameter pe_id = 0, // unique ID, used to seed LFSR
    parameter logic [12:0] SEED_BASE = 13'h1ACE, // constant for PRNG
    localparam fi_width     = man_width + 2, // padded operand width for multiply
    localparam frac_width   = 2 * man_width + 1,   // == sr_adder_fp12's CW
    localparam NUM_LANES    = 7, // lane count = adder latency + 1 
    localparam LANE_W       = $clog2(NUM_LANES), // bit width needed to index the lane registers 
    localparam FP12_MANT_W  = 5, 
    localparam EXTRA        = frac_width - FP12_MANT_W, // extra precision bits beyond FP12
    localparam IDX_W        = $clog2(k + 1) // bit width needed to count elements 0 to k 
)(
    input  logic clk,
    input  logic rst,

    input  logic [bit_width-1:0] data_in_left, // MX element from west 
    input  logic [bit_width-1:0] data_in_top, // MX element from north
    input  logic valid_in_left,
    input  logic valid_in_top, // valid signals

    output logic [bit_width-1:0] data_out_right, 
    output logic [bit_width-1:0] data_out_bottom,
    output logic valid_out_right,
    output logic valid_out_bottom, // valid signals 

    output logic result_valid, // pulses for one cycle when done 
    output logic result_sign, // final FP12 results 
    output logic [5:0] result_exp,
    output logic [FP12_MANT_W-1:0] result_mant
);

    // ------------------------------------------------- systolic pass-through

    logic [bit_width-1:0] data_right_s1, data_right_s2, data_right_s3;
    logic [bit_width-1:0] data_bottom_s1, data_bottom_s2, data_bottom_s3;
    logic valid_right_s1, valid_right_s2, valid_right_s3;
    logic valid_bottom_s1, valid_bottom_s2, valid_bottom_s3; // registers for data (3 stage pipeline, 1 for data and 1 for valid signals)

    always_ff @(posedge clk) begin
        if (rst) begin
            data_right_s1 <= '0; data_right_s2 <= '0; data_right_s3 <= '0;
            data_bottom_s1 <= '0; data_bottom_s2 <= '0; data_bottom_s3 <= '0;
            valid_right_s1 <= 1'b0; valid_right_s2 <= 1'b0; valid_right_s3 <= 1'b0;
            valid_bottom_s1 <= 1'b0; valid_bottom_s2 <= 1'b0; valid_bottom_s3 <= 1'b0;
        end else begin // passing data through PE
            data_right_s1 <= data_in_left;   data_right_s2 <= data_right_s1;   data_right_s3 <= data_right_s2;
            data_bottom_s1 <= data_in_top;   data_bottom_s2 <= data_bottom_s1; data_bottom_s3 <= data_bottom_s2;
            valid_right_s1 <= valid_in_left; valid_right_s2 <= valid_right_s1; valid_right_s3 <= valid_right_s2;
            valid_bottom_s1 <= valid_in_top; valid_bottom_s2 <= valid_bottom_s1; valid_bottom_s3 <= valid_bottom_s2;
        end
    end

    assign data_out_right   = data_right_s3;
    assign data_out_bottom  = data_bottom_s3;
    assign valid_out_right  = valid_right_s3;
    assign valid_out_bottom = valid_bottom_s3;

    // --------------------------------------------------------------- S1
    // Reused exact-design decode (pe_exact_4s.sv stage 1), with Sub-OFF:
    // mantissa forced to 0 whenever the operand's own exponent field is 0
    // (nrm==0), instead of the exact design's IEEE-subnormal {0,mantissa}.

    logic [fi_width-1:0] u_op0_s1, u_op1_s1; // decoded unsigned operands 
    logic prd_sign_s1; // XOR of input signs 
    logic [exp_width-1:0] exp0_s1, exp1_s1; // biased exponents passed through 
    logic valid_s1;
    logic [IDX_W-1:0] idx_s1; // element index
    logic [IDX_W-1:0] elem_count; // running count of elements seen 
    logic is_new_elem;

    assign is_new_elem = valid_in_left && valid_in_top;

    always_ff @(posedge clk) begin
        if (rst) begin
            u_op0_s1 <= '0; u_op1_s1 <= '0;
            prd_sign_s1 <= 1'b0;
            exp0_s1 <= '0; exp1_s1 <= '0;
            valid_s1 <= 1'b0;
            idx_s1 <= '0;
            elem_count <= '0;
        end else begin 
            automatic logic op0_sgn, op1_sgn, op0_nrm, op1_nrm; // sign bits + normal check
            automatic logic [exp_width-1:0] op0_ef, op1_ef; 
            automatic logic [man_width-1:0] op0_mb, op1_mb;
            automatic logic [man_width:0] op0_ext, op1_ext; // break up MX element

            op0_sgn = data_in_left[bit_width-1];
            op0_ef  = data_in_left[bit_width-2:man_width];
            op0_mb  = data_in_left[man_width-1:0];
            op0_nrm = |op0_ef;
            op0_ext = op0_nrm ? {1'b1, op0_mb} : '0;   // Sub-OFF flush

            op1_sgn = data_in_top[bit_width-1];
            op1_ef  = data_in_top[bit_width-2:man_width];
            op1_mb  = data_in_top[man_width-1:0];
            op1_nrm = |op1_ef;
            op1_ext = op1_nrm ? {1'b1, op1_mb} : '0;   // Sub-OFF flush

            u_op0_s1 <= {{(fi_width-man_width-1){1'b0}}, op0_ext};
            u_op1_s1 <= {{(fi_width-man_width-1){1'b0}}, op1_ext};
            prd_sign_s1 <= op0_sgn ^ op1_sgn; // product sign = XOR input signs 
            exp0_s1 <= op0_ef; // passses exponents through S2 
            exp1_s1 <= op1_ef;
            valid_s1 <= is_new_elem;
            idx_s1 <= elem_count; // tag element with its index 
            if (is_new_elem && elem_count < k[IDX_W-1:0])
                elem_count <= elem_count + 1'b1; // incriment counter 
        end
    end

    // --------------------------------------------------------------- S2
    // Reused exact-design exact multiply (pe_exact_4s.sv stage 2).


    logic [2*fi_width-1:0] u_prd_s2;
    logic prd_sign_s2;
    logic [exp_width-1:0] exp0_s2, exp1_s2;
    logic valid_s2;
    logic [IDX_W-1:0] idx_s2;

    always_ff @(posedge clk) begin
        if (rst) begin
            u_prd_s2 <= '0;
            prd_sign_s2 <= 1'b0;
            exp0_s2 <= '0; exp1_s2 <= '0;
            valid_s2 <= 1'b0;
            idx_s2 <= '0;
        end else begin
            u_prd_s2 <= u_op0_s1 * u_op1_s1; // multilpy mantissa 
            prd_sign_s2 <= prd_sign_s1; // pass sign 
            exp0_s2 <= exp0_s1; // pass exponents 
            exp1_s2 <= exp1_s1;
            valid_s2 <= valid_s1;
            idx_s2 <= idx_s1;
        end
    end

    // --------------------------------------------------------------- S3
    // conversion from MX product of previous stage into FP12 for the accumilation 

    logic valid_s3, sign_s3;
    logic [5:0] exp_s3;
    logic [frac_width-1:0] frac_s3;
    logic [IDX_W-1:0] idx_s3;

    mx_product_to_fp_operand #(
        .exp_width(exp_width),
        .man_width(man_width)
    ) u_s3 (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_s2),
        .prd_sign(prd_sign_s2),
        .u_prd(u_prd_s2),
        .exp0_field(exp0_s2),
        .exp1_field(exp1_s2),
        .valid_out(valid_s3),
        .sign_out(sign_s3),
        .exp_out(exp_s3),
        .frac_out(frac_s3)
    );

    // adding one cycle of delay to match S3

    always_ff @(posedge clk) begin
        if (rst) idx_s3 <= '0;
        else     idx_s3 <= idx_s2;
    end

    logic [LANE_W-1:0] lane_idx;
    assign lane_idx = idx_s3 % NUM_LANES; 

    // ------------------------------------------------------- PRNG (per-PE)
    // Dispatch-gated, NOT free-running: the LFSR advances exactly once per
    // add issued to the shared adder (enable = adder_valid_in, stepping at
    // the same posedge that captures rand_in into the adder's S4 stage).
    // The draw an add consumes therefore depends only on its position in
    // this PE's own dispatch order -- intake element i consumes draw i,
    // combine step j consumes draw k+j -- never on absolute cycle time.
    // This makes the PE's result a pure function of its local element
    // stream, identical at any array position behind any systolic stagger
    // or feed gap, which is what lets one block-level golden replay
    // (fp12sr_golden.py::pe_fp12sr_single_block) verify every PE.

    logic [12:0] lfsr_rand;
    logic [12:0] seed_in;
    assign seed_in = (SEED_BASE ^ pe_id[12:0]) | 13'h1;

    lfsr_galois #(.width_i(13)) u_lfsr (
        .clk(clk),
        .rst(rst),
        .seed_in(seed_in),
        .enable(adder_valid_in),
        .rand_out(lfsr_rand)
    );

    // ------------------------------------------- lane register file (P=7)

    logic lane_sign [NUM_LANES];
    logic [5:0] lane_exp [NUM_LANES];
    logic [FP12_MANT_W-1:0] lane_mant [NUM_LANES];

    // ------------------------------------------------------- combine FSM - used to manage lanes and keep PE busy 

    typedef enum logic [1:0] {ST_INTAKE, ST_COMBINE_ISSUE, ST_COMBINE_WAIT, ST_DONE} state_t;
    state_t state;
    logic [IDX_W-1:0] results_received, results_received_next;
    logic [LANE_W-1:0] combine_step;

    assign results_received_next = results_received + (adder_valid_out ? 1'b1 : 1'b0);

    // ------------------------------------------------- shared adder dispatch

    logic adder_valid_in, adder_sign_a, adder_sign_b;
    logic [5:0] adder_exp_a, adder_exp_b;
    logic [FP12_MANT_W-1:0] adder_mant_a;
    logic [frac_width-1:0] adder_frac_b;
    logic [LANE_W-1:0] dispatch_tag;
    logic adder_valid_out, adder_sign_out;
    logic [5:0] adder_exp_out;
    logic [FP12_MANT_W-1:0] adder_mant_out;

    always_comb begin
        if (state == ST_COMBINE_ISSUE || state == ST_COMBINE_WAIT) begin
            adder_valid_in = (state == ST_COMBINE_ISSUE);
            adder_sign_a = lane_sign[0];
            adder_exp_a  = lane_exp[0];
            adder_mant_a = lane_mant[0];
            adder_sign_b = lane_sign[combine_step + 1'b1];
            adder_exp_b  = lane_exp[combine_step + 1'b1];
            adder_frac_b = {lane_mant[combine_step + 1'b1], {EXTRA{1'b0}}};
            dispatch_tag = '0;
        end else begin
            adder_valid_in = valid_s3;
            adder_sign_a = lane_sign[lane_idx];
            adder_exp_a  = lane_exp[lane_idx];
            adder_mant_a = lane_mant[lane_idx];
            adder_sign_b = sign_s3;
            adder_exp_b  = exp_s3;
            adder_frac_b = frac_s3;
            dispatch_tag = lane_idx;
        end
    end

    sr_adder_fp12 #(.man_width(man_width)) u_sr_adder (
        .clk(clk),
        .rst(rst),
        .valid_in(adder_valid_in),
        .sign_a(adder_sign_a),
        .exp_a(adder_exp_a),
        .mant_a(adder_mant_a),
        .sign_b(adder_sign_b),
        .exp_b(adder_exp_b),
        .frac_b(adder_frac_b),
        .rand_in(lfsr_rand),
        .valid_out(adder_valid_out),
        .sign_out(adder_sign_out),
        .exp_out(adder_exp_out),
        .mant_out(adder_mant_out)
    );

    // 6-stage tag pipeline, matching sr_adder_fp12's own 6 register stages,
    // so tag_pipe[5] is available at the same cycle as adder_valid_out/
    // adder_*_out (see module header comment for the cycle-by-cycle proof).

    logic [LANE_W-1:0] tag_pipe [6];
    integer t;
    always_ff @(posedge clk) begin
        if (rst) begin
            for (t = 0; t < 6; t = t + 1) tag_pipe[t] <= '0;
        end else begin
            tag_pipe[0] <= dispatch_tag;
            for (t = 1; t < 6; t = t + 1) tag_pipe[t] <= tag_pipe[t-1];
        end
    end

    integer li;
    always_ff @(posedge clk) begin
        if (rst) begin
            for (li = 0; li < NUM_LANES; li = li + 1) begin
                lane_sign[li] <= 1'b0;
                lane_exp[li]  <= '0;
                lane_mant[li] <= '0;
            end
        end else if (adder_valid_out) begin
            lane_sign[tag_pipe[5]] <= adder_sign_out;
            lane_exp[tag_pipe[5]]  <= adder_exp_out;
            lane_mant[tag_pipe[5]] <= adder_mant_out;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_INTAKE;
            results_received <= '0;
            combine_step <= '0;
            result_valid <= 1'b0;
            result_sign <= 1'b0;
            result_exp <= '0;
            result_mant <= '0;
        end else begin
            result_valid <= 1'b0;
            if (state == ST_INTAKE)
                results_received <= results_received_next;

            case (state)
                ST_INTAKE: begin
                    if (results_received_next == k[IDX_W-1:0])
                        state <= ST_COMBINE_ISSUE;
                end
                ST_COMBINE_ISSUE: begin
                    state <= ST_COMBINE_WAIT;
                end
                ST_COMBINE_WAIT: begin
                    if (adder_valid_out) begin
                        if (combine_step == (NUM_LANES - 2)) begin
                            state <= ST_DONE;
                            result_valid <= 1'b1;
                            result_sign <= adder_sign_out;
                            result_exp  <= adder_exp_out;
                            result_mant <= adder_mant_out;
                        end else begin
                            combine_step <= combine_step + 1'b1;
                            state <= ST_COMBINE_ISSUE;
                        end
                    end
                end
                ST_DONE: begin
                    // hold final result; result_valid already pulsed for one cycle
                end
                default: state <= ST_INTAKE;
            endcase
        end
    end

endmodule
