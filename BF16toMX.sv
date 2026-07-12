// stage of encoding: 
// split BF16 into the exp,mantissa, and sign -> clean up any NaNs or inf (depedning on format)
// -> find max exponent -> floor values to ensure no overflow -> find the shared exponent
// -> encode per element with shifting and mantissa -> round-to-even -> output 

module BF16toMX #(
    parameter k = 32, // size of block 
    parameter exp_size = 4, 
    parameter man_size = 3,
    parameter sat = 1 // saturation mode if 1, otherwise overflow 
)(
    input logic clk, 
    input logic [15:0] BF16 [k], 
    output logic [exp_size+man_size:0] MXout [k], 
    output logic [7:0] MXscale 
);

// define max exponent size for format 

logic [7:0] max_exp_elem; 

always_comb begin 
case ({exp_size[3:0], man_size[3:0]})
    {4'd4, 4'd3}: max_exp_elem = 8'd8;   // MXFP8  E4M3
    {4'd5, 4'd2}: max_exp_elem = 8'd15;  // MXFP8  E5M2
    {4'd3, 4'd2}: max_exp_elem = 8'd4;   // MXFP6  E3M2
    {4'd2, 4'd3}: max_exp_elem = 8'd2;   // MXFP6  E2M3
    {4'd2, 4'd1}: max_exp_elem = 8'd2;   // MXFP4  E2M1
    default:      max_exp_elem = 8'd8;
endcase
end 

// split BF16 input into exponent, mantissa and sign 

logic sign [k]; 
logic [7:0] exp [k]; 
logic [6:0] man [k];
logic [7:0] max; 


always_comb begin
    for (int i=0; i<k; i++) begin 
        sign[i] = BF16[i][15];
        exp[i] = BF16[i][14:7]; 
        man[i] = BF16[i][6:0]; 
    end
end

// clean up NaN/inf 

logic [7:0] exp_cleaned [k]; 

always_comb begin
    for (int j = 0; j<k; j++) begin 
        exp_cleaned[j] = (exp[j] == 8'hFF) ? 8'h00 : exp[j];  
    end 
end

// find max exponent 

always_comb begin
    max = exp_cleaned[0];
    for (int l = 1; l < k; l++) begin
        if (exp_cleaned[l] > max) begin
            max = exp_cleaned[l];
        end
    end
end
 

// floor values to prevent overflow 

logic [7:0] max_floor; 

assign max_floor = (max >= max_exp_elem) ? max : max_exp_elem; 

// assign shared exponent and format mantissa 

logic [7:0] shared_exp; 
logic [7:0] elem_exp[k]; 
logic [7:0] elem_man[k]; 
logic nans[k]; 


assign shared_exp = max_floor - max_exp_elem; 

for (genvar e=0; e<k; e++) begin 
    assign elem_exp[e] = max_floor - exp[e]; 
    assign elem_man[e] = |exp[e] ? {1'b1, man[e]} : {man[e], 1'b0};
    assign nans[e] = (exp[e] == 8'hff);
end 

// round to even 

logic [man_size+exp_size-1:0] p2_elems [k];

    for(genvar i=0; i<k; i++) begin
        fp_rnd_nan_rne # (
            .width_i(8),
            .width_o_exp(exp_size),
            .width_o_man(man_size),
            .width_shift(8),
            .sat(sat),
            .e4m3_spec(((exp_size == 4) && (man_size == 3)))
        ) u0_fp_rnd (
            .i_num(elem_man[i]),
            .i_shift(elem_exp[i]),
            .i_nan(nans[i]),
            .o_exp(p2_elems[i][man_size+exp_size-1:man_size]),
            .o_man(p2_elems[i][man_size-1:0])
        );
    end

    always_ff @(posedge clk) begin
        MXscale <= shared_exp;

        for (int i=0; i<k; i++) begin
            MXout[i] <= {sign[i], p2_elems[i]};
        end
    end


endmodule

module fp_rnd_nan_rne #(
    parameter width_i     = 8,
    parameter width_o_exp = 3,
    parameter width_o_man = 2,
    parameter width_shift = 8,
    parameter sat = 1,
    parameter e4m3_spec = (width_o_exp == 4) && (width_o_man == 3)
)(
    input  logic signed       [width_i-1:0] i_num,
    input  logic unsigned [width_shift-1:0] i_shift,
    input  logic                            i_nan,
    output logic          [width_o_exp-1:0] o_exp,
    output logic          [width_o_man-1:0] o_man
);

    // Max exponent/mantissa values of element type.
    localparam max_exp_elem = (1 << width_o_exp) - 1 - (e4m3_spec ? 0 : 1);
    localparam max_man_elem = (1 << width_o_man) - 1 - (e4m3_spec ? 1 : 0);

    // NaN/Inf element encodings.
    localparam exp_spec = (1 << width_o_exp) - 1;
    localparam man_inf  = 0;
    localparam man_nan  = (1 << width_o_man) - 1;

    // Count leading zeros, align input.
    logic [$clog2(width_i):0] lz_num;
    logic [width_i-1:0] aligned_num;

    clz_int #(
        .width_i(width_i)
    ) u_clz (
        .i_num(i_num),
        .o_lz(lz_num)
    );

    assign aligned_num = i_num << lz_num;

    // Handle non-denormal case. Round mantissa.
    logic R_nrm;
    logic S_nrm;
    logic p0_nrm_rnd;

    assign R_nrm =  aligned_num[width_i-width_o_man-2];
    assign S_nrm = |aligned_num[width_i-width_o_man-3:0];
    assign p0_nrm_rnd = R_nrm && (aligned_num[width_i-width_o_man-1] || S_nrm);

    logic   [width_o_man:0] p0_man_nrm_ofl;  // Extra bit to check for overflow after rounding.
    logic [width_o_man-1:0] p0_man_nrm;      // Output mantissa if output is not denormal.
    logic                   p0_exp_nrm_ofl;  // 1 if mantissa overflowed, 0 otherwise.

    assign p0_man_nrm_ofl = aligned_num[width_i-2:width_i-width_o_man-1] + {{(width_o_man-1){1'b0}}, p0_nrm_rnd};

    assign p0_exp_nrm_ofl = p0_man_nrm_ofl[width_o_man];
    assign p0_man_nrm     = p0_man_nrm_ofl[width_o_man-1:0];

    // Handle denormal case. Round mantissa.
    logic [width_shift+1:0] p0_dnm_shift; // Amount to shift by if output is denormal.
    logic [width_i-2:0] sticky_mask;

    logic R_dnm;
    logic S_dnm;
    logic p0_dnm_rnd;

    assign p0_dnm_shift = $unsigned(i_shift) + $unsigned(lz_num) + $unsigned(width_i) - $unsigned(max_exp_elem) - $unsigned(width_o_man);

    assign sticky_mask = ~({(width_i-1){1'b1}} << (p0_dnm_shift-1));

    assign S_dnm = |(aligned_num[width_i-2:0] & sticky_mask);

    always_comb begin
        if($signed({1'b0, i_shift}) <= $signed($unsigned(max_exp_elem) + $unsigned(width_o_man) - $unsigned(lz_num))) begin
            R_dnm = aligned_num[p0_dnm_shift - 1];
        end else begin
            R_dnm = 0;
        end

        if($signed({1'b0, i_shift}) < $signed($unsigned(max_exp_elem) + $unsigned(width_o_man) - $unsigned(lz_num))) begin
            p0_dnm_rnd = R_dnm && (aligned_num[p0_dnm_shift] || S_dnm);
        end else begin
            p0_dnm_rnd = R_dnm && S_dnm;
        end
    end

    logic [width_o_man-1:0] p0_man_dnm_shifted;
    logic   [width_o_man:0] p0_man_dnm_ofl;  // Extra bit to check for overflow after rounding.
    logic [width_o_man-1:0] p0_man_dnm;      // Output mantissa if output is not denormal.
    logic                   p0_exp_dnm_ofl;  // 1 if mantissa overflowed, 0 otherwise.

    assign p0_man_dnm_shifted = aligned_num >> (p0_dnm_shift);

    assign p0_man_dnm_ofl = p0_man_dnm_shifted + {{(width_o_man-1){1'b0}}, p0_dnm_rnd};

    assign p0_exp_dnm_ofl = p0_man_dnm_ofl[width_o_man];
    assign p0_man_dnm = p0_man_dnm_ofl[width_o_man-1:0];

    // Calculate output exponent, and whether output is denormal, not accounting for overflow.
    logic [width_shift+1:0] p0_exp_out;
    logic                   p0_dnm_out;   // Is output denormal?

    assign p0_exp_out = $unsigned(max_exp_elem) + $unsigned(p0_exp_nrm_ofl) - $unsigned(i_shift) - $unsigned(lz_num);
    assign p0_dnm_out = $signed($unsigned(max_exp_elem) - $unsigned(lz_num)) <= $signed({1'b0, i_shift});

    // Saturation and overflow.
    logic p0_nan;
    logic p0_inf;
    logic p0_max;
    logic p0_spec_out;

    logic [width_o_exp-1:0] p0_spec_exp;
    logic [width_o_man-1:0] p0_spec_man;

    assign p0_nan = i_nan &&  |i_num[width_i-2:0];
    assign p0_inf = i_nan && ~|i_num[width_i-2:0];
    // assign p0_max = p0_exp_nrm_ofl && (i_shift == 0) && (lz_num == 0) &&  (i_num != 0);
    assign p0_max = ~p0_dnm_out && (i_num != 0) && ((p0_exp_out > max_exp_elem) || ((p0_exp_out == max_exp_elem) && (p0_man_nrm > max_man_elem)));

    assign p0_spec_out = p0_max || p0_nan || p0_inf;

    always_comb begin
        if(p0_nan) begin
            p0_spec_exp = exp_spec;
            p0_spec_man = man_nan;

        end else begin
            if(sat) begin
                p0_spec_exp = max_exp_elem;
                p0_spec_man = max_man_elem;
            end else if(e4m3_spec) begin
                p0_spec_exp = exp_spec;
                p0_spec_man = man_nan;
            end else begin
                p0_spec_exp = exp_spec;
                p0_spec_man = man_inf;
            end
        end
    end


    // Assign outputs.
    always_comb begin
        if(p0_spec_out) begin
            o_exp = p0_spec_exp;
        end else if(p0_dnm_out) begin
            o_exp = (i_num != 0) ? p0_exp_dnm_ofl : 0;
        end else begin
            o_exp = (i_num != 0) ? p0_exp_out : 0;
        end
    end

    always_comb begin
        if(p0_spec_out) begin
            o_man = p0_spec_man;
        end else if(p0_dnm_out) begin
            o_man = p0_man_dnm;
        end else begin
            o_man = p0_man_nrm;
        end
    end

endmodule

module clz_int #(
    parameter width_i = 8,
    parameter width_o = $clog2(width_i+1)
)(
    input  logic [width_i-1:0] i_num,
    output logic [width_o-1:0] o_lz
);

    logic [width_o-1:0] num_lz;

    always_comb begin
        num_lz = 0;
        for(int i=1; i<=width_i; i++) begin
            if(i_num[width_i-i]) begin
                break;
            end else begin
                num_lz++;
            end
        end
    end

    assign o_lz = num_lz;

endmodule

