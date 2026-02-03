/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <stdint.h>

#define OPERATING_POINTS_CNT_MAX 32

#define SELECT_SCREEN_CONTENT_TOOLS 2
#define SELECT_INTEGER_MV 2

#define CP_BT_709 1
#define CP_UNSPECIFIED 2

#define TC_SRGB 13
#define TC_UNSPECIFIED 2

#define MC_IDENTITY 0
#define MC_UNSPECIFIED 2

#define CSP_UNKNOWN 0

#define NUM_REF_FRAMES 8
#define PRIMARY_REF_NONE 7
#define REFS_PER_FRAME 7  // Number of reference frames that can be used for inter prediction
#define TOTAL_REFS_PER_FRAME 8  // Number of reference frame types (including intra type)
#define BUFFER_POOL_MAX_SIZE 10 // Number of frames in buffer pool

#define MAX_TILE_WIDTH 4096  // Maximum width of a tile in units of luma samples
#define MAX_TILE_AREA 4096 * 2304  // Maximum area of a tile in units of luma samples
#define MAX_TILE_ROWS 64  // Maximum number of tile rows
#define MAX_TILE_COLS 64  // Maximum number of tile columns

#define SUPERRES_NUM 8  // Numerator for upscaling ratio
#define SUPERRES_DENOM_MIN 9  // Smallest denominator for upscaling ratio
#define SUPERRES_DENOM_BITS 3  // Number of bits sent to specify denominator of upscaling ratio

#define MAX_SEGMENTS 8  // Number of segments allowed in segmentation map
#define SEG_LVL_ALT_Q 0  // Index for quantizer segment feature
#define SEG_LVL_REF_FRAME 5  // Index for reference frame segment feature
#define SEG_LVL_MAX 8  // Number of segment features

#define MAX_LOOP_FILTER 63  // Maximum value used for loop filtering
#define RESTORATION_TILESIZE_MAX 256  // Maximum size of a loop restoration tile

#define WARPEDMODEL_PREC_BITS 16  // Internal precision of warped motion models
#define GM_ABS_TRANS_BITS 12  // Number of bits encoded for translational components of global motion models, if part of a ROTZOOM or AFFINE model
#define GM_ABS_TRANS_ONLY_BITS 9  // Number of bits encoded for translational components of global motion models, if part of a TRANSLATION model
#define GM_ABS_ALPHA_BITS 12  // Number of bits encoded for non-translational components of global motion models
#define GM_ALPHA_PREC_BITS 15  // Number of fractional bits for sending non- translational warp model coefficients
#define GM_TRANS_PREC_BITS 6  // Number of fractional bits for sending translational warp model coefficients
#define GM_TRANS_ONLY_PREC_BITS 3  // Number of fractional bits used for pure translational warps

#define DIV_LUT_BITS 8 // Number of fractional bits for lookup in divisor lookup table
#define DIV_LUT_PREC_BITS 14 // Number of fractional bits of entries in divisor lookup table
#define DIV_LUT_NUM 257 // Number of entries in divisor lookup table
#define WARP_PARAM_REDUCE_BITS 6 // Rounding bitwidth for the parameters to the shear process

typedef enum  {
    kObuSequenceHeader          = 1,
    kObuTemporalDelimiter       = 2,
    kObuFrameHeader             = 3,
    kObuTileGroup               = 4,
    kObuMetaData                = 5,
    kObuFrame                   = 6,
    kObuRedundantFrameHeader    = 7,
    kObuTileList                = 8,
    kObuPadding                 = 15,
} ObuType;

typedef enum {
    kKeyFrame        = 0,
    kInterFrame      = 1,
    kIntraOnlyFrame  = 2,
    kSwitchFrame     = 3,
} FrameType;

typedef enum {
    kEightTap        = 0,
    kEightTapSmooth  = 1,
    kEightTapSharp   = 2,
    kBilinear        = 3,
    kSwitchable      = 4,
} InterpolotionFilterType;

typedef enum {
    kNone           = -1,
    kIntraFrame     = 0,
    kLastFrame      = 1,
    kLast2Frame     = 2,
    kLast3Frame     = 3,
    kGoldenFrame    = 4,
    kBwdRefFrame    = 5,
    kAltRef2Frame   = 6,
    kAltRefFrame    = 7,
} RefFrame;

typedef enum {
    kRestoreNone        = 0,
    kRestoreSwitchable  = 3,
    kRestoreWiener      = 1,
    kRestoreSgrproj     = 2,
} FrameRestorationType;

typedef enum {
    kOnly4x4        = 0,
    kTxModeLargest  = 1,
    kTxModeSelect   = 2,
} Tx_Mode;

typedef enum {
    kIdentity       = 0, // Warp model is just an identity transform
    kTranslation    = 1,  // Warp model is a pure translation
    kRotZoom        = 2,  // Warp model is a rotation + symmetric zoom + translation
    kAffine         = 3,  // Warp model is a general affine transform
} WarpModel;

typedef struct {
    uint32_t size;
    uint32_t obu_forbidden_bit;
    uint32_t obu_type;
    uint32_t obu_extension_flag;
    uint32_t obu_has_size_field;
    uint32_t obu_reserved_1bit;
    uint32_t temporal_id;
    uint32_t spatial_id;
    uint32_t extension_header_reserved_3bits;
} Av1ObuHeader;

typedef struct {
    uint32_t num_units_in_display_tick;
    uint32_t time_scale;
    uint32_t equal_picture_interval;
    uint32_t num_ticks_per_picture_minus_1;
} Av1TimingInfo;

typedef struct {
    uint32_t buffer_delay_length_minus_1;
    uint32_t num_units_in_decoding_tick;
    uint32_t buffer_removal_time_length_minus_1;
    uint32_t frame_presentation_time_length_minus_1;
} Av1DecoderModelInfo;

typedef struct {
    uint32_t decoder_buffer_delay;
    uint32_t encoder_buffer_delay;
    uint32_t low_delay_mode_flag;
} Av1OperatingParametersInfo;

typedef struct {
    uint32_t high_bitdepth;
    uint32_t twelve_bit;
    uint32_t bit_depth; // BitDepth
    uint32_t mono_chrome;
    uint32_t num_planes; // NumPlanes
    uint32_t color_description_present_flag;
    uint32_t color_primaries;
    uint32_t transfer_characteristics;
    uint32_t matrix_coefficients;
    uint32_t color_range;
    uint32_t subsampling_x;
    uint32_t subsampling_y;
    uint32_t chroma_sample_position;
    uint32_t separate_uv_delta_q;
} Av1ColorConfig;

typedef struct {
    uint32_t is_received;
    uint32_t seq_profile;
    uint32_t still_picture;
    uint32_t reduced_still_picture_header;
    uint32_t timing_info_present_flag;
    Av1TimingInfo timing_info;
    uint32_t decoder_model_info_present_flag;
    Av1DecoderModelInfo decoder_model_info;
    uint32_t initial_display_delay_present_flag;
    uint32_t operating_points_cnt_minus_1;
    uint32_t operating_point_idc[OPERATING_POINTS_CNT_MAX];
    uint32_t seq_level_idx[OPERATING_POINTS_CNT_MAX];
    uint32_t seq_tier[OPERATING_POINTS_CNT_MAX];
    uint32_t decoder_model_present_for_this_op[OPERATING_POINTS_CNT_MAX];
    Av1OperatingParametersInfo operating_parameters_info[OPERATING_POINTS_CNT_MAX];
    uint32_t initial_display_delay_present_for_this_op[OPERATING_POINTS_CNT_MAX];
    uint32_t initial_display_delay_minus_1[OPERATING_POINTS_CNT_MAX];
    uint32_t frame_width_bits_minus_1;
    uint32_t frame_height_bits_minus_1;
    uint32_t max_frame_width_minus_1;
    uint32_t max_frame_height_minus_1;
    uint32_t frame_id_numbers_present_flag;
    uint32_t delta_frame_id_length_minus_2;
    uint32_t additional_frame_id_length_minus_1;
    uint32_t use_128x128_superblock;
    uint32_t enable_filter_intra;
    uint32_t enable_intra_edge_filter;
    uint32_t enable_interintra_compound;
    uint32_t enable_masked_compound;
    uint32_t enable_warped_motion;
    uint32_t enable_dual_filter;
    uint32_t enable_order_hint;
    uint32_t enable_jnt_comp;
    uint32_t enable_ref_frame_mvs;
    uint32_t seq_choose_screen_content_tools;
    uint32_t seq_force_screen_content_tools;
    uint32_t seq_choose_integer_mv;
    uint32_t seq_force_integer_mv;
    uint32_t order_hint_bits_minus_1;
    uint32_t order_hint_bits;  // OrderHintBits
    uint32_t enable_superres;
    uint32_t enable_cdef;
    uint32_t enable_restoration;
    Av1ColorConfig color_config;
    uint32_t film_grain_params_present;
} Av1SequenceHeader;

typedef struct {
    uint32_t frame_presentation_time;
} Av1TemporalPointInfo;

typedef struct {
    uint32_t use_superres;
    uint32_t coded_denom;
    uint32_t super_res_denom;
} Av1SuperResParams;

typedef struct {
    uint32_t frame_width_minus_1;
    uint32_t frame_width; // FrameWidth
    uint32_t frame_height_minus_1;
    uint32_t frame_height; // FrameHeight
    uint32_t upscaled_width; // UpscaledWidth
    Av1SuperResParams superres_params;
    uint32_t mi_cols;
    uint32_t mi_rows;
} Av1FrameSize;

typedef struct {
    uint32_t render_and_frame_size_different;
    uint32_t render_width_minus_1;
    uint32_t render_width; // RenderWidth
    uint32_t render_height_minus_1;
    uint32_t render_height; // RenderHeight
} Av1RenderSize;

typedef struct {
    uint32_t uniform_tile_spacing_flag;
    int32_t  tile_cols_log2;
    int32_t  tile_rows_log2;
    uint32_t increment_tile_cols_log2;
    uint32_t increment_tile_rows_log2;
    int32_t  mi_col_starts[MAX_TILE_COLS + 1];
    int32_t  mi_row_starts[MAX_TILE_ROWS + 1];
    int32_t  tile_cols;
    int32_t  tile_rows;
    uint32_t width_in_sbs_minus_1[MAX_TILE_COLS];
    uint32_t height_in_sbs_minus_1[MAX_TILE_ROWS];
    uint32_t context_update_tile_id;
    uint32_t tile_size_bytes_minus_1;
} Av1TileInfoSyntx;

typedef struct {
    uint32_t base_q_idx;
    uint32_t delta_coded;
    int32_t  delta_q;
    int32_t  delta_q_y_dc;
    uint32_t diff_uv_delta;
    int32_t  delta_q_u_dc;
    int32_t  delta_q_u_ac;
    int32_t  delta_q_v_dc;
    int32_t  delta_q_v_ac;
    uint32_t using_qmatrix;
    uint32_t qm_y;
    uint32_t qm_u;
    uint32_t qm_v;
} Av1QuantizationParams;

typedef struct {
    uint32_t segmentation_enabled;
    uint32_t segmentation_update_map;
    uint32_t segmentation_temporal_update;
    uint32_t segmentation_update_data;
    uint32_t feature_enabled;
    uint8_t  feature_enabled_flags[MAX_SEGMENTS][SEG_LVL_MAX];
    uint32_t feature_value;
    int16_t  feature_data[MAX_SEGMENTS][SEG_LVL_MAX];
    uint32_t seg_id_pre_skip;
    uint32_t last_active_seg_id;
} Av1SegmentationParams;

typedef struct {
    uint32_t delta_q_present;
    uint32_t delta_q_res;
} Av1DeltaQParams;

typedef struct {
    uint32_t delta_lf_present;
    uint32_t delta_lf_res;
    uint32_t delta_lf_multi;
} Av1DeltaLFParams;

typedef struct {
    uint32_t loop_filter_level[4];
    uint32_t loop_filter_sharpness;
    uint32_t loop_filter_delta_enabled;
    uint32_t loop_filter_delta_update;
    uint32_t update_ref_delta;
    int32_t  loop_filter_ref_deltas[TOTAL_REFS_PER_FRAME];
    uint32_t update_mode_delta;
    int32_t  loop_filter_mode_deltas[2];
} Av1LoopFilterParams;

typedef struct {
    uint32_t cdef_damping_minus_3;
    uint32_t cdef_bits;
    uint32_t cdef_y_pri_strength[8];
    uint32_t cdef_y_sec_strength[8];
    uint32_t cdef_uv_pri_strength[8];
    uint32_t cdef_uv_sec_strength[8];
    uint32_t cdef_damping;
} Av1CdefParams;

typedef struct {
    uint32_t frame_restoration_type[3];
    uint32_t uses_lr;
    uint32_t lr_type[3];
    uint32_t lr_unit_shift;
    uint32_t lr_unit_extra_shift;
    uint32_t loop_restoration_size[3];
    uint32_t lr_uv_shift;
} Av1LRParams;

typedef struct {
    uint32_t tx_mode_select;
    uint32_t tx_mode;
} Av1TxMode;

typedef struct {
    uint32_t reference_select;
} Av1FrameReferenceMode;

typedef struct {
    uint32_t skip_mode_frame[2];
    uint32_t skip_mode_present;
} Av1SkipModeParams;

typedef struct {
    uint8_t  gm_invalid[NUM_REF_FRAMES];
    uint8_t  gm_type[NUM_REF_FRAMES];
    int32_t  gm_params[NUM_REF_FRAMES][6];
    uint32_t is_global;
    uint32_t is_rot_zoom;
    uint32_t is_translation;
} Av1GlobalMotionParams;

typedef struct {
    uint32_t apply_grain;
    uint32_t grain_seed;
    uint32_t update_grain;
    uint32_t film_grain_params_ref_idx;
    uint32_t num_y_points;
    uint32_t point_y_value[14];
    uint32_t point_y_scaling[14];
    uint32_t chroma_scaling_from_luma;
    uint32_t num_cb_points;
    uint32_t num_cr_points;
    uint32_t point_cb_value[10];
    uint32_t point_cb_scaling[10];
    uint32_t point_cr_value[10];
    uint32_t point_cr_scaling[10];
    uint32_t grain_scaling_minus_8;
    uint32_t ar_coeff_lag;
    uint32_t ar_coeffs_y_plus_128[24];
    uint32_t ar_coeffs_cb_plus_128[25];
    uint32_t ar_coeffs_cr_plus_128[25];
    uint32_t ar_coeff_shift_minus_6;
    uint32_t grain_scale_shift;
    uint32_t cb_mult;
    uint32_t cb_luma_mult;
    uint32_t cb_offset;
    uint32_t cr_mult;
    uint32_t cr_luma_mult;
    uint32_t cr_offset;
    uint32_t overlap_flag;
    uint32_t clip_to_restricted_range;
} Av1FilmGrainParams;

typedef struct {
    uint32_t is_received;
    uint32_t show_existing_frame;
    uint32_t frame_to_show_map_idx;
    Av1TemporalPointInfo temporal_point_info;
    uint32_t display_frame_id;
    uint32_t frame_type;
    uint32_t frame_is_intra;
    uint32_t show_frame;
    uint32_t showable_frame;
    uint32_t error_resilient_mode;
    uint32_t disable_cdf_update;
    uint32_t allow_screen_content_tools;
    uint32_t force_integer_mv;
    uint32_t current_frame_id;
    uint32_t prev_frame_id;
    uint32_t frame_size_override_flag;
    uint32_t order_hint;
    uint32_t order_hints[NUM_REF_FRAMES];
    uint32_t primary_ref_frame;
    uint32_t buffer_removal_time_present_flag;
    uint32_t buffer_removal_time[OPERATING_POINTS_CNT_MAX];
    uint32_t refresh_frame_flags;
    uint32_t ref_order_hint[NUM_REF_FRAMES];
    uint32_t ref_frame_sign_bias[NUM_REF_FRAMES];
    uint32_t found_ref;
    Av1FrameSize frame_size;
    Av1RenderSize render_size;
    uint32_t allow_intrabc;
    uint32_t frame_refs_short_signaling;
    uint32_t last_frame_idx;
    uint32_t gold_frame_idx;
    int32_t  ref_frame_idx[REFS_PER_FRAME];
    uint32_t delta_frame_id_minus_1;
    uint32_t expected_frame_id[REFS_PER_FRAME];
    uint32_t allow_high_precision_mv;
    uint32_t is_filter_switchable;
    uint32_t interpolation_filter;
    uint32_t is_motion_mode_switchable;
    uint32_t use_ref_frame_mvs;
    uint32_t disable_frame_end_update_cdf;
    Av1TileInfoSyntx tile_info;
    Av1QuantizationParams quantization_params;
    Av1SegmentationParams segmentation_params;
    Av1DeltaQParams delta_q_params;
    Av1DeltaLFParams delta_lf_params;
    uint32_t coded_lossless;
    uint32_t lossless_array[MAX_SEGMENTS];
    uint32_t seg_qm_level[3][MAX_SEGMENTS];
    uint32_t all_lossless;
    Av1LoopFilterParams loop_filter_params;
    Av1CdefParams cdef_params;
    Av1LRParams lr_params;
    Av1TxMode tx_mode;
    Av1FrameReferenceMode frame_reference_mode;
    Av1SkipModeParams skip_mode_params;
    uint32_t allow_warped_motion;
    uint32_t reduced_tx_set;
    Av1GlobalMotionParams global_motion_params;
    Av1FilmGrainParams film_grain_params;
} Av1FrameHeader;