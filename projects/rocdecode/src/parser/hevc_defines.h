/*
Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#define MAX_VPS_COUNT 16    // 7.3.2.1
#define MAX_SPS_COUNT 16    // 7.3.2.2.1
#define MAX_PPS_COUNT 64    // 7.4.3.3.1

#define HEVC_SCALING_LIST_NUM 6                              ///< list number for quantization matrix
#define HEVC_SCALING_LIST_MAX_INDEX 64

#define HEVC_MAX_DPB_FRAMES 16  // (A-2)
#define HEVC_MAX_NUM_REF_PICS 16
// 7.4.7.1. (num_tile_columns_minus1 + 1) * PicHeightInCtbsY − 1. Max tile columns = 20 (A.4.2). Pic height in 16x16 CTB of 8K = 270.
#define MAX_ENTRY_POINT_OFFSETS 20 * 270

/*! \brief Enumerator for the NAL Unit types - ISO-IEC 14496-15-2004.pdf, page 14, table 1 " NAL unit types in elementary streams
 */
enum HevcNalUnitType {
    NAL_UNIT_CODED_SLICE_TRAIL_N = 0, // 0
    NAL_UNIT_CODED_SLICE_TRAIL_R,     // 1

    NAL_UNIT_CODED_SLICE_TSA_N,       // 2
    NAL_UNIT_CODED_SLICE_TLA_R,       // 3

    NAL_UNIT_CODED_SLICE_STSA_N,      // 4
    NAL_UNIT_CODED_SLICE_STSA_R,      // 5

    NAL_UNIT_CODED_SLICE_RADL_N,      // 6
    NAL_UNIT_CODED_SLICE_RADL_R,      // 7

    NAL_UNIT_CODED_SLICE_RASL_N,      // 8
    NAL_UNIT_CODED_SLICE_RASL_R,      // 9

    NAL_UNIT_RESERVED_VCL_N10,
    NAL_UNIT_RESERVED_VCL_R11,
    NAL_UNIT_RESERVED_VCL_N12,
    NAL_UNIT_RESERVED_VCL_R13,
    NAL_UNIT_RESERVED_VCL_N14,
    NAL_UNIT_RESERVED_VCL_R15,

    NAL_UNIT_CODED_SLICE_BLA_W_LP,    // 16
    NAL_UNIT_CODED_SLICE_BLA_W_RADL,  // 17
    NAL_UNIT_CODED_SLICE_BLA_N_LP,    // 18
    NAL_UNIT_CODED_SLICE_IDR_W_RADL,  // 19
    NAL_UNIT_CODED_SLICE_IDR_N_LP,    // 20
    NAL_UNIT_CODED_SLICE_CRA_NUT,     // 21
    NAL_UNIT_RESERVED_IRAP_VCL22,
    NAL_UNIT_RESERVED_IRAP_VCL23,

    NAL_UNIT_RESERVED_VCL24,
    NAL_UNIT_RESERVED_VCL25,
    NAL_UNIT_RESERVED_VCL26,
    NAL_UNIT_RESERVED_VCL27,
    NAL_UNIT_RESERVED_VCL28,
    NAL_UNIT_RESERVED_VCL29,
    NAL_UNIT_RESERVED_VCL30,
    NAL_UNIT_RESERVED_VCL31,

    NAL_UNIT_VPS,                     // 32
    NAL_UNIT_SPS,                     // 33
    NAL_UNIT_PPS,                     // 34
    NAL_UNIT_ACCESS_UNIT_DELIMITER,   // 35
    NAL_UNIT_EOS,                     // 36
    NAL_UNIT_EOB,                     // 37
    NAL_UNIT_FILLER_DATA,             // 38
    NAL_UNIT_PREFIX_SEI,              // 39
    NAL_UNIT_SUFFIX_SEI,              // 40
    NAL_UNIT_RESERVED_NVCL41,
    NAL_UNIT_RESERVED_NVCL42,
    NAL_UNIT_RESERVED_NVCL43,
    NAL_UNIT_RESERVED_NVCL44,
    NAL_UNIT_RESERVED_NVCL45,
    NAL_UNIT_RESERVED_NVCL46,
    NAL_UNIT_RESERVED_NVCL47,
    NAL_UNIT_UNSPECIFIED_48,
    NAL_UNIT_UNSPECIFIED_49,
    NAL_UNIT_UNSPECIFIED_50,
    NAL_UNIT_UNSPECIFIED_51,
    NAL_UNIT_UNSPECIFIED_52,
    NAL_UNIT_UNSPECIFIED_53,
    NAL_UNIT_UNSPECIFIED_54,
    NAL_UNIT_UNSPECIFIED_55,
    NAL_UNIT_UNSPECIFIED_56,
    NAL_UNIT_UNSPECIFIED_57,
    NAL_UNIT_UNSPECIFIED_58,
    NAL_UNIT_UNSPECIFIED_59,
    NAL_UNIT_UNSPECIFIED_60,
    NAL_UNIT_UNSPECIFIED_61,
    NAL_UNIT_UNSPECIFIED_62,
    NAL_UNIT_UNSPECIFIED_63,
    NAL_UNIT_INVALID,
};

/*! \brief Structure to hold the NAL Unit Header
 */
struct HevcNalUnitHeader {
    uint32_t forbidden_zero_bit;
    uint32_t nal_unit_type;
    uint32_t nuh_layer_id;
    uint32_t nuh_temporal_id_plus1;
    uint32_t num_emu_byte_removed;
};

/*! \brief Enumerator for the scaling list sizes
 */
enum HevcScalingListSize {
    HEVC_SCALING_LIST_4x4 = 0,
    HEVC_SCALING_LIST_8x8,
    HEVC_SCALING_LIST_16x16,
    HEVC_SCALING_LIST_32x32,
    HEVC_SCALING_LIST_SIZE_NUM
};

/*! \brief Slice type
 */
enum HevcSliceType {
    HEVC_SLICE_TYPE_B = 0,
    HEVC_SLICE_TYPE_P,
    HEVC_SLICE_TYPE_I
};

/*! \brief Reference picture type
 */
enum HevcRefMarking {
    kUnusedForReference = 0,
    kUsedForShortTerm = 1,
    kUsedForLongTerm = 2
};

/*! \brief Structure for Profile Tier Levels
 */
typedef struct {
    uint32_t general_profile_space;                      //u(2)
    bool general_tier_flag;                              //u(1)
    uint32_t general_profile_idc;                        //u(5)
    bool general_profile_compatibility_flag[32];         //u(1)
    bool general_progressive_source_flag;                //u(1)
    bool general_interlaced_source_flag;                 //u(1)
    bool general_non_packed_constraint_flag;             //u(1)
    bool general_frame_only_constraint_flag;             //u(1)
    uint64_t general_reserved_zero_44bits;               //u(44)
    uint32_t general_level_idc;                          //u(8)
    //max_num_sub_layers_minus1 max is 7 - 1 = 6
    bool sub_layer_profile_present_flag[6];              //u(1)
    bool sub_layer_level_present_flag[6];                //u(1)

    uint32_t reserved_zero_2bits[8];                     //u(2)

    uint32_t sub_layer_profile_space[6];                 //u(2)
    bool sub_layer_tier_flag[6];                         //u(1)
    uint32_t sub_layer_profile_idc[6];                   //u(5)
    bool sub_layer_profile_compatibility_flag[6][32];    //u(1)
    bool sub_layer_progressive_source_flag[6];           //u(1)
    bool sub_layer_interlaced_source_flag[6];            //u(1)
    bool sub_layer_non_packed_constraint_flag[6];        //u(1)
    bool sub_layer_frame_only_constraint_flag[6];        //u(1)
    uint64_t sub_layer_reserved_zero_44bits[6];          //u(44)
    uint32_t sub_layer_level_idc[6];                     //u(8)
} HevcProfileTierLevel;

/*! \brief Structure for Scaling List Data
 */
typedef struct {
    bool scaling_list_pred_mode_flag[4][6];              //u(1)
    uint32_t scaling_list_pred_matrix_id_delta[4][6];    //ue(v)
    int32_t scaling_list_dc_coef_minus8[4][6];           //se(v)
    int32_t scaling_list_delta_coef;                     //se(v)
    int32_t scaling_list[HEVC_SCALING_LIST_SIZE_NUM][HEVC_SCALING_LIST_NUM][HEVC_SCALING_LIST_MAX_INDEX];
    int32_t scaling_list_dc_coef[2][6];  // DC coefficient for 16x16 and 32x32
} HevcScalingListData;

/*! \brief Structure for Short Term Reference Picture Set
 */
typedef struct {
    uint8_t     inter_ref_pic_set_prediction_flag;
    uint32_t    delta_idx_minus1;
    uint8_t     delta_rps_sign;
    uint32_t    abs_delta_rps_minus1;
    uint8_t     used_by_curr_pic_flag[HEVC_MAX_DPB_FRAMES];
    uint8_t     use_delta_flag[HEVC_MAX_DPB_FRAMES];

    uint32_t    delta_poc_s0_minus1[HEVC_MAX_DPB_FRAMES];
    uint8_t     used_by_curr_pic_s0_flag[HEVC_MAX_DPB_FRAMES];
    uint32_t    delta_poc_s1_minus1[HEVC_MAX_DPB_FRAMES];
    uint8_t     used_by_curr_pic_s1_flag[HEVC_MAX_DPB_FRAMES];

    uint32_t    num_negative_pics; // NumNegativePics
    uint32_t    num_positive_pics; // NumPositivePics
    uint32_t    num_of_delta_pocs; // NumDeltaPocs
    uint8_t     used_by_curr_pic_s0[HEVC_MAX_DPB_FRAMES]; // UsedByCurrPicS0
    uint8_t     used_by_curr_pic_s1[HEVC_MAX_DPB_FRAMES]; // UsedByCurrPicS1
    int32_t     delta_poc_s0[HEVC_MAX_DPB_FRAMES]; // DeltaPocS0
    int32_t     delta_poc_s1[HEVC_MAX_DPB_FRAMES]; // DeltaPocS1
} HevcShortTermRps;

/*! \brief Structure for Long Term Reference Picture Set
 */
typedef struct {
    int32_t num_of_pics;
    int32_t pocs[32];  // PocLsbLt
    bool used_by_curr_pic[32];  // UsedByCurrPicLt
} HevcLongTermRps;

/*! \brief Structure for Sub Layer Hypothetical Reference Decoder Parameters
 */
typedef struct {
    //CpbCnt = cpb_cnt_minus1
    uint32_t bit_rate_value_minus1[32];                  //ue(v)
    uint32_t cpb_size_value_minus1[32];                  //ue(v)
    uint32_t cpb_size_du_value_minus1[32];               //ue(v)
    uint32_t bit_rate_du_value_minus1[32];               //ue(v)
    bool cbr_flag[32];                                   //u(1)
} HevcSubLayerHrdParameters;

/*! \brief Structure for Hypothetical Reference Decoder Parameters
 */
typedef struct {
    bool nal_hrd_parameters_present_flag;                //u(1)
    bool vcl_hrd_parameters_present_flag;                //u(1)
    bool sub_pic_hrd_params_present_flag;                //u(1)
    uint32_t tick_divisor_minus2;                        //u(8)
    uint32_t du_cpb_removal_delay_increment_length_minus1;  //u(5)
    bool sub_pic_cpb_params_in_pic_timing_sei_flag;      //u(1)
    uint32_t dpb_output_delay_du_length_minus1;          //u(5)
    uint32_t bit_rate_scale;                             //u(4)
    uint32_t cpb_size_scale;                             //u(4)
    uint32_t cpb_size_du_scale;                          //u(4)
    uint32_t initial_cpb_removal_delay_length_minus1;    //u(5)
    uint32_t au_cpb_removal_delay_length_minus1;         //u(5)
    uint32_t dpb_output_delay_length_minus1;             //u(5)
    bool fixed_pic_rate_general_flag[7];                 //u(1)
    bool fixed_pic_rate_within_cvs_flag[7];              //u(1)
    uint32_t elemental_duration_in_tc_minus1[7];         //ue(v)
    bool low_delay_hrd_flag[7];                          //u(1)
    uint32_t cpb_cnt_minus1[7];                          //ue(v)
    //sub_layer_hrd_parameters()
    HevcSubLayerHrdParameters sub_layer_hrd_parameters_0[7];
    //sub_layer_hrd_parameters()
    HevcSubLayerHrdParameters sub_layer_hrd_parameters_1[7];
} HevcHrdParameters;

/*! \brief Structure for Video Usability Information Parameters
 */
typedef struct {
    bool aspect_ratio_info_present_flag;                 //u(1)
    uint32_t aspect_ratio_idc;                           //u(8)
    uint32_t sar_width;                                  //u(16)
    uint32_t sar_height;                                 //u(16)
    bool overscan_info_present_flag;                     //u(1)
    bool overscan_appropriate_flag;                      //u(1)
    bool video_signal_type_present_flag;                 //u(1)
    uint32_t video_format;                               //u(3)
    bool video_full_range_flag;                          //u(1)
    bool colour_description_present_flag;                //u(1)
    uint32_t colour_primaries;                           //u(8)
    uint32_t transfer_characteristics;                   //u(8)
    uint32_t matrix_coeffs;                              //u(8)
    bool chroma_loc_info_present_flag;                   //u(1)
    uint32_t chroma_sample_loc_type_top_field;           //ue(v)
    uint32_t chroma_sample_loc_type_bottom_field;        //ue(v)
    bool neutral_chroma_indication_flag;                 //u(1)
    bool field_seq_flag;                                 //u(1)
    bool frame_field_info_present_flag;                  //u(1)
    bool default_display_window_flag;                    //u(1)
    uint32_t def_disp_win_left_offset;                   //ue(v)
    uint32_t def_disp_win_right_offset;                  //ue(v)
    uint32_t def_disp_win_top_offset;                    //ue(v)
    uint32_t def_disp_win_bottom_offset;                 //ue(v)
    bool vui_timing_info_present_flag;                   //u(1)
    uint32_t vui_num_units_in_tick;                      //u(32)
    uint32_t vui_time_scale;                             //u(32)
    bool vui_poc_proportional_to_timing_flag;            //u(1)
    uint32_t vui_num_ticks_poc_diff_one_minus1;          //ue(v)
    bool vui_hrd_parameters_present_flag;                //u(1)
    //hrd_parameters()
    HevcHrdParameters hrd_parameters;
    bool bitstream_restriction_flag;                     //u(1)
    bool tiles_fixed_structure_flag;                     //u(1)
    bool motion_vectors_over_pic_boundaries_flag;        //u(1)
    bool restricted_ref_pic_lists_flag;                  //u(1)
    uint32_t min_spatial_segmentation_idc;               //ue(v)
    uint32_t max_bytes_per_pic_denom;                    //ue(v)
    uint32_t max_bits_per_min_cu_denom;                  //ue(v)
    uint32_t log2_max_mv_length_horizontal;              //ue(v)
    uint32_t log2_max_mv_length_vertical;                //ue(v)
} HevcVuiParameters;

typedef struct {
    uint32_t    luma_log2_weight_denom;                     //ue(v)
    int32_t     delta_chroma_log2_weight_denom;             //se(v)
    uint8_t     luma_weight_l0_flag[16];                    //u(1)
    uint8_t     chroma_weight_l0_flag[16];                  //u(1)
    int32_t     delta_luma_weight_l0[16];                   //se(v)
    int32_t     luma_offset_l0[16];                         //se(v)
    int32_t     delta_chroma_weight_l0[16][2];              //se(v)
    int32_t     delta_chroma_offset_l0[16][2];              //se(v)
    int32_t     chroma_weight_l0[16][2];                    //ChromaWeightL0[]
    int32_t     chroma_offset_l0[16][2];                    //ChromaOffsetL0[]
    uint8_t     luma_weight_l1_flag[16];                    //u(1)
    uint8_t     chroma_weight_l1_flag[16];                  //u(1)
    int32_t     delta_luma_weight_l1[16];                   //se(v)
    int32_t     luma_offset_l1[16];                         //se(v)
    int32_t     delta_chroma_weight_l1[16][2];              //se(v)
    int32_t     delta_chroma_offset_l1[16][2];              //se(v)
    int32_t     chroma_weight_l1[16][2];                    //ChromaWeightL1[]
    int32_t     chroma_offset_l1[16][2];                    //ChromaOffsetL1[]
} HevcPredWeightTable;

/*! \brief Structure for Raw Byte Sequence Payload Trialing Bits
 */
typedef struct {
    uint32_t rbsp_stop_one_bit; /* equal to 1 */
    uint32_t rbsp_alignment_zero_bit; /* equal to 0 */
} HevcRbspTrailingBits;

/*! \brief Structure for Video Parameter Set
 */
typedef struct {
    uint32_t is_received;                                // received with vps_video_parameter_set_id
    uint32_t vps_video_parameter_set_id;                 //u(4)
    uint32_t vps_base_layer_internal_flag;               //u(1)
    uint32_t vps_base_layer_available_flag;              //u(1)
    uint32_t vps_max_layers_minus1;                      //u(6)
    uint32_t vps_max_sub_layers_minus1;                  //u(3)
    bool vps_temporal_id_nesting_flag;                   //u(1)
    uint32_t vps_reserved_0xffff_16bits;                 //u(16)
    //profile_tier_level( vps_max_sub_layers_minus1 )
    HevcProfileTierLevel profile_tier_level;
    bool vps_sub_layer_ordering_info_present_flag;       //u(1)
    //vps_max_sub_layers_minus1 max is 6, need to +1
    uint32_t vps_max_dec_pic_buffering_minus1[7];        //ue(v)
    uint32_t vps_max_num_reorder_pics[7];                //ue(v)
    uint32_t vps_max_latency_increase_plus1[7];          //ue(v)
    uint32_t vps_max_layer_id;                           //u(6)
    uint32_t vps_num_layer_sets_minus1;                  //ue(v)
    //vps_num_layer_sets_minus1 max is  1023  (dont +1 since starts from 1)
    //vps_max_layer_id max is 62                   (+1 since starts from 0 and <= condition)
    bool layer_id_included_flag[1023][63];               //u(1)
    bool vps_timing_info_present_flag;                   //u(1)
    uint32_t vps_num_units_in_tick;                      //u(32)
    uint32_t vps_time_scale;                             //u(32)
    bool vps_poc_proportional_to_timing_flag;            //u(1)
    uint32_t vps_num_ticks_poc_diff_one_minus1;          //ue(v)
    uint32_t vps_num_hrd_parameters;                     //ue(v)
    //vps_num_hrd_parameters max is 1024
    uint32_t hrd_layer_set_idx[1024];                    //ue(v)
    bool cprms_present_flag[1024];                       //u(1)
    //hrd_parameters()
    HevcHrdParameters hrd_parameters[1024];
    bool vps_extension_flag;                             //u(1)
    bool vps_extension_data_flag;                        //u(1)
    //rbsp_trailing_bits()
    HevcRbspTrailingBits rbsp_trailing_bits;
} HevcVideoParamSet;

/*! \brief Structure for Sequence Parameter Set
 */
typedef struct {
    uint32_t is_received;                                // received with sps_seq_parameter_set_id
    uint32_t sps_video_parameter_set_id;                 //u(4)
    uint32_t sps_max_sub_layers_minus1;                  //u(3)
    bool sps_temporal_id_nesting_flag;                   //u(1)
    //profile_tier_level( sps_max_sub_layers_minus1 )
    HevcProfileTierLevel profile_tier_level;
    uint32_t sps_seq_parameter_set_id;                   //ue(v)
    uint32_t chroma_format_idc;                          //ue(v)
    bool separate_colour_plane_flag;                     //u(1)
    uint32_t pic_width_in_luma_samples;                  //ue(v)
    uint32_t pic_height_in_luma_samples;                 //ue(v)
    uint32_t max_cu_width;
    uint32_t max_cu_height;
    uint32_t max_cu_depth;
    bool conformance_window_flag;                        //u(1)
    uint32_t conf_win_left_offset;                       //ue(v)
    uint32_t conf_win_right_offset;                      //ue(v)
    uint32_t conf_win_top_offset;                        //ue(v)
    uint32_t conf_win_bottom_offset;                     //ue(v)
    uint32_t bit_depth_luma_minus8;                      //ue(v)
    uint32_t bit_depth_chroma_minus8;                    //ue(v)
    uint32_t log2_max_pic_order_cnt_lsb_minus4;          //ue(v)
    bool sps_sub_layer_ordering_info_present_flag;       //u(1)
    uint32_t sps_max_dec_pic_buffering_minus1[7];        //ue(v)
    uint32_t sps_max_num_reorder_pics[7];                //ue(v)
    uint32_t sps_max_latency_increase_plus1[7];          //ue(v)
    uint32_t log2_min_luma_coding_block_size_minus3;     //ue(v)
    uint32_t log2_diff_max_min_luma_coding_block_size;   //ue(v)
    uint32_t log2_min_luma_transform_block_size_minus2;       //ue(v)
    uint32_t log2_diff_max_min_luma_transform_block_size;     //ue(v)
    uint32_t max_transform_hierarchy_depth_inter;        //ue(v)
    uint32_t max_transform_hierarchy_depth_intra;        //ue(v)
    bool scaling_list_enabled_flag;                      //u(1)
    bool sps_scaling_list_data_present_flag;             //u(1)
    //scaling_list_data()
    HevcScalingListData scaling_list_data;
    bool amp_enabled_flag;                               //u(1)
    bool sample_adaptive_offset_enabled_flag;            //u(1)
    bool pcm_enabled_flag;                               //u(1)
    uint32_t pcm_sample_bit_depth_luma_minus1;           //u(4)
    uint32_t pcm_sample_bit_depth_chroma_minus1;         //u(4)
    uint32_t log2_min_pcm_luma_coding_block_size_minus3; //ue(v)
    uint32_t log2_diff_max_min_pcm_luma_coding_block_size;  //ue(v)
    bool pcm_loop_filter_disabled_flag;                  //u(1)
    uint32_t num_short_term_ref_pic_sets;                //ue(v)
    //short_term_ref_pic_set(i) max is 64
    HevcShortTermRps st_rps[64];
    HevcLongTermRps lt_rps;
    bool long_term_ref_pics_present_flag;                //u(1)
    uint32_t num_long_term_ref_pics_sps;                 //ue(v)
    //max is 32
    uint32_t lt_ref_pic_poc_lsb_sps[32];                 //u(v)
    bool used_by_curr_pic_lt_sps_flag[32];               //u(1)
    bool sps_temporal_mvp_enabled_flag;                  //u(1)
    bool strong_intra_smoothing_enabled_flag;            //u(1)
    bool vui_parameters_present_flag;                    //u(1)
    //vui_parameters()
    HevcVuiParameters vui_parameters;
    bool sps_extension_flag;                             //u(1)
    bool sps_extension_data_flag;                        //u(1)
    //rbsp_trailing_bits( )
    HevcRbspTrailingBits rbsp_trailing_bits;
} HevcSeqParamSet;

/*! \brief  Structure for Picture Parameter Set
 */
typedef struct {
    uint32_t is_received;                                // received with pps_pic_parameter_set_id
    uint32_t pps_pic_parameter_set_id;                   //ue(v)
    uint32_t pps_seq_parameter_set_id;                   //ue(v)
    bool dependent_slice_segments_enabled_flag;          //u(1)
    bool output_flag_present_flag;                       //u(1)
    uint32_t num_extra_slice_header_bits;                //u(3)
    bool sign_data_hiding_enabled_flag;                  //u(1)
    bool cabac_init_present_flag;                        //u(1)
    uint32_t num_ref_idx_l0_default_active_minus1;       //ue(v)
    uint32_t num_ref_idx_l1_default_active_minus1;       //ue(v)
    int32_t init_qp_minus26;                             //se(v)
    bool constrained_intra_pred_flag;                    //u(1)
    bool transform_skip_enabled_flag;                    //u(1)
    bool cu_qp_delta_enabled_flag;                       //u(1)
    uint32_t diff_cu_qp_delta_depth;                     //ue(v)
    int32_t pps_cb_qp_offset;                            //se(v)
    int32_t pps_cr_qp_offset;                            //se(v)
    bool pps_slice_chroma_qp_offsets_present_flag;       //u(1)
    bool weighted_pred_flag;                             //u(1)
    bool weighted_bipred_flag;                           //u(1)
    bool transquant_bypass_enabled_flag;                 //u(1)
    bool tiles_enabled_flag;                             //u(1)
    bool entropy_coding_sync_enabled_flag;               //u(1)
    uint32_t num_tile_columns_minus1;                    //ue(v)
    uint32_t num_tile_rows_minus1;                       //ue(v)
    bool uniform_spacing_flag;                           //u(1)
    //PicWidthInCtbsY = Ceil( pic_width_in_luma_samples / CtbSizeY )  = 256 assume max width is 4096
    //CtbSizeY = 1<<CtbLog2SizeY   so min is 16
    // 4 <= CtbLog2SizeY <= 6
    uint32_t column_width_minus1[265];                   //ue(v)
    //2304/16=144 assume max height is 2304
    uint32_t row_height_minus1[144];                     //ue(v)
    bool loop_filter_across_tiles_enabled_flag;          //u(1)
    bool pps_loop_filter_across_slices_enabled_flag;     //u(1)
    bool deblocking_filter_control_present_flag;         //u(1)
    bool deblocking_filter_override_enabled_flag;        //u(1)
    bool pps_deblocking_filter_disabled_flag;            //u(1)
    int32_t pps_beta_offset_div2;                        //se(v)
    int32_t pps_tc_offset_div2;                          //se(v)
    bool pps_scaling_list_data_present_flag;             //u(1)
    //scaling_list_data( )
    HevcScalingListData scaling_list_data;
    bool lists_modification_present_flag;                //u(1)
    uint32_t log2_parallel_merge_level_minus2;           //ue(v)
    bool slice_segment_header_extension_present_flag;    //u(1)
    bool pps_extension_present_flag;                     //u(1)
    bool pps_range_extension_flag;                       //u(1)
    bool pps_multilayer_extension_flag;                  //u(1)
    uint32_t pps_extension_6bits;                        //u(6)
    // pps_range_extension()
    uint32_t log2_max_transform_skip_block_size_minus2;  //ue(v)
    uint8_t cross_component_prediction_enabled_flag;     //u(1)
    uint8_t chroma_qp_offset_list_enabled_flag;          //u(1)
    uint32_t diff_cu_chroma_qp_offset_depth;             //ue(v)
    uint32_t chroma_qp_offset_list_len_minus1;           //ue(v)
    int32_t cb_qp_offset_list[6];                        //se(v)
    int32_t cr_qp_offset_list[6];                        //se(v)
    uint32_t log2_sao_offset_scale_luma;                 //ue(v)
    uint32_t log2_sao_offset_scale_chroma;               //ue(v)
    bool pps_extension_data_flag;                        //u(1)
    //rbsp_trailing_bits( )
    HevcRbspTrailingBits rbsp_trailing_bits;
} HevcPicParamSet;

/*! \brief Structure for Slice Segment Header
 */
typedef struct {
    uint32_t is_received;
    bool first_slice_segment_in_pic_flag;                //u(1)
    bool no_output_of_prior_pics_flag;                   //u(1)
    uint32_t slice_pic_parameter_set_id;                 //ue(v)
    bool dependent_slice_segment_flag;                   //u(1)
    uint32_t slice_segment_address;                      //u(v)
    //num_extra_slice_header_bits is u(3), so max is 7
    bool slice_reserved_flag[7];                         //u(1)
    uint32_t slice_type;                                 //ue(v)
    bool pic_output_flag;                                //u(1)
    uint32_t colour_plane_id;                            //u(2)
    uint32_t slice_pic_order_cnt_lsb;                    //u(v)
    bool short_term_ref_pic_set_sps_flag;                //u(1)
    //short_term_ref_pic_set( num_short_term_ref_pic_sets )
    uint32_t              short_term_ref_pic_set_size; //MM
    HevcShortTermRps st_rps;
    uint32_t short_term_ref_pic_set_idx;                 //u(v)
    uint32_t num_long_term_sps;                          //ue(v)
    uint32_t num_long_term_pics;                         //ue(v)
    //num_long_term_sps + num_long_term_pics max is 32
    HevcLongTermRps lt_rps;
    uint32_t lt_idx_sps[32];                             //u(v)
    uint32_t poc_lsb_lt[32];                             //u(v)
    bool used_by_curr_pic_lt_flag[32];                   //u(1)
    bool delta_poc_msb_present_flag[32];                 //u(1)
    uint32_t delta_poc_msb_cycle_lt[32];                 //ue(v)
    bool slice_temporal_mvp_enabled_flag;                //u(1)
    bool slice_sao_luma_flag;                            //u(1)
    bool slice_sao_chroma_flag;                          //u(1)
    bool num_ref_idx_active_override_flag;               //u(1)
    uint32_t num_ref_idx_l0_active_minus1;               //ue(v)
    uint32_t num_ref_idx_l1_active_minus1;               //ue(v)
    // Reference picture list modification
    uint32_t ref_pic_list_modification_flag_l0;          //u(1)
    uint32_t list_entry_l0[16];                          //u(v)
    uint32_t ref_pic_list_modification_flag_l1;          //u(1)
    uint32_t list_entry_l1[16];                          //u(v)
    bool mvd_l1_zero_flag;                               //u(1)
    bool cabac_init_flag;                                //u(1)
    bool collocated_from_l0_flag;                        //u(1)
    HevcPredWeightTable pred_weight_table;
    uint32_t collocated_ref_idx;                         //ue(v)
    uint32_t five_minus_max_num_merge_cand;              //ue(v)
    int32_t slice_qp_delta;                              //se(v)
    int32_t slice_cb_qp_offset;                          //se(v)
    int32_t slice_cr_qp_offset;                          //se(v)
    uint8_t cu_chroma_qp_offset_enabled_flag;            //u(1)
    bool deblocking_filter_override_flag;                //u(1)
    bool slice_deblocking_filter_disabled_flag;          //u(1)
    int32_t slice_beta_offset_div2;                      //se(v)
    int32_t slice_tc_offset_div2;                        //se(v)
    bool slice_loop_filter_across_slices_enabled_flag;   //u(1)
    uint32_t num_entry_point_offsets;                    //ue(v)
    uint32_t offset_len_minus1;                          //ue(v)
    uint32_t entry_point_offset_minus1[MAX_ENTRY_POINT_OFFSETS]; //u(v)
    uint32_t slice_segment_header_extension_length;      //ue(v)
    //slice_segment_header_extension_length max is 256
    uint8_t slice_segment_header_extension_data_byte[256];    //u(8)
} HevcSliceSegHeader;

