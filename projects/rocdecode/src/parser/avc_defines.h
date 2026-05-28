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

#define AVC_MAX_SPS_NUM                                 32
#define AVC_MAX_PPS_NUM                                 256
#define AVC_MAX_SLICE_NUM                               256
#define AVC_MAX_CPB_COUNT                               32
#define AVC_MAX_NUM_SLICE_GROUPS_MINUS                  8
#define AVC_MAX_NUM_REF_FRAMES_IN_PIC_ORDER_CNT_CYCLE   256

#define AVC_MAX_REF_FRAME_NUM                           16
#define AVC_MAX_REF_PICTURE_NUM                         32
#define AVC_MAX_DPB_FRAMES                              16
#define AVC_MAX_DPB_FIELDS                              AVC_MAX_DPB_FRAMES * 2

#define AVC_MACRO_BLOCK_SIZE                            16

#define NO_LONG_TERM_FRAME_INDICES                      -1

// AVC spec. Table 7-1 – NAL unit type codes, syntax element categories, and NAL unit type classes.
enum AvcNalUnitType {
    kAvcNalTypeUnspecified                    = 0, 
    kAvcNalTypeSlice_Non_IDR                  = 1,
    kAvcNalTypeSlice_Data_Partition_A         = 2,
    kAvcNalTypeSlice_Data_Partition_B         = 3,
    kAvcNalTypeSlice_Data_Partition_C         = 4,
    kAvcNalTypeSlice_IDR                      = 5,
    kAvcNalTypeSEI_Info                       = 6,
    kAvcNalTypeSeq_Parameter_Set              = 7,
    kAvcNalTypePic_Parameter_Set              = 8,
    kAvcNalTypeAccess_Unit_Delimiter          = 9,
    kAvcNalTypeEnd_Of_Seq                     = 10,
    kAvcNalTypeEnd_Of_Stream                  = 11,
    kAvcNalTypeFiller_Data                    = 12,
    kAvcNalTypeSeq_Parameter_Set_Ext          = 13,
    kAvcNalTypePrefix_NAL_Unit                = 14,
    kAvcNalTypeSubset_Seq_Parameter_Set       = 15,
    kAvcNalTypeDepth_Parameter_Set            = 16,
};

// AVC spec. Table 7-6 – Name association to slice_type.
enum AvcSliceType {
    kAvcSliceTypeP                    = 0, 
    kAvcSliceTypeB                    = 1, 
    kAvcSliceTypeI                    = 2, 
    kAvcSliceTypeSP                   = 3, 
    kAvcSliceTypeSI                   = 4, 
    kAvcSliceTypeP_5                  = 5, 
    kAvcSliceTypeB_6                  = 6, 
    kAvcSliceTypeI_7                  = 7,
    kAvcSliceTypeSP_8                 = 8, 
    kAvcSliceTypeSI_9                 = 9
};

// AVC spec. Annex A.2.
enum AvcProfile {
    kAvcBaselineProfile        = 66,
    kAvcMainProfile            = 77,
    kAvcExtendedProfile        = 88,
    kAvcHighProfile            = 100,
    kAvcHigh10Profile          = 110,
    kAvcHigh422Profile         = 122,
    kAvcHigh444Profile         = 144
};

// AVC spec. Annex A.3.
enum AvcLevel {
    kAvcLevel1      = 10,
    kAvcLevel1_1    = 11,
    kAvcLevel1_2    = 12,
    kAvcLevel1_3    = 13,
    kAvcLevel2      = 20,
    kAvcLevel2_1    = 21,
    kAvcLevel2_2    = 22,
    kAvcLevel3      = 30,
    kAvcLevel3_1    = 31,
    kAvcLevel3_2    = 32,
    kAvcLevel4      = 40,
    kAvcLevel4_1    = 41,
    kAvcLevel4_2    = 42,
    kAvcLevel5      = 50,
    kAvcLevel5_1    = 51,
    kAvcLevel5_2    = 52,
    kAvcLevel6      = 60,
    kAvcLevel6_1    = 61,
    kAvcLevel6_2    = 62
};

// NAL unit header syntax. AVC spec. Section 7.3.1.
typedef struct {
    uint32_t    forbidden_zero_bit;     // f(1)
    uint32_t    nal_ref_idc;            // u(2)
    uint32_t    nal_unit_type;          // u(5)
} AvcNalUnitHeader;

// HRD parameters syntax. AVC spec. Section E.1.2.
typedef struct {
    uint32_t    cpb_cnt_minus1;                                     // ue(v)
    uint32_t    bit_rate_scale;                                     // u(4)
    uint32_t    cpb_size_scale;                                     // u(4)
    uint32_t    bit_rate_value_minus1[AVC_MAX_CPB_COUNT];           // ue(v)
    uint32_t    cpb_size_value_minus1[AVC_MAX_CPB_COUNT];           // ue(v)
    uint32_t    cbr_flag[AVC_MAX_CPB_COUNT];                        // u(1)
    uint32_t    initial_cpb_removal_delay_length_minus1;            // u(5)
    uint32_t    cpb_removal_delay_length_minus1;                    // u(5
    uint32_t    dpb_output_delay_length_minus1;                     // u(5)
    uint32_t    time_offset_length;                                 // u(5)
} AvcHrdParameters;

// VUI parameters syntax. AVC spec. Section E.1.1.
typedef struct {
    uint32_t            aspect_ratio_info_present_flag;                         // u(1)
    uint32_t            aspect_ratio_idc;                                       // u(8)
    uint32_t            sar_width;                                              // u(16)
    uint32_t            sar_height;                                             // u(16)
    uint32_t            overscan_info_present_flag;                             // u(1)
    uint32_t            overscan_appropriate_flag;                              // u(1)
    uint32_t            video_signal_type_present_flag;                         // u(1)
    uint32_t            video_format;                                           // u(3)
    uint32_t            video_full_range_flag;                                  // u(1)
    uint32_t            colour_description_present_flag;                        // u(1)
    uint32_t            colour_primaries;                                       // u(8)
    uint32_t            transfer_characteristics;                               // u(8)
    uint32_t            matrix_coefficients;                                    // u(8)
    uint32_t            chroma_loc_info_present_flag;                           // u(1)
    uint32_t            chroma_sample_loc_type_top_field;                       // ue(v)
    uint32_t            chroma_sample_loc_type_bottom_field;                    // ue(v)
    uint32_t            timing_info_present_flag;                               // u(1)
    uint32_t            num_units_in_tick;                                      // u(32)
    uint32_t            time_scale;                                             // u(32)
    uint32_t            fixed_frame_rate_flag;                                  // u(1)
    uint32_t            nal_hrd_parameters_present_flag;                        // u(1)
    AvcHrdParameters    nal_hrd_parameters;                                     // hrd_paramters()
    uint32_t            vcl_hrd_parameters_present_flag;                        // u(1)
    AvcHrdParameters    vcl_hrd_parameters;                                     // hrd_paramters()
    uint32_t            low_delay_hrd_flag;                                     // u(1)
    uint32_t            pic_struct_present_flag;                                // u(1)
    uint32_t            bitstream_restriction_flag;                             // u(1)
    uint32_t            motion_vectors_over_pic_boundaries_flag;                // u(1)
    uint32_t            max_bytes_per_pic_denom;                                // ue(v)
    uint32_t            max_bits_per_mb_denom;                                  // ue(v)
    uint32_t            log2_max_mv_length_vertical;                            // ue(v)
    uint32_t            log2_max_mv_length_horizontal;                          // ue(v)
    uint32_t            max_num_reorder_frames;                                 // ue(v)
    uint32_t            max_dec_frame_buffering;                                // ue(v)
} AvcVuiSeqParameters;

// Sequence parameter set data syntax. AVC spec. 7.3.2.1.1.
typedef struct {
    uint32_t    is_received;                                                            // received with seq_parameter_set_id
    uint32_t    profile_idc;                                                            // u(8)
    uint32_t    constraint_set0_flag;                                                   // u(1)
    uint32_t    constraint_set1_flag;                                                   // u(1)
    uint32_t    constraint_set2_flag;                                                   // u(1)
    uint32_t    constraint_set3_flag;                                                   // u(1)
    uint32_t    constraint_set4_flag;                                                   // u(1)
    uint32_t    constraint_set5_flag;                                                   // u(1)
    uint32_t    reserved_zero_2bits;                                                    // u(2)
    uint32_t    level_idc;                                                              // u(8)
    uint32_t    seq_parameter_set_id;                                                   // ue(v)
    uint32_t    chroma_format_idc;                                                      // ue(v)
    uint32_t    separate_colour_plane_flag;                                             // u(1)
    uint32_t    bit_depth_luma_minus8;                                                  // ue(v)
    uint32_t    bit_depth_chroma_minus8;                                                // ue(v)
    uint32_t    qpprime_y_zero_transform_bypass_flag;                                   // u(1)
    uint32_t    seq_scaling_matrix_present_flag;                                        // u(1)
    uint32_t    seq_scaling_list_present_flag[12];                                      // u(1)
    uint32_t    scaling_list_4x4[6][16];                                                // ScalingList4x4
    uint32_t    scaling_list_8x8[6][64];                                                // ScalingList8x8
    uint32_t    use_default_scaling_matrix_4x4_flag[6];                                 // UseDefaultScalingMatrix4x4Flag
    uint32_t    use_default_scaling_matrix_8x8_flag[6];                                 // UseDefaultScalingMatrix8x8Flag
    uint32_t    log2_max_frame_num_minus4;                                              // ue(v)
    uint32_t    pic_order_cnt_type;                                                     // ue(v)
    uint32_t    log2_max_pic_order_cnt_lsb_minus4;                                      // ue(v)
    uint32_t    delta_pic_order_always_zero_flag;                                       // u(1)
    int32_t     offset_for_non_ref_pic;                                                 // se(v)
    int32_t     offset_for_top_to_bottom_field;                                         // se(v)
    uint32_t    num_ref_frames_in_pic_order_cnt_cycle;                                  // ue(v)
    int32_t     offset_for_ref_frame[AVC_MAX_NUM_REF_FRAMES_IN_PIC_ORDER_CNT_CYCLE];    // se(v)
    uint32_t    max_num_ref_frames;                                                     // ue(v)
    uint32_t    gaps_in_frame_num_value_allowed_flag;                                   // u(1)
    uint32_t    pic_width_in_mbs_minus1;                                                // ue(v)
    uint32_t    pic_height_in_map_units_minus1;                                         // ue(v)
    uint32_t    frame_mbs_only_flag;                                                    // u(1)
    uint32_t    mb_adaptive_frame_field_flag;                                           // u(1)
    uint32_t    direct_8x8_inference_flag;                                              // u(1)
    uint32_t    frame_cropping_flag;                                                    // u(1)
    uint32_t    frame_crop_left_offset;                                                 // ue(v)
    uint32_t    frame_crop_right_offset;                                                // ue(v)
    uint32_t    frame_crop_top_offset;                                                  // ue(v)
    uint32_t    frame_crop_bottom_offset;                                               // ue(v)
    uint32_t    vui_parameters_present_flag;                                            // u(1)
    AvcVuiSeqParameters vui_seq_parameters;                                             // vui_parameters()	
} AvcSeqParameterSet;

// Picture parameter set RBSP syntax. AVC Spec. 7.3.2.2.
typedef struct {
    uint32_t    is_received;                                                        // is received with pic_parameter_set_id
    uint32_t    pic_parameter_set_id;                                               // ue(v)
    uint32_t    seq_parameter_set_id;                                               // ue(v)
    uint32_t    entropy_coding_mode_flag;                                           // u(1)
    uint32_t    bottom_field_pic_order_in_frame_present_flag;                       // u(1)
    uint32_t    num_slice_groups_minus1;                                            // ue(v)
    uint32_t    slice_group_map_type;                                               // ue(v)
    uint32_t    run_length_minus1[AVC_MAX_NUM_SLICE_GROUPS_MINUS];                  // ue(v)
    uint32_t    top_left[AVC_MAX_NUM_SLICE_GROUPS_MINUS];                           // ue(v)
    uint32_t    bottom_right[AVC_MAX_NUM_SLICE_GROUPS_MINUS];                       // ue(v)
    uint32_t    slice_group_change_direction_flag;                                  // u(1)
    uint32_t    slice_group_change_rate_minus1;                                     // ue(v)
    uint32_t    pic_size_in_map_units_minus1;                                       // ue(v)
    uint32_t    *slice_group_id;                                                    // complete MBAmap u(v)
    uint32_t    num_ref_idx_l0_default_active_minus1;                               // ue(v)
    uint32_t    num_ref_idx_l1_default_active_minus1;                               // ue(v)
    uint32_t    weighted_pred_flag;                                                 // u(1)
    uint32_t    weighted_bipred_idc;                                                // u(2)
    int32_t     pic_init_qp_minus26;                                                // se(v)
    int32_t     pic_init_qs_minus26;                                                // se(v)
    int32_t	    chroma_qp_index_offset;                                             // se(v)
    uint32_t    deblocking_filter_control_present_flag;                             // u(1)
    uint32_t    constrained_intra_pred_flag;                                        // u(1)
    uint32_t    redundant_pic_cnt_present_flag;                                     // u(1)
    uint32_t    transform_8x8_mode_flag;                                            // u(1)
    uint32_t    pic_scaling_matrix_present_flag;                                    // u(1)
    uint32_t    pic_scaling_list_present_flag[12];                                  // u(1)
    uint32_t    scaling_list_4x4[6][16];                                            // ScalingList4x4
    uint32_t    scaling_list_8x8[6][64];                                            // ScalingList8x8
    uint32_t    use_default_scaling_matrix_4x4_flag[6];                             // UseDefaultScalingMatrix4x4Flag
    uint32_t    use_default_scaling_matrix_8x8_flag[6];	                            // UseDefaultScalingMatrix8x8Flag
    int32_t     second_chroma_qp_index_offset;                                      // se(v)
} AvcPicParameterSet;

// Reference picture list modification syntax. AVC spec. 7.3.3.1.
typedef struct {
	uint32_t	modification_of_pic_nums_idc;                   // ue(v)
	uint32_t	abs_diff_pic_num_minus1;                        // ue(v)
	uint32_t	long_term_pic_num;                              // ue(v)
} AvcListMod;

typedef struct {
	uint32_t        ref_pic_list_modification_flag_l0;                  // u(1)
	AvcListMod      modification_l0[AVC_MAX_REF_PICTURE_NUM];
	uint32_t        ref_pic_list_modification_flag_l1;                  // u(1)
	AvcListMod      modification_l1[AVC_MAX_REF_PICTURE_NUM];
} AvcRefPicListMod;

// Prediction weight table syntax. AVC spec. 7.3.3.2.
typedef struct {
    uint32_t    luma_weight_l0_flag;                        // u(1)
    int32_t     luma_weight_l0;                             // se(v)
    int32_t     luma_offset_l0;                             // se(v)
    uint32_t    chroma_weight_l0_flag;                      // u(1)
    int32_t     chroma_weight_l0[2];                        // se(v)
    int32_t     chroma_offset_l0[2];                        // se(v)
    uint32_t    luma_weight_l1_flag;                        // u(1)
    int32_t     luma_weight_l1;                             // se(v)
    int32_t     luma_offset_l1;                             // se(v)
    uint32_t    chroma_weight_l1_flag;                      // u(1)
    int32_t     chroma_weight_l1[2];                        // se(v)
    int32_t     chroma_offset_l1[2];                        // se(v)
} AvcWeightFactor;

typedef struct {
    uint32_t    luma_log2_weight_denom;                     // ue(v)
    uint32_t    chroma_log2_weight_denom;                   // ue(v)
    AvcWeightFactor	weight_factor[AVC_MAX_REF_PICTURE_NUM];		
} AvcPredWeightTable;

// Decoded reference picture marking syntax. AVC spec. 7.3.3.3.
typedef struct {
    uint32_t    memory_management_control_operation;                    // ue(v)
    uint32_t    difference_of_pic_nums_minus1;                          // ue(v)
    uint32_t    long_term_pic_num;                                      // ue(v)
    uint32_t    long_term_frame_idx;                                    // ue(v)
    uint32_t    max_long_term_frame_idx_plus1;                          // ue(v)
} AvcMmco;

typedef struct {
    uint32_t    no_output_of_prior_pics_flag;                           // u(1)
    uint32_t    long_term_reference_flag;                               // u(1)
    uint32_t    adaptive_ref_pic_marking_mode_flag;                     // u(1)
    AvcMmco	    mmco[AVC_MAX_REF_PICTURE_NUM];
    uint32_t    mmco_count;
} AvcDecRefPicMarking;

// Slice header syntax. AVC Spec. 7.3.3.
typedef struct {
    uint32_t    first_mb_in_slice;                                          // ue(v)
    uint32_t    slice_type;                                                 // ue(v)
    uint32_t    pic_parameter_set_id;                                       // ue(v)
    uint32_t    colour_plane_id;                                            // u(2)
    uint32_t    frame_num;                                                  // u(v)
    uint32_t    field_pic_flag;                                             // u(1)
    uint32_t    bottom_field_flag;                                          // u(1)
    uint32_t    idr_pic_id;                                                 // ue(v)
    uint32_t    pic_order_cnt_lsb;                                          // u(v)
    int32_t     delta_pic_order_cnt_bottom;                                 // se(v)
    int32_t     delta_pic_order_cnt[2];                                     // se(v)
    uint32_t    redundant_pic_cnt;                                          // ue(v)
    uint32_t    direct_spatial_mv_pred_flag;                                // u(1)
    uint32_t    num_ref_idx_active_override_flag;                           // u(1)
    uint32_t    num_ref_idx_l0_active_minus1;                               // ue(v)
    uint32_t    num_ref_idx_l1_active_minus1;                               // ue(v)
    AvcRefPicListMod            ref_pic_list;		
    AvcPredWeightTable          pred_weight_table;	
    AvcDecRefPicMarking         dec_ref_pic_marking;	
    uint32_t    cabac_init_idc;                                             // ue(v)
    int32_t     slice_qp_delta;                                             // se(v)
    uint32_t    sp_for_switch_flag;                                         // u(1)
    int32_t     slice_qs_delta;                                             // se(v)
    uint32_t    disable_deblocking_filter_idc;                              // ue(v)
    int32_t     slice_alpha_c0_offset_div2;                                 // se(v)
    int32_t     slice_beta_offset_div2;                                     // se(v)
    uint32_t    slice_group_change_cycle;                                   // u(v)
} AvcSliceHeader;