/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hevc_parser.h"

HevcVideoParser::HevcVideoParser() {
    first_pic_after_eos_nal_unit_ = 0;
    slice_info_list_.assign(INIT_SLICE_LIST_NUM, {{0}});
    active_vps_id_ = -1;
    active_sps_id_ = -1;
    active_pps_id_ = -1;
    slice_info_list_.assign(INIT_SLICE_LIST_NUM, {{0}});
    slice_param_list_.assign(INIT_SLICE_LIST_NUM, {0});
    memset(&curr_pic_info_, 0, sizeof(HevcPicInfo));
    for (int i = 0; i < MAX_VPS_COUNT; i++) {
        vps_list_[i].is_received = 0;
    }
    for (int i = 0; i < MAX_SPS_COUNT; i++) {
        sps_list_[i].is_received = 0;
    }
    for (int i = 0; i < MAX_PPS_COUNT; i++) {
        pps_list_[i].is_received = 0;
    }
    InitDpb();
}

HevcVideoParser::~HevcVideoParser() {
}

rocDecStatus HevcVideoParser::Initialize(RocdecParserParams *p_params) {
    return RocVideoParser::Initialize(p_params);
}

rocDecStatus HevcVideoParser::UnInitialize() {
    //todo:: do any uninitialization here
    return ROCDEC_SUCCESS;
}

rocDecStatus HevcVideoParser::ParseVideoData(RocdecSourceDataPacket *p_data) {
    if (p_data->payload && p_data->payload_size) {
        curr_pts_ = p_data->pts;
        if (ParsePictureData(p_data->payload, p_data->payload_size) != PARSER_OK) {
            logger_.ErrorLog(MakeMsg(STR("Parser failed!")));
            return ROCDEC_RUNTIME_ERROR;
        }

        // Init Roc decoder for the first time or reconfigure the existing decoder
        if (new_seq_activated_) {
            if (FillSeqCallbackFn(&sps_list_[active_sps_id_]) != PARSER_OK) {
                return ROCDEC_RUNTIME_ERROR;
            }
            new_seq_activated_ = false;
        }

        // Whenever new sei message found
        if (pfn_get_sei_message_cb_ && sei_message_count_ > 0) {
            SendSeiMsgPayload();
        }

        // Error handling: if there is no slice data, return gracefully.
        if (num_slices_ == 0) {
            return ROCDEC_SUCCESS;
        }

        // Decode the picture
        if (SendPicForDecode() != PARSER_OK) {
            logger_.ErrorLog(MakeMsg(STR("Failed to decode!")));
            return ROCDEC_RUNTIME_ERROR;
        }

        // Output decoded pictures from DPB if any are ready
        if (pfn_display_picture_cb_ && num_output_pics_ > 0) {
            if (OutputDecodedPictures(false) != PARSER_OK) {
                return ROCDEC_RUNTIME_ERROR;
            }
        }

        pic_count_++;
    } else if (!(p_data->flags & ROCDEC_PKT_ENDOFSTREAM)) {
        // If no payload and EOS is not set, treated as invalid.
        return ROCDEC_INVALID_PARAMETER;
    }

    if (p_data->flags & ROCDEC_PKT_ENDOFSTREAM) {
        if (FlushDpb() != PARSER_OK) {
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    return ROCDEC_SUCCESS;
}

int HevcVideoParser::FillSeqCallbackFn(HevcSeqParamSet* sps_data) {
    video_format_params_.codec = rocDecVideoCodec_HEVC;
    video_format_params_.frame_rate.numerator = frame_rate_.numerator;
    video_format_params_.frame_rate.denominator = frame_rate_.denominator;
    video_format_params_.bit_depth_luma_minus8 = sps_data->bit_depth_luma_minus8;
    video_format_params_.bit_depth_chroma_minus8 = sps_data->bit_depth_chroma_minus8;
    if (sps_data->profile_tier_level.general_progressive_source_flag && !sps_data->profile_tier_level.general_interlaced_source_flag)
        video_format_params_.progressive_sequence = 1;
    else if (!sps_data->profile_tier_level.general_progressive_source_flag && sps_data->profile_tier_level.general_interlaced_source_flag)
        video_format_params_.progressive_sequence = 0;
    else // default value
        video_format_params_.progressive_sequence = 1;
    video_format_params_.min_num_decode_surfaces = dec_buf_pool_size_;
    video_format_params_.coded_width = sps_data->pic_width_in_luma_samples;
    video_format_params_.coded_height = sps_data->pic_height_in_luma_samples;
    video_format_params_.chroma_format = static_cast<rocDecVideoChromaFormat>(sps_data->chroma_format_idc);
    int sub_width_c, sub_height_c;
    switch (video_format_params_.chroma_format) {
        case rocDecVideoChromaFormat_Monochrome: {
            sub_width_c = 1;
            sub_height_c = 1;
            break;
        }
        case rocDecVideoChromaFormat_420: {
            sub_width_c = 2;
            sub_height_c = 2;
            break;
        }
        case rocDecVideoChromaFormat_422: {
            sub_width_c = 2;
            sub_height_c = 1;
            break;
        }
        case rocDecVideoChromaFormat_444: {
            sub_width_c = 1;
            sub_height_c = 1;
            break;
        }
        default:
            logger_.ErrorLog(MakeMsg(STR("Error: Sequence Callback function - Chroma Format is not supported")));
            return PARSER_FAIL;
    }
    if(sps_data->conformance_window_flag) {
        video_format_params_.display_area.left = sub_width_c * sps_data->conf_win_left_offset;
        video_format_params_.display_area.top = sub_height_c * sps_data->conf_win_top_offset;
        video_format_params_.display_area.right = sps_data->pic_width_in_luma_samples - (sub_width_c * sps_data->conf_win_right_offset);
        video_format_params_.display_area.bottom = sps_data->pic_height_in_luma_samples - (sub_height_c * sps_data->conf_win_bottom_offset);
    }  else { // default values
        video_format_params_.display_area.left = 0;
        video_format_params_.display_area.top = 0;
        video_format_params_.display_area.right = video_format_params_.coded_width;
        video_format_params_.display_area.bottom = video_format_params_.coded_height;
    }
    
    video_format_params_.bitrate = 0;

    // Dispaly aspect ratio
    // Table E-1.
    static const Rational hevc_sar[] = {
        {0, 0}, // unspecified
        {1, 1}, {12, 11}, {10, 11}, {16, 11}, {40, 33}, {24, 11}, {20, 11}, {32, 11},
        {80, 33}, {18, 11}, {15, 11}, {64, 33}, {160, 99}, {4, 3}, {3, 2}, {2, 1},
    };
    Rational sar;
    sar.numerator = 1; // set to square pixel if not present or unspecified
    sar.denominator = 1; // set to square pixel if not present or unspecified
    if (sps_data->vui_parameters_present_flag) {
        if (sps_data->vui_parameters.aspect_ratio_info_present_flag) {
            if (sps_data->vui_parameters.aspect_ratio_idc == 255 /*Extended_SAR*/) {
                sar.numerator = sps_data->vui_parameters.sar_width;
                sar.denominator = sps_data->vui_parameters.sar_height;
            } else if (sps_data->vui_parameters.aspect_ratio_idc > 0 && sps_data->vui_parameters.aspect_ratio_idc < 17) {
                sar = hevc_sar[sps_data->vui_parameters.aspect_ratio_idc];
            }
        }
    }
    int disp_width = (video_format_params_.display_area.right - video_format_params_.display_area.left) * sar.numerator;
    int disp_height = (video_format_params_.display_area.bottom - video_format_params_.display_area.top) * sar.denominator;
    int gcd = std::__gcd(disp_width, disp_height); // greatest common divisor
    if (gcd) {
        video_format_params_.display_aspect_ratio.x = disp_width / gcd;
        video_format_params_.display_aspect_ratio.y = disp_height / gcd;
    }

    video_format_params_.reconfig_options = ROCDEC_RECONFIG_NEW_SURFACES;
    video_format_params_.video_signal_description = {0};
    if (sps_data->vui_parameters_present_flag) {
        video_format_params_.video_signal_description.video_format = sps_data->vui_parameters.video_format;
        video_format_params_.video_signal_description.video_full_range_flag = sps_data->vui_parameters.video_full_range_flag;
        video_format_params_.video_signal_description.color_primaries = sps_data->vui_parameters.colour_primaries;
        video_format_params_.video_signal_description.transfer_characteristics = sps_data->vui_parameters.transfer_characteristics;
        video_format_params_.video_signal_description.matrix_coefficients = sps_data->vui_parameters.matrix_coeffs;
        video_format_params_.video_signal_description.reserved_zero_bits = 0;
    }
    video_format_params_.seqhdr_data_length = 0;

    // callback function with RocdecVideoFormat params filled out
    if (pfn_sequence_cb_(parser_params_.user_data, &video_format_params_) == 0) {
        logger_.ErrorLog(MakeMsg("Sequence callback function failed."));
        return PARSER_FAIL;
    } else {
        return PARSER_OK;
    }
}

void HevcVideoParser::SendSeiMsgPayload() {
    sei_message_info_params_.sei_message_count = sei_message_count_;
    sei_message_info_params_.sei_message = sei_message_list_.data();
    sei_message_info_params_.sei_data = (void*)sei_payload_buf_;
    sei_message_info_params_.picIdx = curr_pic_info_.dec_buf_idx;

    // callback function with RocdecSeiMessageInfo params filled out
    if (pfn_get_sei_message_cb_) pfn_get_sei_message_cb_(parser_params_.user_data, &sei_message_info_params_);
}

int HevcVideoParser::SendPicForDecode() {
    int i, j, ref_idx, buf_idx;
    HevcSeqParamSet *sps_ptr = &sps_list_[active_sps_id_];
    HevcPicParamSet *pps_ptr = &pps_list_[active_pps_id_];
    dec_pic_params_ = {0};

    dec_pic_params_.pic_width = sps_ptr->pic_width_in_luma_samples;
    dec_pic_params_.pic_height = sps_ptr->pic_height_in_luma_samples;
    dec_pic_params_.curr_pic_idx = curr_pic_info_.dec_buf_idx;
    dec_pic_params_.field_pic_flag = sps_ptr->profile_tier_level.general_interlaced_source_flag;
    dec_pic_params_.bottom_field_flag = 0; // For now. Need to parse VUI/SEI pic_timing()
    dec_pic_params_.second_field = 0; // For now. Need to parse VUI/SEI pic_timing()

    dec_pic_params_.bitstream_data_len = pic_stream_data_size_;
    dec_pic_params_.bitstream_data = pic_stream_data_ptr_;
    dec_pic_params_.num_slices = num_slices_;

    dec_pic_params_.ref_pic_flag = 1;  // HEVC decoded picture is always marked as short term at first.
    dec_pic_params_.intra_pic_flag = slice_info_list_[0].slice_header.slice_type == HEVC_SLICE_TYPE_I ? 1 : 0;

    // Todo: field_pic_flag, bottom_field_flag, second_field, ref_pic_flag, and intra_pic_flag seems to be associated with AVC/H.264.
    // Do we need them for general purpose? Reomve if not.

    // Fill picture parameters
    RocdecHevcPicParams *pic_param_ptr = &dec_pic_params_.pic_params.hevc;

    // Current picture
    pic_param_ptr->curr_pic.pic_idx = curr_pic_info_.dec_buf_idx;
    pic_param_ptr->curr_pic.poc = curr_pic_info_.pic_order_cnt;

    // Reference pictures
    ref_idx = 0;
    for (i = 0; i < num_poc_st_curr_before_; i++) {
        buf_idx = ref_pic_set_st_curr_before_[i];  // buffer index in DPB
        pic_param_ptr->ref_frames[ref_idx].pic_idx = dpb_buffer_.frame_buffer_list[buf_idx].dec_buf_idx;
        pic_param_ptr->ref_frames[ref_idx].poc = dpb_buffer_.frame_buffer_list[buf_idx].pic_order_cnt;
        pic_param_ptr->ref_frames[ref_idx].flags = 0; // assume frame picture for now
        pic_param_ptr->ref_frames[ref_idx].flags |= RocdecHevcPicture_RPS_ST_CURR_BEFORE;
        ref_idx++;
    }

    for (i = 0; i < num_poc_st_curr_after_; i++) {
        buf_idx = ref_pic_set_st_curr_after_[i]; // buffer index in DPB
        pic_param_ptr->ref_frames[ref_idx].pic_idx = dpb_buffer_.frame_buffer_list[buf_idx].dec_buf_idx;
        pic_param_ptr->ref_frames[ref_idx].poc = dpb_buffer_.frame_buffer_list[buf_idx].pic_order_cnt;
        pic_param_ptr->ref_frames[ref_idx].flags = 0; // assume frame picture for now
        pic_param_ptr->ref_frames[ref_idx].flags |= RocdecHevcPicture_RPS_ST_CURR_AFTER;
        ref_idx++;
    }

    for (i = 0; i < num_poc_lt_curr_; i++) {
        buf_idx = ref_pic_set_lt_curr_[i]; // buffer index in DPB
        pic_param_ptr->ref_frames[ref_idx].pic_idx = dpb_buffer_.frame_buffer_list[buf_idx].dec_buf_idx;
        pic_param_ptr->ref_frames[ref_idx].poc = dpb_buffer_.frame_buffer_list[buf_idx].pic_order_cnt;
        pic_param_ptr->ref_frames[ref_idx].flags = 0; // assume frame picture for now
        pic_param_ptr->ref_frames[ref_idx].flags |= RocdecHevcPicture_LONG_TERM_REFERENCE | RocdecHevcPicture_RPS_LT_CURR;
        ref_idx++;
    }

    for (i = 0; i < num_poc_st_foll_; i++) {
        buf_idx = ref_pic_set_st_foll_[i]; // buffer index in DPB
        pic_param_ptr->ref_frames[ref_idx].pic_idx = dpb_buffer_.frame_buffer_list[buf_idx].dec_buf_idx;
        pic_param_ptr->ref_frames[ref_idx].poc = dpb_buffer_.frame_buffer_list[buf_idx].pic_order_cnt;
        pic_param_ptr->ref_frames[ref_idx].flags = 0; // assume frame picture for now
        ref_idx++;
    }

    for (i = 0; i < num_poc_lt_foll_; i++) {
        buf_idx = ref_pic_set_lt_foll_[i]; // buffer index in DPB
        pic_param_ptr->ref_frames[ref_idx].pic_idx = dpb_buffer_.frame_buffer_list[buf_idx].dec_buf_idx;
        pic_param_ptr->ref_frames[ref_idx].poc = dpb_buffer_.frame_buffer_list[buf_idx].pic_order_cnt;
        pic_param_ptr->ref_frames[ref_idx].flags = 0; // assume frame picture for now
        ref_idx++;
    }

    for (i = ref_idx; i < 15; i++) {
        pic_param_ptr->ref_frames[i].pic_idx = 0xFF;
    }

    pic_param_ptr->picture_width_in_luma_samples = sps_ptr->pic_width_in_luma_samples;
    pic_param_ptr->picture_height_in_luma_samples = sps_ptr->pic_height_in_luma_samples;

    pic_param_ptr->pic_fields.bits.chroma_format_idc = sps_ptr->chroma_format_idc;
    pic_param_ptr->pic_fields.bits.separate_colour_plane_flag = sps_ptr->separate_colour_plane_flag;
    pic_param_ptr->pic_fields.bits.pcm_enabled_flag = sps_ptr->pcm_enabled_flag;
    pic_param_ptr->pic_fields.bits.scaling_list_enabled_flag = sps_ptr->scaling_list_enabled_flag;
    pic_param_ptr->pic_fields.bits.transform_skip_enabled_flag = pps_ptr->transform_skip_enabled_flag;
    pic_param_ptr->pic_fields.bits.amp_enabled_flag = sps_ptr->amp_enabled_flag;
    pic_param_ptr->pic_fields.bits.strong_intra_smoothing_enabled_flag = sps_ptr->strong_intra_smoothing_enabled_flag;
    pic_param_ptr->pic_fields.bits.sign_data_hiding_enabled_flag = pps_ptr->sign_data_hiding_enabled_flag;
    pic_param_ptr->pic_fields.bits.constrained_intra_pred_flag = pps_ptr->constrained_intra_pred_flag;
    pic_param_ptr->pic_fields.bits.cu_qp_delta_enabled_flag = pps_ptr->cu_qp_delta_enabled_flag;
    pic_param_ptr->pic_fields.bits.weighted_pred_flag = pps_ptr->weighted_pred_flag;
    pic_param_ptr->pic_fields.bits.weighted_bipred_flag = pps_ptr->weighted_bipred_flag;
    pic_param_ptr->pic_fields.bits.transquant_bypass_enabled_flag = pps_ptr->transquant_bypass_enabled_flag;
    pic_param_ptr->pic_fields.bits.tiles_enabled_flag = pps_ptr->tiles_enabled_flag;
    pic_param_ptr->pic_fields.bits.entropy_coding_sync_enabled_flag = pps_ptr->entropy_coding_sync_enabled_flag;
    pic_param_ptr->pic_fields.bits.pps_loop_filter_across_slices_enabled_flag = pps_ptr->pps_loop_filter_across_slices_enabled_flag;
    pic_param_ptr->pic_fields.bits.loop_filter_across_tiles_enabled_flag = pps_ptr->loop_filter_across_tiles_enabled_flag;
    pic_param_ptr->pic_fields.bits.pcm_loop_filter_disabled_flag = sps_ptr->pcm_loop_filter_disabled_flag;
    pic_param_ptr->pic_fields.bits.no_pic_reordering_flag = sps_ptr->sps_max_num_reorder_pics[0] ? 0 : 1;
    pic_param_ptr->pic_fields.bits.no_bi_pred_flag = slice_info_list_[0].slice_header.slice_type == HEVC_SLICE_TYPE_B ? 0 : 1;

    pic_param_ptr->sps_max_dec_pic_buffering_minus1 = sps_ptr->sps_max_dec_pic_buffering_minus1[sps_ptr->sps_max_sub_layers_minus1];  // HighestTid
    pic_param_ptr->bit_depth_luma_minus8 = sps_ptr->bit_depth_luma_minus8;
    pic_param_ptr->bit_depth_chroma_minus8 = sps_ptr->bit_depth_chroma_minus8;
    pic_param_ptr->pcm_sample_bit_depth_luma_minus1 = sps_ptr->pcm_sample_bit_depth_luma_minus1;
    pic_param_ptr->pcm_sample_bit_depth_chroma_minus1 = sps_ptr->pcm_sample_bit_depth_chroma_minus1;
    pic_param_ptr->log2_min_luma_coding_block_size_minus3 = sps_ptr->log2_min_luma_coding_block_size_minus3;
    pic_param_ptr->log2_diff_max_min_luma_coding_block_size = sps_ptr->log2_diff_max_min_luma_coding_block_size;
    pic_param_ptr->log2_min_luma_transform_block_size_minus2 = sps_ptr->log2_min_luma_transform_block_size_minus2;
    pic_param_ptr->log2_diff_max_min_luma_transform_block_size = sps_ptr->log2_diff_max_min_luma_transform_block_size;
    pic_param_ptr->log2_min_pcm_luma_coding_block_size_minus3 = sps_ptr->log2_min_pcm_luma_coding_block_size_minus3;
    pic_param_ptr->log2_diff_max_min_pcm_luma_coding_block_size = sps_ptr->log2_diff_max_min_pcm_luma_coding_block_size;
    pic_param_ptr->max_transform_hierarchy_depth_intra = sps_ptr->max_transform_hierarchy_depth_intra;
    pic_param_ptr->max_transform_hierarchy_depth_inter = sps_ptr->max_transform_hierarchy_depth_inter;
    pic_param_ptr->init_qp_minus26 = pps_ptr->init_qp_minus26;
    pic_param_ptr->diff_cu_qp_delta_depth = pps_ptr->diff_cu_qp_delta_depth;
    pic_param_ptr->pps_cb_qp_offset = pps_ptr->pps_cb_qp_offset;
    pic_param_ptr->pps_cr_qp_offset = pps_ptr->pps_cr_qp_offset;
    pic_param_ptr->log2_parallel_merge_level_minus2 = pps_ptr->log2_parallel_merge_level_minus2;

    if (pps_ptr->tiles_enabled_flag) {
        pic_param_ptr->num_tile_columns_minus1 = pps_ptr->num_tile_columns_minus1;
        pic_param_ptr->num_tile_rows_minus1 = pps_ptr->num_tile_rows_minus1;
        for (i = 0; i <= pps_ptr->num_tile_columns_minus1; i++) {
            pic_param_ptr->column_width_minus1[i] = pps_ptr->column_width_minus1[i];
        }
        for (i = 0; i <= pps_ptr->num_tile_rows_minus1; i++) {
            pic_param_ptr->row_height_minus1[i] = pps_ptr->row_height_minus1[i];
        }
    }

    pic_param_ptr->slice_parsing_fields.bits.lists_modification_present_flag = pps_ptr->lists_modification_present_flag;
    pic_param_ptr->slice_parsing_fields.bits.long_term_ref_pics_present_flag = sps_ptr->long_term_ref_pics_present_flag;
    pic_param_ptr->slice_parsing_fields.bits.sps_temporal_mvp_enabled_flag = sps_ptr->sps_temporal_mvp_enabled_flag;
    pic_param_ptr->slice_parsing_fields.bits.cabac_init_present_flag = pps_ptr->cabac_init_present_flag;
    pic_param_ptr->slice_parsing_fields.bits.output_flag_present_flag = pps_ptr->output_flag_present_flag;
    pic_param_ptr->slice_parsing_fields.bits.dependent_slice_segments_enabled_flag = pps_ptr->dependent_slice_segments_enabled_flag;
    pic_param_ptr->slice_parsing_fields.bits.pps_slice_chroma_qp_offsets_present_flag = pps_ptr->pps_slice_chroma_qp_offsets_present_flag;
    pic_param_ptr->slice_parsing_fields.bits.sample_adaptive_offset_enabled_flag = sps_ptr->sample_adaptive_offset_enabled_flag;
    pic_param_ptr->slice_parsing_fields.bits.deblocking_filter_override_enabled_flag = pps_ptr->deblocking_filter_override_enabled_flag;
    pic_param_ptr->slice_parsing_fields.bits.pps_disable_deblocking_filter_flag = pps_ptr->pps_deblocking_filter_disabled_flag;
    pic_param_ptr->slice_parsing_fields.bits.slice_segment_header_extension_present_flag = pps_ptr->slice_segment_header_extension_present_flag;
    pic_param_ptr->slice_parsing_fields.bits.rap_pic_flag = IsIrapPic(&slice_nal_unit_header_) ? 1 : 0;
    pic_param_ptr->slice_parsing_fields.bits.idr_pic_flag = IsIdrPic(&slice_nal_unit_header_) ? 1 : 0;
    pic_param_ptr->slice_parsing_fields.bits.intra_pic_flag = slice_info_list_[0].slice_header.slice_type == HEVC_SLICE_TYPE_I ? 1 : 0;

    pic_param_ptr->log2_max_pic_order_cnt_lsb_minus4 = sps_ptr->log2_max_pic_order_cnt_lsb_minus4;
    pic_param_ptr->num_short_term_ref_pic_sets = sps_ptr->num_short_term_ref_pic_sets;
    pic_param_ptr->num_long_term_ref_pic_sps = sps_ptr->num_long_term_ref_pics_sps;
    pic_param_ptr->num_ref_idx_l0_default_active_minus1 = pps_ptr->num_ref_idx_l0_default_active_minus1;
    pic_param_ptr->num_ref_idx_l1_default_active_minus1 = pps_ptr->num_ref_idx_l1_default_active_minus1;
    pic_param_ptr->pps_beta_offset_div2 = pps_ptr->pps_beta_offset_div2;
    pic_param_ptr->pps_tc_offset_div2 = pps_ptr->pps_tc_offset_div2;
    pic_param_ptr->num_extra_slice_header_bits = pps_ptr->num_extra_slice_header_bits;

    pic_param_ptr->st_rps_bits = slice_info_list_[0].slice_header.short_term_ref_pic_set_size;

    /// Fill slice parameters
    // Resize if needed
    if (num_slices_ > slice_param_list_.size()) {
        slice_param_list_.resize(num_slices_, {0});
    }
    for (int slice_index = 0; slice_index < num_slices_; slice_index++) {
        RocdecHevcSliceParams *slice_params_ptr = &slice_param_list_[slice_index];
        HevcSliceInfo *p_slice_info = &slice_info_list_[slice_index];
        HevcSliceSegHeader *p_slice_header = &p_slice_info->slice_header;

        // We put all slices into one slice data buffer.
        slice_params_ptr->slice_data_size = p_slice_info->slice_data_size;
        slice_params_ptr->slice_data_offset = p_slice_info->slice_data_offset; // point to the start code
        slice_params_ptr->slice_data_flag = 0x00; // VA_SLICE_DATA_FLAG_ALL;
        slice_params_ptr->slice_data_byte_offset = 0;  // VCN consumes from the start code
        slice_params_ptr->slice_segment_address = p_slice_header->slice_segment_address;

        // Ref lists
        memset(slice_params_ptr->ref_pic_list, 0xFF, sizeof(slice_params_ptr->ref_pic_list));
        if (p_slice_header->slice_type != HEVC_SLICE_TYPE_I) {
            for (i = 0; i <= p_slice_header->num_ref_idx_l0_active_minus1; i++) {
                int idx = p_slice_info->ref_pic_list_0_[i]; // pic_idx of the ref pic
                int dec_buf_idx = dpb_buffer_.frame_buffer_list[idx].dec_buf_idx;
                for (j = 0; j < 15; j++) {
                    if (pic_param_ptr->ref_frames[j].pic_idx == dec_buf_idx) {
                        break;
                    }
                }
                if (j == 15) {
                    logger_.ErrorLog(MakeMsg("Could not find matching pic in ref_frames list. The slice type is P/B, and the idx from the ref_pic_list_0_ is: " + TOSTR(idx)));
                    return PARSER_FAIL;
                } else {
                    slice_params_ptr->ref_pic_list[0][i] = j;
                }
            }

            if (p_slice_header->slice_type == HEVC_SLICE_TYPE_B) {
                for (i = 0; i <= p_slice_header->num_ref_idx_l1_active_minus1; i++) {
                    int idx = p_slice_info->ref_pic_list_1_[i]; // pic_idx of the ref pic
                    int dec_buf_idx = dpb_buffer_.frame_buffer_list[idx].dec_buf_idx;
                    for (j = 0; j < 15; j++) {
                        if (pic_param_ptr->ref_frames[j].pic_idx == dec_buf_idx) {
                            break;
                        }
                    }
                    if (j == 15) {
                        logger_.ErrorLog(MakeMsg("Could not find matching pic in ref_frames list. The slice type is B, and the idx from the ref_pic_list_1_ is: " + TOSTR(idx)));
                        return PARSER_FAIL;
                    } else {
                        slice_params_ptr->ref_pic_list[1][i] = j;
                    }
                }
            }
        }

        slice_params_ptr->long_slice_flags.fields.last_slice_of_pic = (slice_index == num_slices_ - 1) ? 1 : 0;
        slice_params_ptr->long_slice_flags.fields.dependent_slice_segment_flag = p_slice_header->dependent_slice_segment_flag;
        slice_params_ptr->long_slice_flags.fields.slice_type = p_slice_header->slice_type;
        slice_params_ptr->long_slice_flags.fields.color_plane_id = p_slice_header->colour_plane_id;
        slice_params_ptr->long_slice_flags.fields.slice_sao_luma_flag = p_slice_header->slice_sao_luma_flag;
        slice_params_ptr->long_slice_flags.fields.slice_sao_chroma_flag = p_slice_header->slice_sao_chroma_flag;
        slice_params_ptr->long_slice_flags.fields.mvd_l1_zero_flag = p_slice_header->mvd_l1_zero_flag;
        slice_params_ptr->long_slice_flags.fields.cabac_init_flag = p_slice_header->cabac_init_flag;
        slice_params_ptr->long_slice_flags.fields.slice_temporal_mvp_enabled_flag = p_slice_header->slice_temporal_mvp_enabled_flag;
        slice_params_ptr->long_slice_flags.fields.slice_deblocking_filter_disabled_flag = p_slice_header->slice_deblocking_filter_disabled_flag;
        slice_params_ptr->long_slice_flags.fields.collocated_from_l0_flag = p_slice_header->collocated_from_l0_flag;
        slice_params_ptr->long_slice_flags.fields.slice_loop_filter_across_slices_enabled_flag = p_slice_header->slice_loop_filter_across_slices_enabled_flag;

        slice_params_ptr->collocated_ref_idx = p_slice_header->collocated_ref_idx;
        slice_params_ptr->num_ref_idx_l0_active_minus1 = p_slice_header->num_ref_idx_l0_active_minus1;
        slice_params_ptr->num_ref_idx_l1_active_minus1 = p_slice_header->num_ref_idx_l1_active_minus1;
        slice_params_ptr->slice_qp_delta = p_slice_header->slice_qp_delta;
        slice_params_ptr->slice_cb_qp_offset = p_slice_header->slice_cb_qp_offset;
        slice_params_ptr->slice_cr_qp_offset = p_slice_header->slice_cr_qp_offset;
        slice_params_ptr->slice_beta_offset_div2 = p_slice_header->slice_beta_offset_div2;
        slice_params_ptr->slice_tc_offset_div2 = p_slice_header->slice_tc_offset_div2;

        if ((pps_ptr->weighted_pred_flag && p_slice_header->slice_type == HEVC_SLICE_TYPE_P) || (pps_ptr->weighted_bipred_flag && p_slice_header->slice_type == HEVC_SLICE_TYPE_B)) {
            slice_params_ptr->luma_log2_weight_denom = p_slice_header->pred_weight_table.luma_log2_weight_denom;
            slice_params_ptr->delta_chroma_log2_weight_denom = p_slice_header->pred_weight_table.delta_chroma_log2_weight_denom;
            for (i = 0; i < p_slice_header->num_ref_idx_l0_active_minus1; i++) {
                slice_params_ptr->delta_luma_weight_l0[i] = p_slice_header->pred_weight_table.delta_luma_weight_l0[i];
                slice_params_ptr->luma_offset_l0[i] = p_slice_header->pred_weight_table.luma_offset_l0[i];
                slice_params_ptr->delta_chroma_weight_l0[i][0] = p_slice_header->pred_weight_table.delta_chroma_weight_l0[i][0];
                slice_params_ptr->delta_chroma_weight_l0[i][1] = p_slice_header->pred_weight_table.delta_chroma_weight_l0[i][1];
                slice_params_ptr->chroma_offset_l0[i][0] = p_slice_header->pred_weight_table.chroma_offset_l0[i][0];
                slice_params_ptr->chroma_offset_l0[i][1] = p_slice_header->pred_weight_table.chroma_offset_l0[i][1];
            }

            if (p_slice_header->slice_type == HEVC_SLICE_TYPE_B) {
                for (i = 0; i < p_slice_header->num_ref_idx_l1_active_minus1; i++) {
                    slice_params_ptr->delta_luma_weight_l1[i] = p_slice_header->pred_weight_table.delta_luma_weight_l1[i];
                    slice_params_ptr->luma_offset_l1[i] = p_slice_header->pred_weight_table.luma_offset_l1[i];
                    slice_params_ptr->delta_chroma_weight_l1[i][0] = p_slice_header->pred_weight_table.delta_chroma_weight_l1[i][0];
                    slice_params_ptr->delta_chroma_weight_l1[i][1] = p_slice_header->pred_weight_table.delta_chroma_weight_l1[i][1];
                    slice_params_ptr->chroma_offset_l1[i][0] = p_slice_header->pred_weight_table.chroma_offset_l1[i][0];
                    slice_params_ptr->chroma_offset_l1[i][1] = p_slice_header->pred_weight_table.chroma_offset_l1[i][1];
                }
            }
        }

        slice_params_ptr->five_minus_max_num_merge_cand = p_slice_header->five_minus_max_num_merge_cand;
        slice_params_ptr->num_entry_point_offsets = p_slice_header->num_entry_point_offsets;
        slice_params_ptr->entry_offset_to_subset_array = 0; // don't care
        slice_params_ptr->slice_data_num_emu_prevn_bytes = 0; // don't care
    }
    dec_pic_params_.slice_params.hevc = slice_param_list_.data();

    /// Fill scaling lists
    if (sps_ptr->scaling_list_enabled_flag) {
        RocdecHevcIQMatrix *iq_matrix_ptr = &dec_pic_params_.iq_matrix.hevc;
        HevcScalingListData *scaling_list_data_ptr = &pps_ptr->scaling_list_data;
        for (i = 0; i < 6; i++) {
            for (j = 0; j < 16; j++) {
                    iq_matrix_ptr->scaling_list_4x4[i][j] = scaling_list_data_ptr->scaling_list[0][i][j];
            }

            for (j = 0; j < 64; j++) {
                iq_matrix_ptr->scaling_list_8x8[i][j] = scaling_list_data_ptr->scaling_list[1][i][j];
                iq_matrix_ptr->scaling_list_16x16[i][j] = scaling_list_data_ptr->scaling_list[2][i][j];
                if (i < 2) {
                    iq_matrix_ptr->scaling_list_32x32[i][j] = scaling_list_data_ptr->scaling_list[3][i * 3][j];
                }
            }

            iq_matrix_ptr->scaling_list_dc_16x16[i] = scaling_list_data_ptr->scaling_list_dc_coef[0][i];
            if (i < 2) {
                iq_matrix_ptr->scaling_list_dc_32x32[i] = scaling_list_data_ptr->scaling_list_dc_coef[1][i * 3];
            }
        }
    }

#if DBGINFO
    PrintVappiBufInfo();
#endif // DBGINFO

    if (pfn_decode_picture_cb_(parser_params_.user_data, &dec_pic_params_) == 0) {
        logger_.ErrorLog(MakeMsg("Decode error occurred."));
        return PARSER_FAIL;
    } else {
        return PARSER_OK;
    }
}

ParserResult HevcVideoParser::ParsePictureData(const uint8_t* p_stream, uint32_t pic_data_size) {
    ParserResult ret = PARSER_OK;
    ParserResult ret2;

    pic_data_buffer_ptr_ = (uint8_t*)p_stream;
    pic_data_size_ = pic_data_size;
    curr_byte_offset_ = 0;
    start_code_num_ = 0;
    curr_start_code_offset_ = 0;
    next_start_code_offset_ = 0;

    num_slices_ = 0;
    sei_message_count_ = 0;
    sei_payload_size_ = 0;

    do {
        ret = GetNalUnit();
        if (ret == PARSER_NOT_FOUND) {
            logger_.ErrorLog(MakeMsg(STR("Error: no start code found in the frame data.")));
            return ret;
        }
        // Parse the NAL unit
        if (nal_unit_size_ >= 5) {
            // start code + NAL unit header = 5 bytes
            int ebsp_size = nal_unit_size_ - 5 > RBSP_BUF_SIZE ? RBSP_BUF_SIZE : nal_unit_size_ - 5; // only copy enough bytes for header parsing

            nal_unit_header_ = ParseNalUnitHeader(&pic_data_buffer_ptr_[curr_start_code_offset_ + 3]);
            switch (nal_unit_header_.nal_unit_type) {
                case NAL_UNIT_VPS: {
                    memcpy(rbsp_buf_, (pic_data_buffer_ptr_ + curr_start_code_offset_ + 5), ebsp_size);
                    rbsp_size_ = EbspToRbsp(rbsp_buf_, 0, ebsp_size);
                    ParseVps(rbsp_buf_, rbsp_size_);
                    break;
                }

                case NAL_UNIT_SPS: {
                    memcpy(rbsp_buf_, (pic_data_buffer_ptr_ + curr_start_code_offset_ + 5), ebsp_size);
                    rbsp_size_ = EbspToRbsp(rbsp_buf_, 0, ebsp_size);
                    ParseSps(rbsp_buf_, rbsp_size_);
                    break;
                }

                case NAL_UNIT_PPS: {
                    memcpy(rbsp_buf_, (pic_data_buffer_ptr_ + curr_start_code_offset_ + 5), ebsp_size);
                    rbsp_size_ = EbspToRbsp(rbsp_buf_, 0, ebsp_size);
                    ParsePps(rbsp_buf_, rbsp_size_);
                    break;
                }
                
                case NAL_UNIT_CODED_SLICE_TRAIL_R:
                case NAL_UNIT_CODED_SLICE_TRAIL_N:
                case NAL_UNIT_CODED_SLICE_TLA_R:
                case NAL_UNIT_CODED_SLICE_TSA_N:
                case NAL_UNIT_CODED_SLICE_STSA_R:
                case NAL_UNIT_CODED_SLICE_STSA_N:
                case NAL_UNIT_CODED_SLICE_BLA_W_LP:
                case NAL_UNIT_CODED_SLICE_BLA_W_RADL:
                case NAL_UNIT_CODED_SLICE_BLA_N_LP:
                case NAL_UNIT_CODED_SLICE_IDR_W_RADL:
                case NAL_UNIT_CODED_SLICE_IDR_N_LP:
                case NAL_UNIT_CODED_SLICE_CRA_NUT:
                case NAL_UNIT_CODED_SLICE_RADL_N:
                case NAL_UNIT_CODED_SLICE_RADL_R:
                case NAL_UNIT_CODED_SLICE_RASL_N:
                case NAL_UNIT_CODED_SLICE_RASL_R: {
                    // Save slice NAL unit header
                    slice_nal_unit_header_ = nal_unit_header_;

                    // Resize slice info list if needed
                    if ((num_slices_ + 1) > slice_info_list_.size()) {
                        slice_info_list_.resize(num_slices_ + 1, {{0}});
                    }

                    memcpy(rbsp_buf_, (pic_data_buffer_ptr_ + curr_start_code_offset_ + 5), ebsp_size);
                    rbsp_size_ = EbspToRbsp(rbsp_buf_, 0, ebsp_size);
                    HevcSliceSegHeader *p_slice_header = &slice_info_list_[num_slices_].slice_header;
                    if ((ret2 = ParseSliceHeader(rbsp_buf_, rbsp_size_, p_slice_header)) != PARSER_OK) {
                        // we got an error while parsing this NAL unit. ignore and continue with next NAL unit
                        break;      // ignore and continue to next nal_unit
                    }

                    slice_info_list_[num_slices_].slice_data_offset = curr_start_code_offset_;
                    slice_info_list_[num_slices_].slice_data_size = nal_unit_size_;

                    // Start decode process
                    if (num_slices_ == 0) {
                        // Use the data directly from demuxer without copying
                        pic_stream_data_ptr_ = pic_data_buffer_ptr_ + curr_start_code_offset_;
                        // Picture stream data size is calculated as the diff between the frame end and the first slice offset.
                        // This is to consider the possibility of non-slice NAL units between slices.
                        pic_stream_data_size_ = pic_data_size - curr_start_code_offset_;

                        if (IsIrapPic(&slice_nal_unit_header_)) {
                            if (IsIdrPic(&slice_nal_unit_header_) || IsBlaPic(&slice_nal_unit_header_) || pic_count_ == 0 || first_pic_after_eos_nal_unit_) {
                                no_rasl_output_flag_ = 1;
                            } else {
                                no_rasl_output_flag_ = 0;
                            }
                        }

                        if (first_pic_after_eos_nal_unit_) {
                            first_pic_after_eos_nal_unit_ = 0;  // clear the flag
                        }

                        if (IsRaslPic(&slice_nal_unit_header_) && no_rasl_output_flag_ == 1) {
                            curr_pic_info_.pic_output_flag = 0;
                        } else {
                            curr_pic_info_.pic_output_flag = p_slice_header->pic_output_flag;
                        }

                        // Get POC. 8.3.1.
                        CalculateCurrPoc();

                        // Locate a free buffer for the current picutre in decode buffer pool before output picture marking (C.5.2.2)
                        if (FindFreeInDecBufPool() != PARSER_OK) {
                            return PARSER_FAIL;
                        }

                        // Decode RPS. 8.3.2.
                        DecodeRps();
                    }

                    // Construct ref lists. 8.3.4.
                    if(p_slice_header->slice_type != HEVC_SLICE_TYPE_I) {
                        ConstructRefPicLists(&slice_info_list_[num_slices_]);
                    }

                    if (num_slices_ == 0) {
                        // C.5.2.2. Mark output buffers. (After 8.3.2.)
                        if (MarkOutputPictures() != PARSER_OK) {
                            return PARSER_FAIL;
                        }

                        // C.5.2.3. Find a free buffer in DPB and mark as used. (After 8.3.2.)
                        if (FindFreeInDpbAndMark() != PARSER_OK) {
                            return PARSER_FAIL;
                        }

#if DBGINFO
                        PrintDpb();
#endif // DBGINFO
                    }
                    num_slices_++;
                    break;
                }
                
                case NAL_UNIT_PREFIX_SEI:
                case NAL_UNIT_SUFFIX_SEI: {
                    if (pfn_get_sei_message_cb_) {
                        int sei_ebsp_size = nal_unit_size_ - 5; // copy the entire NAL unit
                        if (sei_rbsp_buf_) {
                            if (sei_ebsp_size > sei_rbsp_buf_size_) {
                                delete [] sei_rbsp_buf_;
                                sei_rbsp_buf_ = new uint8_t [sei_ebsp_size];
                                sei_rbsp_buf_size_ = sei_ebsp_size;
                            }
                        } else {
                            sei_rbsp_buf_size_ = sei_ebsp_size > INIT_SEI_PAYLOAD_BUF_SIZE ? sei_ebsp_size : INIT_SEI_PAYLOAD_BUF_SIZE;
                            sei_rbsp_buf_ = new uint8_t [sei_rbsp_buf_size_];
                        }
                        memcpy(sei_rbsp_buf_, (pic_data_buffer_ptr_ + curr_start_code_offset_ + 5), sei_ebsp_size);
                        rbsp_size_ = EbspToRbsp(sei_rbsp_buf_, 0, sei_ebsp_size);
                        ParseSeiMessage(sei_rbsp_buf_, rbsp_size_);
                    }
                    break;
                }

                case NAL_UNIT_EOS: {
                    first_pic_after_eos_nal_unit_ = 1;
                    break;
                }

                case NAL_UNIT_EOB: {
                    pic_count_ = 0;
                    break;
                }

                default:
                    break;
            }
        }

        // Break if this is the last NAL unit
        if (ret == PARSER_EOF) {
            break;
        }
    } while (1);

    return PARSER_OK;
}

void HevcVideoParser::ParsePtl(HevcProfileTierLevel *ptl, bool profile_present_flag, uint32_t max_num_sub_layers_minus1, uint8_t *nalu, size_t size, size_t& offset) {
    if (profile_present_flag) {
        ptl->general_profile_space = Parser::ReadBits(nalu, offset, 2);
        ptl->general_tier_flag = Parser::GetBit(nalu, offset);
        ptl->general_profile_idc = Parser::ReadBits(nalu, offset, 5);
        for (int i = 0; i < 32; i++) {
            ptl->general_profile_compatibility_flag[i] = Parser::GetBit(nalu, offset);
        }
        ptl->general_progressive_source_flag = Parser::GetBit(nalu, offset);
        ptl->general_interlaced_source_flag = Parser::GetBit(nalu, offset);
        ptl->general_non_packed_constraint_flag = Parser::GetBit(nalu, offset);
        ptl->general_frame_only_constraint_flag = Parser::GetBit(nalu, offset);
        // ReadBits is limited to 32
        offset += 44; // skip 44 bits
        // Todo: add constrant flags parsing for higher profiles when needed
    }

    ptl->general_level_idc = Parser::ReadBits(nalu, offset, 8);
    for(uint32_t i = 0; i < max_num_sub_layers_minus1; i++) {
        ptl->sub_layer_profile_present_flag[i] = Parser::GetBit(nalu, offset);
        ptl->sub_layer_level_present_flag[i] = Parser::GetBit(nalu, offset);
    }
    if (max_num_sub_layers_minus1 > 0) {
        for(uint32_t i = max_num_sub_layers_minus1; i < 8; i++) {               
            ptl->reserved_zero_2bits[i] = Parser::ReadBits(nalu, offset, 2);
        }
    }
    for (uint32_t i = 0; i < max_num_sub_layers_minus1; i++) {
        if (ptl->sub_layer_profile_present_flag[i]) {
            ptl->sub_layer_profile_space[i] = Parser::ReadBits(nalu, offset, 2);
            ptl->sub_layer_tier_flag[i] = Parser::GetBit(nalu, offset);
            ptl->sub_layer_profile_idc[i] = Parser::ReadBits(nalu, offset, 5);
            for (int j = 0; j < 32; j++) {
                ptl->sub_layer_profile_compatibility_flag[i][j] = Parser::GetBit(nalu, offset);
            }
            ptl->sub_layer_progressive_source_flag[i] = Parser::GetBit(nalu, offset);
            ptl->sub_layer_interlaced_source_flag[i] = Parser::GetBit(nalu, offset);
            ptl->sub_layer_non_packed_constraint_flag[i] = Parser::GetBit(nalu, offset);
            ptl->sub_layer_frame_only_constraint_flag[i] = Parser::GetBit(nalu, offset);
            // ReadBits is limited to 32
            offset += 44;  // skip 44 bits
            // Todo: add constrant flags parsing for higher profiles when needed
        }
        if (ptl->sub_layer_level_present_flag[i]) {
            ptl->sub_layer_level_idc[i] = Parser::ReadBits(nalu, offset, 8);
        }
    }
}

void HevcVideoParser::ParseSubLayerHrdParameters(HevcSubLayerHrdParameters *sub_hrd, uint32_t cpb_cnt, bool sub_pic_hrd_params_present_flag, uint8_t *nalu, size_t /*size*/, size_t& offset) {
    for (uint32_t i = 0; i <= cpb_cnt; i++) {
        sub_hrd->bit_rate_value_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
        sub_hrd->cpb_size_value_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
        if(sub_pic_hrd_params_present_flag) {
            sub_hrd->cpb_size_du_value_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            sub_hrd->bit_rate_du_value_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
        }
        sub_hrd->cbr_flag[i] = Parser::GetBit(nalu, offset);
    }
}

ParserResult HevcVideoParser::ParseHrdParameters(HevcHrdParameters *hrd, bool common_inf_present_flag, uint32_t max_num_sub_layers_minus1, uint8_t *nalu, size_t size,size_t &offset) {
    if (common_inf_present_flag) {
        hrd->nal_hrd_parameters_present_flag = Parser::GetBit(nalu, offset);
        hrd->vcl_hrd_parameters_present_flag = Parser::GetBit(nalu, offset);
        if (hrd->nal_hrd_parameters_present_flag || hrd->vcl_hrd_parameters_present_flag) {
            hrd->sub_pic_hrd_params_present_flag = Parser::GetBit(nalu, offset);
            if (hrd->sub_pic_hrd_params_present_flag) {
                hrd->tick_divisor_minus2 = Parser::ReadBits(nalu, offset, 8);
                hrd->du_cpb_removal_delay_increment_length_minus1 = Parser::ReadBits(nalu, offset, 5);
                hrd->sub_pic_cpb_params_in_pic_timing_sei_flag = Parser::GetBit(nalu, offset);
                hrd->dpb_output_delay_du_length_minus1 = Parser::ReadBits(nalu, offset, 5);
            }
            hrd->bit_rate_scale = Parser::ReadBits(nalu, offset, 4);
            hrd->cpb_size_scale = Parser::ReadBits(nalu, offset, 4);
            if (hrd->sub_pic_hrd_params_present_flag) {
                hrd->cpb_size_du_scale = Parser::ReadBits(nalu, offset, 4);
            }
            hrd->initial_cpb_removal_delay_length_minus1 = Parser::ReadBits(nalu, offset, 5);
            hrd->au_cpb_removal_delay_length_minus1 = Parser::ReadBits(nalu, offset, 5);
            hrd->dpb_output_delay_length_minus1 = Parser::ReadBits(nalu, offset, 5);
        }
    }
    for (uint32_t i = 0; i <= max_num_sub_layers_minus1; i++) {
        hrd->fixed_pic_rate_general_flag[i] = Parser::GetBit(nalu, offset);
        if (!hrd->fixed_pic_rate_general_flag[i]) {
            hrd->fixed_pic_rate_within_cvs_flag[i] = Parser::GetBit(nalu, offset);
        } else {
            hrd->fixed_pic_rate_within_cvs_flag[i] = hrd->fixed_pic_rate_general_flag[i];
        }

        if (hrd->fixed_pic_rate_within_cvs_flag[i]) {
            hrd->elemental_duration_in_tc_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("elemental_duration_in_tc_minus1", hrd->elemental_duration_in_tc_minus1[i], 0, 2047);
        } else {
            hrd->low_delay_hrd_flag[i] = Parser::GetBit(nalu, offset);
        }
        if (!hrd->low_delay_hrd_flag[i]) {
            hrd->cpb_cnt_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("cpb_cnt_minus1", hrd->cpb_cnt_minus1[i], 0, 31);
        }
        if (hrd->nal_hrd_parameters_present_flag) {
            //sub_layer_hrd_parameters( i )
            ParseSubLayerHrdParameters(&hrd->sub_layer_hrd_parameters_0[i], hrd->cpb_cnt_minus1[i], hrd->sub_pic_hrd_params_present_flag, nalu, size, offset);
        }
        if (hrd->vcl_hrd_parameters_present_flag) {
            //sub_layer_hrd_parameters( i )
            ParseSubLayerHrdParameters(&hrd->sub_layer_hrd_parameters_1[i], hrd->cpb_cnt_minus1[i], hrd->sub_pic_hrd_params_present_flag, nalu, size, offset);
        }
    }
    return PARSER_OK;
}

// Table 7-5. Default values of ScalingList[ 0 ][ matrixId ][ i ] with i = 0..15.
static const uint8_t default_scaling_list_side_id_0[] = {
    16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16
};

// Table 7-6. Default values of ScalingList[ 1..3 ][ 0..2 ][ i ] with i = 0..63.
static const uint8_t default_scaling_list_intra[] = {
    16, 16, 16, 16, 17, 18, 21, 24,
    16, 16, 16, 16, 17, 19, 22, 25,
    16, 16, 17, 18, 20, 22, 25, 29,
    16, 16, 18, 21, 24, 27, 31, 36,
    17, 17, 20, 24, 30, 35, 41, 47,
    18, 19, 22, 27, 35, 44, 54, 65,
    21, 22, 25, 31, 41, 54, 70, 88,
    24, 25, 29, 36, 47, 65, 88, 115
};

// Table 7-6. Default values of ScalingList[ 1..3 ][ 3..5 ][ i ] with i = 0..63.
static const uint8_t default_scaling_list_inter[] = {
    16, 16, 16, 16, 17, 18, 20, 24,
    16, 16, 16, 17, 18, 20, 24, 25,
    16, 16, 17, 18, 20, 24, 25, 28,
    16, 17, 18, 20, 24, 25, 28, 33,
    17, 18, 20, 24, 25, 28, 33, 41,
    18, 20, 24, 25, 28, 33, 41, 54,
    20, 24, 25, 28, 33, 41, 54, 71,
    24, 25, 28, 33, 41, 54, 71, 91
};

static const int diag_scan_4x4[16] = {
    0, 4, 1, 8,
    5, 2,12, 9,
    6, 3,13,10,
    7,14,11,15
};

static const int diag_scan_8x8[64] = {
    0, 8, 1, 16, 9, 2,24,17,
    10, 3,32,25,18,11, 4,40,
    33,26,19,12, 5,48,41,34,
    27,20,13, 6,56,49,42,35,
    28,21,14, 7,57,50,43,36,
    29,22,15,58,51,44,37,30,
    23,59,52,45,38,31,60,53,
    46,39,61,54,47,62,55,63
};

void HevcVideoParser::SetDefaultScalingList(HevcScalingListData *sl_ptr) {
    int size_id, matrix_id, i;

    // DC coefficient for 16x16 and 32x32
    for (matrix_id = 0; matrix_id < 6; matrix_id++) {
        sl_ptr->scaling_list_dc_coef[0][matrix_id] = 16;
        sl_ptr->scaling_list_dc_coef[1][matrix_id] = 16;
    }

    // sizeId 0
    for (matrix_id = 0; matrix_id < 6; matrix_id++) {
        for (i = 0; i < 16; i++) {
            sl_ptr->scaling_list[0][matrix_id][i] = default_scaling_list_side_id_0[i];
        }
    }
    // sizeId 1..3, matrixId 0..2
    for (size_id = 1; size_id <= 3; size_id++) {
        for (matrix_id = 0; matrix_id <= 2; matrix_id++) {
            for (i = 0; i < 64; i++) {
            sl_ptr->scaling_list[size_id][matrix_id][i] = default_scaling_list_intra[i];
            }
        }
    }
    // sizeId 1..3, matrixId 3..5
    for (size_id = 1; size_id <= 3; size_id++) {
        for (matrix_id = 3; matrix_id <= 5; matrix_id++) {
            for (i = 0; i < 64; i++) {
            sl_ptr->scaling_list[size_id][matrix_id][i] = default_scaling_list_inter[i];
            }
        }
    }
}

ParserResult HevcVideoParser::ParseScalingList(HevcScalingListData * sl_ptr, uint8_t *nalu, size_t size, size_t& offset, HevcSeqParamSet *sps_ptr) {
    for (int size_id = 0; size_id < 4; size_id++) {
        for (int matrix_id = 0; matrix_id < 6; matrix_id += (size_id == 3) ? 3 : 1) {
            sl_ptr->scaling_list_pred_mode_flag[size_id][matrix_id] = Parser::GetBit(nalu, offset);
            if(!sl_ptr->scaling_list_pred_mode_flag[size_id][matrix_id]) {
                sl_ptr->scaling_list_pred_matrix_id_delta[size_id][matrix_id] = Parser::ExpGolomb::ReadUe(nalu, offset);
                if (size_id == 3) {
                    CHECK_ALLOWED_RANGE("scaling_list_pred_matrix_id_delta", sl_ptr->scaling_list_pred_matrix_id_delta[size_id][matrix_id], 0, matrix_id / 3);
                } else {
                    CHECK_ALLOWED_RANGE("scaling_list_pred_matrix_id_delta", sl_ptr->scaling_list_pred_matrix_id_delta[size_id][matrix_id], 0, matrix_id);
                }
                // If scaling_list_pred_matrix_id_delta is 0, infer from default scaling list. We have filled the scaling
                // list with default values earlier.
                if (sl_ptr->scaling_list_pred_matrix_id_delta[size_id][matrix_id]) {
                    // Infer from the reference scaling list
                    int ref_matrix_id = matrix_id - sl_ptr->scaling_list_pred_matrix_id_delta[size_id][matrix_id] * (size_id == 3 ? 3 : 1);
                    int coef_num = std::min(64, (1 << (4 + (size_id << 1))));
                    for (int i = 0; i < coef_num; i++) {
                        sl_ptr->scaling_list[size_id][matrix_id][i] = sl_ptr->scaling_list[size_id][ref_matrix_id][i];
                    }

                    // Copy to DC coefficient for 16x16 or 32x32
                    if (size_id > 1) {
                        sl_ptr->scaling_list_dc_coef[size_id - 2][matrix_id] = sl_ptr->scaling_list_dc_coef[size_id - 2][ref_matrix_id];
                    }
                }                
            } else {
                int next_coef = 8;
                int coef_num = std::min(64, (1 << (4 + (size_id << 1))));
                if (size_id > 1) {
                    sl_ptr->scaling_list_dc_coef_minus8[size_id - 2][matrix_id] = Parser::ExpGolomb::ReadSe(nalu, offset);
                    CHECK_ALLOWED_RANGE("scaling_list_dc_coef_minus8", sl_ptr->scaling_list_dc_coef_minus8[size_id - 2][matrix_id], -7, 247);
                    next_coef = sl_ptr->scaling_list_dc_coef_minus8[size_id - 2][matrix_id] + 8;
                    // Record DC coefficient for 16x16 or 32x32
                    sl_ptr->scaling_list_dc_coef[size_id - 2][matrix_id] = next_coef;
                }
                for (int i = 0; i < coef_num; i++) {
                    sl_ptr->scaling_list_delta_coef = Parser::ExpGolomb::ReadSe(nalu, offset);
                    CHECK_ALLOWED_RANGE("scaling_list_delta_coef", sl_ptr->scaling_list_delta_coef, -128, 127);
                    next_coef = (next_coef + sl_ptr->scaling_list_delta_coef + 256) % 256;
                    if (size_id == 0) {
                        sl_ptr->scaling_list[size_id][matrix_id][diag_scan_4x4[i]] = next_coef;
                    } else {
                        sl_ptr->scaling_list[size_id][matrix_id][diag_scan_8x8[i]] = next_coef;
                    }
                }
            }
        }
    }

    if (sps_ptr->chroma_format_idc == 3) {
        for (int i = 0; i < 64; i++) {
            sl_ptr->scaling_list[3][1][i] = sl_ptr->scaling_list[2][1][i];
            sl_ptr->scaling_list[3][2][i] = sl_ptr->scaling_list[2][2][i];
            sl_ptr->scaling_list[3][4][i] = sl_ptr->scaling_list[2][4][i];
            sl_ptr->scaling_list[3][5][i] = sl_ptr->scaling_list[2][5][i];
        }
        sl_ptr->scaling_list_dc_coef[1][1] = sl_ptr->scaling_list_dc_coef[0][1];
        sl_ptr->scaling_list_dc_coef[1][2] = sl_ptr->scaling_list_dc_coef[0][2];
        sl_ptr->scaling_list_dc_coef[1][4] = sl_ptr->scaling_list_dc_coef[0][4];
        sl_ptr->scaling_list_dc_coef[1][5] = sl_ptr->scaling_list_dc_coef[0][5];
    }
    return PARSER_OK;
}

ParserResult HevcVideoParser::ParseShortTermRefPicSet(HevcSeqParamSet *sps_ptr, HevcShortTermRps *rps, uint32_t st_rps_idx, uint32_t number_short_term_ref_pic_sets, HevcShortTermRps rps_ref[], uint8_t *nalu, size_t /*size*/, size_t& offset) {
    int i, j;

    memset(rps, 0, sizeof(HevcShortTermRps));
     if (st_rps_idx != 0) {
        rps->inter_ref_pic_set_prediction_flag = Parser::GetBit(nalu, offset);
    } else {
        rps->inter_ref_pic_set_prediction_flag = 0;
    }
    if (rps->inter_ref_pic_set_prediction_flag) {
        if (st_rps_idx == number_short_term_ref_pic_sets) {
            rps->delta_idx_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("delta_idx_minus1", rps->delta_idx_minus1, 0, st_rps_idx - 1);
        } else {
            rps->delta_idx_minus1 = 0;
        }
        rps->delta_rps_sign = Parser::GetBit(nalu, offset);
        rps->abs_delta_rps_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("abs_delta_rps_minus1", rps->abs_delta_rps_minus1, 0, ((1 << 15) - 1));
        int ref_rps_idx = st_rps_idx - (rps->delta_idx_minus1 + 1);  // (7-59)
        int delta_rps = (1 - 2 * rps->delta_rps_sign) * (rps->abs_delta_rps_minus1 + 1);  // (7-60)

        HevcShortTermRps *ref_rps = &rps_ref[ref_rps_idx];
        for (j = 0; j <= ref_rps->num_of_delta_pocs; j++) {
            rps->used_by_curr_pic_flag[j] = Parser::GetBit(nalu, offset);
            if (!rps->used_by_curr_pic_flag[j]) {
                rps->use_delta_flag[j] = Parser::GetBit(nalu, offset);
            } else {
                rps->use_delta_flag[j] = 1;
            }
        }

        int d_poc;
        i = 0;
        for (j = ref_rps->num_positive_pics - 1; j >= 0; j--) {
            d_poc = ref_rps->delta_poc_s1[j] + delta_rps;
            if (d_poc < 0 && rps->use_delta_flag[ref_rps->num_negative_pics + j]) {
                rps->delta_poc_s0[i] = d_poc;
                rps->used_by_curr_pic_s0[i++] = rps->used_by_curr_pic_flag[ref_rps->num_negative_pics + j];
            }
        }
        if (delta_rps < 0 && rps->use_delta_flag[ref_rps->num_of_delta_pocs]) {
            rps->delta_poc_s0[i] = delta_rps;
            rps->used_by_curr_pic_s0[i++] = rps->used_by_curr_pic_flag[ref_rps->num_of_delta_pocs];
        }
        for (j = 0; j < ref_rps->num_negative_pics; j++) {
            d_poc = ref_rps->delta_poc_s0[j] + delta_rps;
            if (d_poc < 0 && rps->use_delta_flag[j]) {
                rps->delta_poc_s0[i] = d_poc;
                rps->used_by_curr_pic_s0[i++] = rps->used_by_curr_pic_flag[j];
            }
        }
        rps->num_negative_pics = i;

        i = 0;
        for (j = ref_rps->num_negative_pics - 1; j >= 0; j--) {
            d_poc = ref_rps->delta_poc_s0[j] + delta_rps;
            if (d_poc > 0 && rps->use_delta_flag[j]) {
                rps->delta_poc_s1[i] = d_poc;
                rps->used_by_curr_pic_s1[i++] = rps->used_by_curr_pic_flag[j];
            }
        }
        if (delta_rps > 0 && rps->use_delta_flag[ref_rps->num_of_delta_pocs]) {
            rps->delta_poc_s1[i] = delta_rps;
            rps->used_by_curr_pic_s1[i++] = rps->used_by_curr_pic_flag[ref_rps->num_of_delta_pocs];
        }

        for (j = 0; j < ref_rps->num_positive_pics; j++) {
            d_poc = ref_rps->delta_poc_s1[j] + delta_rps;
            if (d_poc > 0 && rps->use_delta_flag[ref_rps->num_negative_pics + j]) {
                rps->delta_poc_s1[i] = d_poc;
                rps->used_by_curr_pic_s1[i++] = rps->used_by_curr_pic_flag[ref_rps->num_negative_pics + j];
            }
        }
        rps->num_positive_pics = i;
        rps->num_of_delta_pocs = rps->num_negative_pics + rps->num_positive_pics;
    } else {
        rps->num_negative_pics = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("num_negative_pics", rps->num_negative_pics, 0, sps_ptr->sps_max_dec_pic_buffering_minus1[sps_ptr->sps_max_sub_layers_minus1]);
        rps->num_positive_pics = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("num_positive_pics", rps->num_positive_pics, 0, sps_ptr->sps_max_dec_pic_buffering_minus1[sps_ptr->sps_max_sub_layers_minus1] - rps->num_negative_pics);
        rps->num_of_delta_pocs = rps->num_negative_pics + rps->num_positive_pics;

        for (i = 0; i < rps->num_negative_pics; i++) {
            rps->delta_poc_s0_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("delta_poc_s0_minus1", rps->delta_poc_s0_minus1[i], 0, ((1 << 15) - 1));
            if (i == 0) {
                rps->delta_poc_s0[i] = -(rps->delta_poc_s0_minus1[i] + 1);
            } else {
                rps->delta_poc_s0[i] = rps->delta_poc_s0[i - 1] - (rps->delta_poc_s0_minus1[i] + 1);
            }
            rps->used_by_curr_pic_s0[i] = Parser::GetBit(nalu, offset);
        }

        for (i = 0; i < rps->num_positive_pics; i++) {
            rps->delta_poc_s1_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("delta_poc_s1_minus1", rps->delta_poc_s1_minus1[i], 0, ((1 << 15) - 1));
            if (i == 0) {
                rps->delta_poc_s1[i] = rps->delta_poc_s1_minus1[i] + 1;
            } else {
                rps->delta_poc_s1[i] = rps->delta_poc_s1[i - 1] + (rps->delta_poc_s1_minus1[i] + 1);
            }
            rps->used_by_curr_pic_s1[i] = Parser::GetBit(nalu, offset);
        }
    }
    return PARSER_OK;
}

ParserResult HevcVideoParser::ParsePredWeightTable(HevcSliceSegHeader *slice_header_ptr, int chroma_array_type, uint8_t *stream_ptr, size_t &offset) {
    HevcPredWeightTable *pred_weight_table_ptr = &slice_header_ptr->pred_weight_table;
    int chroma_log2_weight_denom; // ChromaLog2WeightDenom
    int i, j;

    pred_weight_table_ptr->luma_log2_weight_denom = Parser::ExpGolomb::ReadUe(stream_ptr, offset);
    CHECK_ALLOWED_RANGE("luma_log2_weight_denom", pred_weight_table_ptr->luma_log2_weight_denom, 0, 7);
    if (chroma_array_type) {
        pred_weight_table_ptr->delta_chroma_log2_weight_denom = Parser::ExpGolomb::ReadSe(stream_ptr, offset);
    }
    chroma_log2_weight_denom = pred_weight_table_ptr->luma_log2_weight_denom + pred_weight_table_ptr->delta_chroma_log2_weight_denom;
    for (i = 0; i <= slice_header_ptr->num_ref_idx_l0_active_minus1; i++) {
        pred_weight_table_ptr->luma_weight_l0_flag[i] = Parser::GetBit(stream_ptr, offset);
    }
    if (chroma_array_type) {
        for (i = 0; i <= slice_header_ptr->num_ref_idx_l0_active_minus1; i++) {
            pred_weight_table_ptr->chroma_weight_l0_flag[i] = Parser::GetBit(stream_ptr, offset);
        }
    }
    for (i = 0; i <= slice_header_ptr->num_ref_idx_l0_active_minus1; i++) {
        if (pred_weight_table_ptr->luma_weight_l0_flag[i]) {
            pred_weight_table_ptr->delta_luma_weight_l0[i] = Parser::ExpGolomb::ReadSe(stream_ptr, offset);
            pred_weight_table_ptr->luma_offset_l0[i] = Parser::ExpGolomb::ReadSe(stream_ptr, offset);
        }
        if (pred_weight_table_ptr->chroma_weight_l0_flag[i]) {
            for (j = 0; j < 2; j++) {
                pred_weight_table_ptr->delta_chroma_weight_l0[i][j] = Parser::ExpGolomb::ReadSe(stream_ptr, offset);
                pred_weight_table_ptr->delta_chroma_offset_l0[i][j] = Parser::ExpGolomb::ReadSe(stream_ptr, offset);
                pred_weight_table_ptr->chroma_weight_l0[i][j] = (1 << chroma_log2_weight_denom) + pred_weight_table_ptr->delta_chroma_weight_l0[i][j];
                pred_weight_table_ptr->chroma_offset_l0[i][j] = std::clamp((pred_weight_table_ptr->delta_chroma_offset_l0[i][j] - ((128 * pred_weight_table_ptr->chroma_weight_l0[i][j]) >> chroma_log2_weight_denom) + 128), -128, 127);
            }
        } else {
            pred_weight_table_ptr->chroma_weight_l0[i][0] = 1 << chroma_log2_weight_denom;
            pred_weight_table_ptr->chroma_offset_l0[i][0] = 0;
            pred_weight_table_ptr->chroma_weight_l0[i][1] = 1 << chroma_log2_weight_denom;
            pred_weight_table_ptr->chroma_offset_l0[i][1] = 0;
        }
    }

    if (slice_header_ptr->slice_type == HEVC_SLICE_TYPE_B) {
        for (i = 0; i <= slice_header_ptr->num_ref_idx_l1_active_minus1; i++) {
            pred_weight_table_ptr->luma_weight_l1_flag[i] = Parser::GetBit(stream_ptr, offset);
        }
        if (chroma_array_type) {
            for (i = 0; i <= slice_header_ptr->num_ref_idx_l1_active_minus1; i++) {
                pred_weight_table_ptr->chroma_weight_l1_flag[i] = Parser::GetBit(stream_ptr, offset);
            }
        }
        for (i = 0; i <= slice_header_ptr->num_ref_idx_l1_active_minus1; i++) {
            if (pred_weight_table_ptr->luma_weight_l1_flag[i]) {
                pred_weight_table_ptr->delta_luma_weight_l1[i] = Parser::ExpGolomb::ReadSe(stream_ptr, offset);
                CHECK_ALLOWED_RANGE("delta_luma_weight_l1", pred_weight_table_ptr->delta_luma_weight_l1[i], -128, 127);
                pred_weight_table_ptr->luma_offset_l1[i] = Parser::ExpGolomb::ReadSe(stream_ptr, offset);
            }
            if (pred_weight_table_ptr->chroma_weight_l1_flag[i]) {
                for (j = 0; j < 2; j++) {
                    pred_weight_table_ptr->delta_chroma_weight_l1[i][j] = Parser::ExpGolomb::ReadSe(stream_ptr, offset);
                    CHECK_ALLOWED_RANGE("delta_chroma_weight_l1", pred_weight_table_ptr->delta_chroma_weight_l1[i][j], -128, 127);
                    pred_weight_table_ptr->delta_chroma_offset_l1[i][j] = Parser::ExpGolomb::ReadSe(stream_ptr, offset);
                    pred_weight_table_ptr->chroma_weight_l1[i][j] = (1 << chroma_log2_weight_denom) + pred_weight_table_ptr->delta_chroma_weight_l1[i][j];
                    pred_weight_table_ptr->chroma_offset_l1[i][j] = std::clamp((pred_weight_table_ptr->delta_chroma_offset_l1[i][j] - ((128 * pred_weight_table_ptr->chroma_weight_l1[i][j]) >> chroma_log2_weight_denom) + 128), -128, 127);
                }
            } else {
                pred_weight_table_ptr->chroma_weight_l1[i][0] = 1 << chroma_log2_weight_denom;
                pred_weight_table_ptr->chroma_offset_l1[i][0] = 0;
                pred_weight_table_ptr->chroma_weight_l1[i][1] = 1 << chroma_log2_weight_denom;
                pred_weight_table_ptr->chroma_offset_l1[i][1] = 0;
            }
        }
    }
    return PARSER_OK;
}

ParserResult HevcVideoParser::ParseVui(HevcSeqParamSet *sps_ptr, uint8_t *nalu, size_t size, size_t &offset) {
    HevcVuiParameters *vui = &sps_ptr->vui_parameters;
    vui->aspect_ratio_info_present_flag = Parser::GetBit(nalu, offset);
    if (vui->aspect_ratio_info_present_flag) {
        vui->aspect_ratio_idc = Parser::ReadBits(nalu, offset, 8);
        if (vui->aspect_ratio_idc == 255) {
            vui->sar_width = Parser::ReadBits(nalu, offset, 16);
            vui->sar_height = Parser::ReadBits(nalu, offset, 16);
        }
    }
    vui->overscan_info_present_flag = Parser::GetBit(nalu, offset);
    if (vui->overscan_info_present_flag) {
        vui->overscan_appropriate_flag = Parser::GetBit(nalu, offset);
    }
    vui->video_signal_type_present_flag = Parser::GetBit(nalu, offset);
    if (vui->video_signal_type_present_flag) {
        vui->video_format = Parser::ReadBits(nalu, offset, 3);
        vui->video_full_range_flag = Parser::GetBit(nalu, offset);
        vui->colour_description_present_flag = Parser::GetBit(nalu, offset);
        if (vui->colour_description_present_flag) {
            vui->colour_primaries = Parser::ReadBits(nalu, offset, 8);
            vui->transfer_characteristics = Parser::ReadBits(nalu, offset, 8);
            vui->matrix_coeffs = Parser::ReadBits(nalu, offset, 8);
        }
    }
    vui->chroma_loc_info_present_flag = Parser::GetBit(nalu, offset);
    if (vui->chroma_loc_info_present_flag) {
        vui->chroma_sample_loc_type_top_field = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("chroma_sample_loc_type_top_field", vui->chroma_sample_loc_type_top_field, 0, 5);
        vui->chroma_sample_loc_type_bottom_field = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("chroma_sample_loc_type_bottom_field", vui->chroma_sample_loc_type_bottom_field, 0, 5);
    }
    vui->neutral_chroma_indication_flag = Parser::GetBit(nalu, offset);
    vui->field_seq_flag = Parser::GetBit(nalu, offset);
    vui->frame_field_info_present_flag = Parser::GetBit(nalu, offset);
    vui->default_display_window_flag = Parser::GetBit(nalu, offset);
    if (vui->default_display_window_flag) {
        vui->def_disp_win_left_offset = Parser::ExpGolomb::ReadUe(nalu, offset);
        vui->def_disp_win_right_offset = Parser::ExpGolomb::ReadUe(nalu, offset);
        uint32_t left_offset = sps_ptr->conf_win_left_offset + vui->def_disp_win_left_offset;
        uint32_t right_offset = sps_ptr->conf_win_right_offset + vui->def_disp_win_right_offset;
        CHECK_ALLOWED_MAX("SubWidthC * (leftOffset + rightOffset)", sub_width_c_ * (left_offset + right_offset), sps_ptr->pic_width_in_luma_samples - 1);
        vui->def_disp_win_top_offset = Parser::ExpGolomb::ReadUe(nalu, offset);
        vui->def_disp_win_bottom_offset = Parser::ExpGolomb::ReadUe(nalu, offset);
        uint32_t top_offset = sps_ptr->conf_win_top_offset + vui->def_disp_win_top_offset;
        uint32_t bottom_offset = sps_ptr->conf_win_bottom_offset + vui->def_disp_win_bottom_offset;
        CHECK_ALLOWED_MAX("SubHeightC * (topOffset + bottomOffset)", sub_height_c_ * (top_offset + bottom_offset), sps_ptr->pic_height_in_luma_samples);
    }
    vui->vui_timing_info_present_flag = Parser::GetBit(nalu, offset);
    if (vui->vui_timing_info_present_flag) {
        vui->vui_num_units_in_tick = Parser::ReadBits(nalu, offset, 32);
        vui->vui_time_scale = Parser::ReadBits(nalu, offset, 32);
        vui->vui_poc_proportional_to_timing_flag = Parser::GetBit(nalu, offset);
        if (vui->vui_poc_proportional_to_timing_flag) {
            vui->vui_num_ticks_poc_diff_one_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
        }
        vui->vui_hrd_parameters_present_flag = Parser::GetBit(nalu, offset);
        if (vui->vui_hrd_parameters_present_flag) {
            ParserResult ret;
            if ((ret = ParseHrdParameters(&vui->hrd_parameters, 1, sps_ptr->sps_max_sub_layers_minus1, nalu, size, offset)) != PARSER_OK) {
                return ret;
            }
        }
    }
    vui->bitstream_restriction_flag = Parser::GetBit(nalu, offset);
    if (vui->bitstream_restriction_flag) {
        vui->tiles_fixed_structure_flag = Parser::GetBit(nalu, offset);
        vui->motion_vectors_over_pic_boundaries_flag = Parser::GetBit(nalu, offset);
        vui->restricted_ref_pic_lists_flag = Parser::GetBit(nalu, offset);
        vui->min_spatial_segmentation_idc = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("min_spatial_segmentation_idc", vui->min_spatial_segmentation_idc, 0, 4095);
        vui->max_bytes_per_pic_denom = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("max_bytes_per_pic_denom", vui->max_bytes_per_pic_denom, 0, 16);
        vui->max_bits_per_min_cu_denom = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("max_bits_per_min_cu_denom", vui->max_bits_per_min_cu_denom, 0, 16);
        vui->log2_max_mv_length_horizontal = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("log2_max_mv_length_horizontal", vui->log2_max_mv_length_horizontal, 0, 16);
        vui->log2_max_mv_length_vertical = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("log2_max_mv_length_vertical", vui->log2_max_mv_length_vertical, 0, 15);
    }
    return PARSER_OK;
}

ParserResult HevcVideoParser::ParseVps(uint8_t *nalu, size_t size) {
    size_t offset = 0; // current bit offset
    uint32_t vps_id = Parser::ReadBits(nalu, offset, 4);
    HevcVideoParamSet *p_vps = &vps_list_[vps_id];
    memset(p_vps, 0, sizeof(HevcVideoParamSet));

    p_vps->vps_video_parameter_set_id = vps_id;
    p_vps->vps_base_layer_internal_flag = Parser::GetBit(nalu, offset);
    p_vps->vps_base_layer_available_flag = Parser::GetBit(nalu, offset);
    p_vps->vps_max_layers_minus1 = Parser::ReadBits(nalu, offset, 6);
    p_vps->vps_max_sub_layers_minus1 = Parser::ReadBits(nalu, offset, 3);
    p_vps->vps_temporal_id_nesting_flag = Parser::GetBit(nalu, offset);
    p_vps->vps_reserved_0xffff_16bits = Parser::ReadBits(nalu, offset, 16);
    if (p_vps->vps_reserved_0xffff_16bits != 0xFFFF) {
        logger_.ErrorLog(MakeMsg("vps_reserved_0xffff_16bits is not equal to 0xFFFF."));
        return PARSER_INVALID_ARG;
    }
    ParsePtl(&p_vps->profile_tier_level, true, p_vps->vps_max_sub_layers_minus1, nalu, size, offset);
    p_vps->vps_sub_layer_ordering_info_present_flag = Parser::GetBit(nalu, offset);
    for (int i = 0; i <= p_vps->vps_max_sub_layers_minus1; i++) {
        if (p_vps->vps_sub_layer_ordering_info_present_flag || (i == 0)) {
            p_vps->vps_max_dec_pic_buffering_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("vps_max_dec_pic_buffering_minus1", p_vps->vps_max_dec_pic_buffering_minus1[i], 0, HEVC_MAX_DPB_FRAMES - 1);
            p_vps->vps_max_num_reorder_pics[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("vps_max_num_reorder_pics", p_vps->vps_max_num_reorder_pics[i], 0, p_vps->vps_max_dec_pic_buffering_minus1[i]);
            p_vps->vps_max_latency_increase_plus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
        } else {
            p_vps->vps_max_dec_pic_buffering_minus1[i] = p_vps->vps_max_dec_pic_buffering_minus1[0];
            p_vps->vps_max_num_reorder_pics[i] = p_vps->vps_max_num_reorder_pics[0];
            p_vps->vps_max_latency_increase_plus1[i] = p_vps->vps_max_latency_increase_plus1[0];
        }
    }
    p_vps->vps_max_layer_id = Parser::ReadBits(nalu, offset, 6);
    p_vps->vps_num_layer_sets_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("vps_num_layer_sets_minus1", p_vps->vps_num_layer_sets_minus1, 0, 1023);
    for (int i = 1; i <= p_vps->vps_num_layer_sets_minus1; i++) {
        for (int j = 0; j <= p_vps->vps_max_layer_id; j++) {
            p_vps->layer_id_included_flag[i][j] = Parser::GetBit(nalu, offset);
        }
    }
    p_vps->vps_timing_info_present_flag = Parser::GetBit(nalu, offset);
    if(p_vps->vps_timing_info_present_flag) {
        p_vps->vps_num_units_in_tick = Parser::ReadBits(nalu, offset, 32);
        p_vps->vps_time_scale = Parser::ReadBits(nalu, offset, 32);
        p_vps->vps_poc_proportional_to_timing_flag = Parser::GetBit(nalu, offset);
        if(p_vps->vps_poc_proportional_to_timing_flag) {
            p_vps->vps_num_ticks_poc_diff_one_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
        }
        p_vps->vps_num_hrd_parameters = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("vps_num_hrd_parameters", p_vps->vps_num_hrd_parameters, 0, p_vps->vps_num_layer_sets_minus1 + 1);
        for (int i = 0; i<p_vps->vps_num_hrd_parameters; i++) {
            p_vps->hrd_layer_set_idx[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("hrd_layer_set_idx", p_vps->hrd_layer_set_idx[i], (p_vps->vps_base_layer_internal_flag ? 0 : 1), p_vps->vps_num_layer_sets_minus1);
            if (i > 0) {
                p_vps->cprms_present_flag[i] = Parser::GetBit(nalu, offset);
            }
            //parse HRD parameters
            ParserResult ret;
            if ((ret = ParseHrdParameters(&p_vps->hrd_parameters[i], p_vps->cprms_present_flag[i], p_vps->vps_max_sub_layers_minus1, nalu, size, offset)) != PARSER_OK) {
                return ret;
            }
        }
    }
    p_vps->vps_extension_flag = Parser::GetBit(nalu, offset);
    p_vps->is_received = 1;

#if DBGINFO
    PrintVps(p_vps);
#endif // DBGINFO
    return PARSER_OK;
}

ParserResult HevcVideoParser::ParseSps(uint8_t *nalu, size_t size) {
    ParserResult ret = PARSER_OK;
    HevcSeqParamSet *sps_ptr = nullptr;
    size_t offset = 0;

    uint32_t vps_id = Parser::ReadBits(nalu, offset, 4);
    uint32_t max_sub_layer_minus1 = Parser::ReadBits(nalu, offset, 3);
    uint32_t sps_temporal_id_nesting_flag = Parser::GetBit(nalu, offset);
    HevcProfileTierLevel ptl;
    memset (&ptl, 0, sizeof(ptl));
    ParsePtl(&ptl, true, max_sub_layer_minus1, nalu, size, offset);

    uint32_t sps_id = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("sps_seq_parameter_set_id", sps_id, 0, 15);
    sps_ptr = &sps_list_[sps_id];

    memset(sps_ptr, 0, sizeof(HevcSeqParamSet));
    sps_ptr->sps_video_parameter_set_id = vps_id;
    sps_ptr->sps_max_sub_layers_minus1 = max_sub_layer_minus1;
    sps_ptr->sps_temporal_id_nesting_flag = sps_temporal_id_nesting_flag;
    memcpy (&sps_ptr->profile_tier_level, &ptl, sizeof(ptl));
    sps_ptr->sps_seq_parameter_set_id = sps_id;
    sps_ptr->chroma_format_idc = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("chroma_format_idc", sps_ptr->chroma_format_idc, 0, 3);
    if (sps_ptr->chroma_format_idc == 3) {
        sps_ptr->separate_colour_plane_flag = Parser::GetBit(nalu, offset);
    }
    switch (sps_ptr->chroma_format_idc) {
        case 0: {
            sub_width_c_ = 1;
            sub_height_c_ = 1;
            break;
        }
        case 1: {
            sub_width_c_ = 2;
            sub_height_c_ = 2;
            break;
        }
        case 2: {
            sub_width_c_ = 2;
            sub_height_c_ = 1;
            break;
        }
        case 3: {
            sub_width_c_ = 1;
            sub_height_c_ = 1;
            break;
        }
    }
    sps_ptr->pic_width_in_luma_samples = Parser::ExpGolomb::ReadUe(nalu, offset);
    sps_ptr->pic_height_in_luma_samples = Parser::ExpGolomb::ReadUe(nalu, offset);
    sps_ptr->conformance_window_flag = Parser::GetBit(nalu, offset);
    if (sps_ptr->conformance_window_flag) {
        sps_ptr->conf_win_left_offset = Parser::ExpGolomb::ReadUe(nalu, offset);
        sps_ptr->conf_win_right_offset = Parser::ExpGolomb::ReadUe(nalu, offset);
        sps_ptr->conf_win_top_offset = Parser::ExpGolomb::ReadUe(nalu, offset);
        sps_ptr->conf_win_bottom_offset = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_MAX("SubWidthC * (conf_win_left_offset + conf_win_right_offset)", sub_width_c_ * (sps_ptr->conf_win_left_offset + sps_ptr->conf_win_right_offset), sps_ptr->pic_width_in_luma_samples - 1);
        CHECK_ALLOWED_MAX("SubHeightC * (conf_win_top_offset + conf_win_bottom_offset)", sub_height_c_ * (sps_ptr->conf_win_top_offset + sps_ptr->conf_win_bottom_offset), sps_ptr->pic_width_in_luma_samples - 1);
    }
    sps_ptr->bit_depth_luma_minus8 = Parser::ExpGolomb::ReadUe(nalu, offset);
    if ( sps_ptr->bit_depth_luma_minus8 != 0 && sps_ptr->bit_depth_luma_minus8 != 2) {
        logger_.ErrorLog(MakeMsg("bit_depth_luma_minus8 = " + TOSTR(sps_ptr->bit_depth_luma_minus8) + " is not supported"));
        return PARSER_OUT_OF_RANGE;
    }
    sps_ptr->bit_depth_chroma_minus8 = Parser::ExpGolomb::ReadUe(nalu, offset);
    if ( sps_ptr->bit_depth_chroma_minus8 != 0 && sps_ptr->bit_depth_chroma_minus8 != 2) {
        logger_.ErrorLog(MakeMsg("bit_depth_chroma_minus8 = " + TOSTR(sps_ptr->bit_depth_chroma_minus8) + " is not supported"));
        return PARSER_OUT_OF_RANGE;
    }
    sps_ptr->log2_max_pic_order_cnt_lsb_minus4 = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("log2_max_pic_order_cnt_lsb_minus4", sps_ptr->log2_max_pic_order_cnt_lsb_minus4, 0, 12);
    sps_ptr->sps_sub_layer_ordering_info_present_flag = Parser::GetBit(nalu, offset);
    for (int i = 0; i <= sps_ptr->sps_max_sub_layers_minus1; i++) {
        if (sps_ptr->sps_sub_layer_ordering_info_present_flag || (i == 0)) {
            sps_ptr->sps_max_dec_pic_buffering_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("sps_max_dec_pic_buffering_minus1", sps_ptr->sps_max_dec_pic_buffering_minus1[i], 0, HEVC_MAX_DPB_FRAMES - 1);
            sps_ptr->sps_max_num_reorder_pics[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("sps_max_num_reorder_pics", sps_ptr->sps_max_num_reorder_pics[i], 0, sps_ptr->sps_max_dec_pic_buffering_minus1[i]);
            sps_ptr->sps_max_latency_increase_plus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
        } else {
            sps_ptr->sps_max_dec_pic_buffering_minus1[i] = sps_ptr->sps_max_dec_pic_buffering_minus1[0];
            sps_ptr->sps_max_num_reorder_pics[i] = sps_ptr->sps_max_num_reorder_pics[0];
            sps_ptr->sps_max_latency_increase_plus1[i] = sps_ptr->sps_max_latency_increase_plus1[0];
        }
    }
    sps_ptr->log2_min_luma_coding_block_size_minus3 = Parser::ExpGolomb::ReadUe(nalu, offset);
    int min_cb_log2_size_y = sps_ptr->log2_min_luma_coding_block_size_minus3 + 3;  // MinCbLog2SizeY
    sps_ptr->log2_diff_max_min_luma_coding_block_size = Parser::ExpGolomb::ReadUe(nalu, offset);
    int max_cu_depth_delta = sps_ptr->log2_diff_max_min_luma_coding_block_size;
    sps_ptr->max_cu_width = (1 << (min_cb_log2_size_y + max_cu_depth_delta));
    sps_ptr->max_cu_height = (1 << (min_cb_log2_size_y + max_cu_depth_delta));
    sps_ptr->log2_min_luma_transform_block_size_minus2 = Parser::ExpGolomb::ReadUe(nalu, offset);
    uint32_t min_tb_log2_size_y = sps_ptr->log2_min_luma_transform_block_size_minus2 + 2; // MinTbLog2SizeY
    CHECK_ALLOWED_MAX("min_tb_log2_size_y", min_tb_log2_size_y, min_cb_log2_size_y - 1);
    int add_cu_depth = std::max(0, min_cb_log2_size_y - (int)min_tb_log2_size_y);
    sps_ptr->max_cu_depth = (max_cu_depth_delta + add_cu_depth);
    sps_ptr->log2_diff_max_min_luma_transform_block_size = Parser::ExpGolomb::ReadUe(nalu, offset);
    max_tb_log2_size_y_ = sps_ptr->log2_min_luma_transform_block_size_minus2 + 2 + sps_ptr->log2_diff_max_min_luma_transform_block_size; // MaxTbLog2SizeY
    ctb_log2_size_y_ = min_cb_log2_size_y + sps_ptr->log2_diff_max_min_luma_coding_block_size;  // CtbLog2SizeY
    CHECK_ALLOWED_MAX("max_tb_log2_size_y_", max_tb_log2_size_y_, std::min(ctb_log2_size_y_, 5));
    sps_ptr->max_transform_hierarchy_depth_inter = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("max_transform_hierarchy_depth_inter", sps_ptr->max_transform_hierarchy_depth_inter, 0, ctb_log2_size_y_ - min_tb_log2_size_y);
    sps_ptr->max_transform_hierarchy_depth_intra = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("max_transform_hierarchy_depth_intra", sps_ptr->max_transform_hierarchy_depth_intra, 0, ctb_log2_size_y_ - min_tb_log2_size_y);

    // Infer dimensional variables
    int ctb_size_y = 1 << ctb_log2_size_y_;  // CtbSizeY
    pic_width_in_ctbs_y_ = (sps_ptr->pic_width_in_luma_samples + ctb_size_y - 1) / ctb_size_y;  // PicWidthInCtbsY
    pic_height_in_ctbs_y_ = (sps_ptr->pic_height_in_luma_samples + ctb_size_y - 1) / ctb_size_y;  // PicHeightInCtbsY
    pic_size_in_ctbs_y_ = pic_width_in_ctbs_y_ * pic_height_in_ctbs_y_;  // PicSizeInCtbsY

    sps_ptr->scaling_list_enabled_flag = Parser::GetBit(nalu, offset);
    if (sps_ptr->scaling_list_enabled_flag) {
        // Set up default values first
        SetDefaultScalingList(&sps_ptr->scaling_list_data);
        sps_ptr->sps_scaling_list_data_present_flag = Parser::GetBit(nalu, offset);
        if (sps_ptr->sps_scaling_list_data_present_flag) {
            if ((ret = ParseScalingList(&sps_ptr->scaling_list_data, nalu, size, offset, sps_ptr)) != PARSER_OK) {
                return ret;
            }
        }
    }
    sps_ptr->amp_enabled_flag = Parser::GetBit(nalu, offset);
    sps_ptr->sample_adaptive_offset_enabled_flag = Parser::GetBit(nalu, offset);
    sps_ptr->pcm_enabled_flag = Parser::GetBit(nalu, offset);
    if (sps_ptr->pcm_enabled_flag) {
        sps_ptr->pcm_sample_bit_depth_luma_minus1 = Parser::ReadBits(nalu, offset, 4);
        sps_ptr->pcm_sample_bit_depth_chroma_minus1 = Parser::ReadBits(nalu, offset, 4);
        sps_ptr->log2_min_pcm_luma_coding_block_size_minus3 = Parser::ExpGolomb::ReadUe(nalu, offset);
        //CHECK_ALLOWED_RANGE("log2_min_pcm_luma_coding_block_size", sps_ptr->log2_min_pcm_luma_coding_block_size_minus3 + 3, std::min(min_cb_log2_size_y, 5), std::min(ctb_log2_size_y_, 5));
        if ((sps_ptr->log2_min_pcm_luma_coding_block_size_minus3 + 3) < std::min(min_cb_log2_size_y, 5) || (sps_ptr->log2_min_pcm_luma_coding_block_size_minus3 + 3) > std::min(ctb_log2_size_y_, 5)) {
            logger_.ErrorLog(MakeMsg("log2_min_pcm_luma_coding_block_size = " + TOSTR(sps_ptr->log2_min_pcm_luma_coding_block_size_minus3 + 3) + " not in allowd range: " + TOSTR(std::min(min_cb_log2_size_y, 5)) + ", " + TOSTR(std::min(ctb_log2_size_y_, 5))));
        }
        sps_ptr->log2_diff_max_min_pcm_luma_coding_block_size = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_MAX("log2_max_ipcm_cb_size_y", sps_ptr->log2_diff_max_min_pcm_luma_coding_block_size + sps_ptr->log2_min_pcm_luma_coding_block_size_minus3 + 3, std::min(ctb_log2_size_y_, 5));
        sps_ptr->pcm_loop_filter_disabled_flag = Parser::GetBit(nalu, offset);
    }
    sps_ptr->num_short_term_ref_pic_sets = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("num_short_term_ref_pic_sets", sps_ptr->num_short_term_ref_pic_sets, 0, 64);
    for (int i = 0; i < sps_ptr->num_short_term_ref_pic_sets; i++) {
        //short_term_ref_pic_set( i )
        if ((ret = ParseShortTermRefPicSet(sps_ptr, &sps_ptr->st_rps[i], i, sps_ptr->num_short_term_ref_pic_sets, sps_ptr->st_rps, nalu, size, offset)) != PARSER_OK) {
            return ret;
        }
    }
    sps_ptr->long_term_ref_pics_present_flag = Parser::GetBit(nalu, offset);
    if (sps_ptr->long_term_ref_pics_present_flag) {
        sps_ptr->num_long_term_ref_pics_sps = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("num_long_term_ref_pics_sps", sps_ptr->num_long_term_ref_pics_sps, 0, 32);
        sps_ptr->lt_rps.num_of_pics = sps_ptr->num_long_term_ref_pics_sps;
        for (int i=0; i<sps_ptr->num_long_term_ref_pics_sps; i++) {
            //The number of bits used to represent lt_ref_pic_poc_lsb_sps[ i ] is equal to log2_max_pic_order_cnt_lsb_minus4 + 4.
            sps_ptr->lt_ref_pic_poc_lsb_sps[i] = Parser::ReadBits(nalu, offset, (sps_ptr->log2_max_pic_order_cnt_lsb_minus4 + 4));
            sps_ptr->used_by_curr_pic_lt_sps_flag[i] = Parser::GetBit(nalu, offset);
            sps_ptr->lt_rps.pocs[i]=sps_ptr->lt_ref_pic_poc_lsb_sps[i];
            sps_ptr->lt_rps.used_by_curr_pic[i] = sps_ptr->used_by_curr_pic_lt_sps_flag[i];            
        }
    }
    sps_ptr->sps_temporal_mvp_enabled_flag = Parser::GetBit(nalu, offset);
    sps_ptr->strong_intra_smoothing_enabled_flag = Parser::GetBit(nalu, offset);
    sps_ptr->vui_parameters_present_flag = Parser::GetBit(nalu, offset);
    if (sps_ptr->vui_parameters_present_flag) {
        //vui_parameters()
        if((ret = ParseVui(sps_ptr, nalu, size, offset)) != PARSER_OK) {
            return ret;
        }
    }
    sps_ptr->sps_extension_flag = Parser::GetBit(nalu, offset);
    sps_ptr->is_received = 1;

#if DBGINFO
    PrintSps(sps_ptr);
#endif // DBGINFO
    return PARSER_OK;
}

ParserResult HevcVideoParser::ParsePps(uint8_t *nalu, size_t size) {
    int i;
    size_t offset = 0;
    uint32_t pps_id = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("pps_pic_parameter_set_id", pps_id, 0, 63);
    HevcPicParamSet *pps_ptr = &pps_list_[pps_id];
    memset(pps_ptr, 0, sizeof(HevcPicParamSet));

    pps_ptr->pps_pic_parameter_set_id = pps_id;
    pps_ptr->pps_seq_parameter_set_id = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("pps_seq_parameter_set_id", pps_ptr->pps_seq_parameter_set_id, 0, 15);
    pps_ptr->dependent_slice_segments_enabled_flag = Parser::GetBit(nalu, offset);
    pps_ptr->output_flag_present_flag = Parser::GetBit(nalu, offset);
    pps_ptr->num_extra_slice_header_bits = Parser::ReadBits(nalu, offset, 3);
    pps_ptr->sign_data_hiding_enabled_flag = Parser::GetBit(nalu, offset);
    pps_ptr->cabac_init_present_flag = Parser::GetBit(nalu, offset);
    pps_ptr->num_ref_idx_l0_default_active_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("num_ref_idx_l0_default_active_minus1", pps_ptr->num_ref_idx_l0_default_active_minus1, 0, 14);
    pps_ptr->num_ref_idx_l1_default_active_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("num_ref_idx_l1_default_active_minus1", pps_ptr->num_ref_idx_l1_default_active_minus1, 0, 14);
    pps_ptr->init_qp_minus26 = Parser::ExpGolomb::ReadSe(nalu, offset);
    CHECK_ALLOWED_RANGE("init_qp_minus26", pps_ptr->init_qp_minus26, static_cast<int>(-(26 + 6 * sps_list_[pps_ptr->pps_seq_parameter_set_id].bit_depth_luma_minus8)), 25);
    pps_ptr->constrained_intra_pred_flag = Parser::GetBit(nalu, offset);
    pps_ptr->transform_skip_enabled_flag = Parser::GetBit(nalu, offset);
    pps_ptr->cu_qp_delta_enabled_flag = Parser::GetBit(nalu, offset);
    if (pps_ptr->cu_qp_delta_enabled_flag) {
        pps_ptr->diff_cu_qp_delta_depth = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("diff_cu_qp_delta_depth", pps_ptr->diff_cu_qp_delta_depth, 0, sps_list_[pps_ptr->pps_seq_parameter_set_id].log2_diff_max_min_luma_coding_block_size);
    }
    pps_ptr->pps_cb_qp_offset = Parser::ExpGolomb::ReadSe(nalu, offset);
    CHECK_ALLOWED_RANGE("pps_cb_qp_offset", pps_ptr->pps_cb_qp_offset, -12, 12);
    pps_ptr->pps_cr_qp_offset = Parser::ExpGolomb::ReadSe(nalu, offset);
    CHECK_ALLOWED_RANGE("pps_cr_qp_offset", pps_ptr->pps_cr_qp_offset, -12, 12);
    pps_ptr->pps_slice_chroma_qp_offsets_present_flag = Parser::GetBit(nalu, offset);
    pps_ptr->weighted_pred_flag = Parser::GetBit(nalu, offset);
    pps_ptr->weighted_bipred_flag = Parser::GetBit(nalu, offset);
    pps_ptr->transquant_bypass_enabled_flag = Parser::GetBit(nalu, offset);
    pps_ptr->tiles_enabled_flag = Parser::GetBit(nalu, offset);
    pps_ptr->entropy_coding_sync_enabled_flag = Parser::GetBit(nalu, offset);
    if (pps_ptr->tiles_enabled_flag) {
        pps_ptr->num_tile_columns_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("num_tile_columns_minus1", pps_ptr->num_tile_columns_minus1, 0, pic_width_in_ctbs_y_ - 1);
        pps_ptr->num_tile_rows_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("num_tile_rows_minus1", pps_ptr->num_tile_rows_minus1, 0, pic_height_in_ctbs_y_ - 1);
        pps_ptr->uniform_spacing_flag = Parser::GetBit(nalu, offset);
        if (!pps_ptr->uniform_spacing_flag) {
            int temp_size = pic_width_in_ctbs_y_; // PicWidthInCtbsY
            for (i = 0; i < pps_ptr->num_tile_columns_minus1; i++) {
                pps_ptr->column_width_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
                CHECK_ALLOWED_RANGE("column_width_minus1", pps_ptr->column_width_minus1[i], 0, pic_width_in_ctbs_y_ - 1);
                temp_size -= pps_ptr->column_width_minus1[i] + 1;
            }
            pps_ptr->column_width_minus1[i] = temp_size - 1; // last column at num_tile_columns_minus1

            temp_size = pic_height_in_ctbs_y_; // PicHeightInCtbsY
            for (i = 0; i < pps_ptr->num_tile_rows_minus1; i++) {
                pps_ptr->row_height_minus1[i] = Parser::ExpGolomb::ReadUe(nalu, offset);
                CHECK_ALLOWED_RANGE("row_height_minus1", pps_ptr->row_height_minus1[i], 0, pic_height_in_ctbs_y_ - 1);
                temp_size -= pps_ptr->row_height_minus1[i] + 1;
            }
            pps_ptr->row_height_minus1[i] = temp_size - 1;  // last row at num_tile_rows_minus1
        } else {
            for (i = 0; i <= pps_ptr->num_tile_columns_minus1; i++) {
                pps_ptr->column_width_minus1[i] = ((i + 1) * pic_width_in_ctbs_y_) / (pps_ptr->num_tile_columns_minus1 + 1) - (i * pic_width_in_ctbs_y_) / (pps_ptr->num_tile_columns_minus1 + 1) - 1;
            }
            for (i = 0; i <= pps_ptr->num_tile_rows_minus1; i++) {
                pps_ptr->row_height_minus1[i] = ((i + 1) * pic_height_in_ctbs_y_) / (pps_ptr->num_tile_rows_minus1 + 1) - (i * pic_height_in_ctbs_y_) / (pps_ptr->num_tile_rows_minus1 + 1) - 1;
            }
        }
        pps_ptr->loop_filter_across_tiles_enabled_flag = Parser::GetBit(nalu, offset);
    } else {
        pps_ptr->loop_filter_across_tiles_enabled_flag = 1;
        pps_ptr->uniform_spacing_flag = 1;
        pps_ptr->num_tile_columns_minus1 = 0;
        pps_ptr->num_tile_rows_minus1 = 0;
    }
    pps_ptr->pps_loop_filter_across_slices_enabled_flag = Parser::GetBit(nalu, offset);
    pps_ptr->deblocking_filter_control_present_flag = Parser::GetBit(nalu, offset);
    if (pps_ptr->deblocking_filter_control_present_flag) {
        pps_ptr->deblocking_filter_override_enabled_flag = Parser::GetBit(nalu, offset);
        pps_ptr->pps_deblocking_filter_disabled_flag = Parser::GetBit(nalu, offset);
        if (!pps_ptr->pps_deblocking_filter_disabled_flag) {
            pps_ptr->pps_beta_offset_div2 = Parser::ExpGolomb::ReadSe(nalu, offset);
            CHECK_ALLOWED_RANGE("pps_beta_offset_div2", pps_ptr->pps_beta_offset_div2, -6, 6);
            pps_ptr->pps_tc_offset_div2 = Parser::ExpGolomb::ReadSe(nalu, offset);
            CHECK_ALLOWED_RANGE("pps_tc_offset_div2", pps_ptr->pps_tc_offset_div2, -6, 6);
        }
    }
    pps_ptr->pps_scaling_list_data_present_flag = Parser::GetBit(nalu, offset);
    if (pps_ptr->pps_scaling_list_data_present_flag) {
        // Set up default values first
        SetDefaultScalingList(&pps_ptr->scaling_list_data);
        ParserResult ret;
        if ((ret = ParseScalingList(&pps_ptr->scaling_list_data, nalu, size, offset, &sps_list_[pps_ptr->pps_seq_parameter_set_id])) != PARSER_OK) {
            return ret;
        }
    } else {
        pps_ptr->scaling_list_data = sps_list_[pps_ptr->pps_seq_parameter_set_id].scaling_list_data;
    }
    pps_ptr->lists_modification_present_flag = Parser::GetBit(nalu, offset);
    pps_ptr->log2_parallel_merge_level_minus2 = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_RANGE("log2_parallel_merge_level_minus2", pps_ptr->log2_parallel_merge_level_minus2, 0, ctb_log2_size_y_ - 2);
    pps_ptr->slice_segment_header_extension_present_flag = Parser::GetBit(nalu, offset);
    pps_ptr->pps_extension_present_flag = Parser::GetBit(nalu, offset);
    if (pps_ptr->pps_extension_present_flag) {
        pps_ptr->pps_range_extension_flag = Parser::GetBit(nalu, offset);
        pps_ptr->pps_multilayer_extension_flag = Parser::GetBit(nalu, offset);
        pps_ptr->pps_extension_6bits = Parser::ReadBits(nalu, offset, 6);
    }

    // pps_range_extension()
    if (pps_ptr->pps_range_extension_flag) {
        if (pps_ptr->transform_skip_enabled_flag) {
            pps_ptr->log2_max_transform_skip_block_size_minus2 = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_MAX("log2_max_transform_skip_block_size_minus2", pps_ptr->log2_max_transform_skip_block_size_minus2, max_tb_log2_size_y_ - 2);
        }
        pps_ptr->cross_component_prediction_enabled_flag = Parser::GetBit(nalu, offset);
        pps_ptr->chroma_qp_offset_list_enabled_flag = Parser::GetBit(nalu, offset);
        if (pps_ptr->chroma_qp_offset_list_enabled_flag) {
            pps_ptr->diff_cu_chroma_qp_offset_depth = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("diff_cu_chroma_qp_offset_depth", pps_ptr->diff_cu_chroma_qp_offset_depth, 0, sps_list_[pps_ptr->pps_seq_parameter_set_id].log2_diff_max_min_luma_coding_block_size);
            pps_ptr->chroma_qp_offset_list_len_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
            CHECK_ALLOWED_RANGE("chroma_qp_offset_list_len_minus1", pps_ptr->chroma_qp_offset_list_len_minus1, 0, 5);
            for (int i = 0; i <= pps_ptr->chroma_qp_offset_list_len_minus1; i++) {
                pps_ptr->cb_qp_offset_list[i] = Parser::ExpGolomb::ReadSe(nalu, offset);
                CHECK_ALLOWED_RANGE("cb_qp_offset_list", pps_ptr->cb_qp_offset_list[i], -12, 12);
                pps_ptr->cr_qp_offset_list[i] = Parser::ExpGolomb::ReadSe(nalu, offset);
                CHECK_ALLOWED_RANGE("cr_qp_offset_list", pps_ptr->cr_qp_offset_list[i], -12, 12);
            }
        }
        pps_ptr->log2_sao_offset_scale_luma = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("log2_sao_offset_scale_luma", pps_ptr->log2_sao_offset_scale_luma, 0, std::max(0, static_cast<int>(sps_list_[pps_ptr->pps_seq_parameter_set_id].bit_depth_luma_minus8 - 2)));
        CHECK_ALLOWED_RANGE("log2_sao_offset_scale_luma", pps_ptr->log2_sao_offset_scale_luma, 0, std::max(0, static_cast<int>(sps_list_[pps_ptr->pps_seq_parameter_set_id].bit_depth_luma_minus8 - 2)));
        pps_ptr->log2_sao_offset_scale_chroma = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_RANGE("log2_sao_offset_scale_chroma", pps_ptr->log2_sao_offset_scale_chroma, 0, std::max(0, static_cast<int>(sps_list_[pps_ptr->pps_seq_parameter_set_id].bit_depth_chroma_minus8 - 2)));
    }

    pps_ptr->is_received = 1;
#if DBGINFO
    PrintPps(pps_ptr);
#endif // DBGINFO
    return PARSER_OK;
}

ParserResult HevcVideoParser::ParseSliceHeader(uint8_t *nalu, size_t size, HevcSliceSegHeader *p_slice_header) {
    HevcPicParamSet *pps_ptr = nullptr;
    HevcSeqParamSet *sps_ptr = nullptr;
    size_t offset = 0;
    HevcSliceSegHeader temp_sh;
    memset(p_slice_header, 0, sizeof(HevcSliceSegHeader));
    memset(&temp_sh, 0, sizeof(temp_sh));

    temp_sh.first_slice_segment_in_pic_flag = p_slice_header->first_slice_segment_in_pic_flag = Parser::GetBit(nalu, offset);
    if (IsIrapPic(&slice_nal_unit_header_)) {
        temp_sh.no_output_of_prior_pics_flag = p_slice_header->no_output_of_prior_pics_flag = Parser::GetBit(nalu, offset);
    }

    // Set active VPS, SPS and PPS for the current slice
    int32_t active_pps_id = Parser::ExpGolomb::ReadUe(nalu, offset);
    CHECK_ALLOWED_MAX("active_pps_id", active_pps_id, (MAX_PPS_COUNT - 1));
    if (pps_list_[active_pps_id].is_received == 0) {
        logger_.ErrorLog(MakeMsg("Empty PPS is referred."));
        return PARSER_WRONG_STATE;
    }
    active_pps_id_ = active_pps_id;
    pps_ptr = &pps_list_[active_pps_id_];
    temp_sh.slice_pic_parameter_set_id = p_slice_header->slice_pic_parameter_set_id = active_pps_id_;

    int32_t active_sps_id = pps_ptr->pps_seq_parameter_set_id;
    if (sps_list_[active_sps_id].is_received == 0) {
        logger_.ErrorLog(MakeMsg("Empty SPS is referred."));
        return PARSER_WRONG_STATE;
    }
    if (active_sps_id_ != active_sps_id) {
        active_sps_id_ = active_sps_id;
        sps_ptr = &sps_list_[active_sps_id_];
        new_seq_activated_ = true;
    }
    sps_ptr = &sps_list_[active_sps_id_];

    int active_vps_id = sps_ptr->sps_video_parameter_set_id;
    if (vps_list_[active_vps_id].is_received == 0) {
        logger_.ErrorLog(MakeMsg("Empty VPS is referred."));
        return PARSER_WRONG_STATE;
    }
    active_vps_id_ = active_vps_id;

    // Check video dimension change
    if (pic_width_ != sps_ptr->pic_width_in_luma_samples || pic_height_ != sps_ptr->pic_height_in_luma_samples) {
        pic_width_ = sps_ptr->pic_width_in_luma_samples;
        pic_height_ = sps_ptr->pic_height_in_luma_samples;
        // Take care of the case where a new SPS replaces the old SPS with the same id but with different dimensions
        new_seq_activated_ = true;
    }
    // Check DPB buffer size change
    uint32_t max_dec_pic_buffering = sps_ptr->sps_max_dec_pic_buffering_minus1[sps_ptr->sps_max_sub_layers_minus1] + 1;
    if (dpb_buffer_.dpb_size != max_dec_pic_buffering) {
        dpb_buffer_.dpb_size = max_dec_pic_buffering;
        dpb_buffer_.dpb_size = dpb_buffer_.dpb_size > HEVC_MAX_DPB_FRAMES ? HEVC_MAX_DPB_FRAMES : dpb_buffer_.dpb_size;
        CheckAndAdjustDecBufPoolSize(dpb_buffer_.dpb_size);
        new_seq_activated_ = true;
    }
    // Check bit depth change
    if (bit_depth_luma_minus8_ != sps_ptr->bit_depth_luma_minus8 || bit_depth_chroma_minus8_ != sps_ptr->bit_depth_chroma_minus8) {
        bit_depth_luma_minus8_ = sps_ptr->bit_depth_luma_minus8;
        bit_depth_chroma_minus8_ = sps_ptr->bit_depth_chroma_minus8;
        new_seq_activated_ = true;
    }

    // Set frame rate if available
    if (new_seq_activated_) {
        if (vps_list_[active_vps_id_].vps_timing_info_present_flag) {
            frame_rate_.numerator = vps_list_[active_vps_id_].vps_time_scale;
            frame_rate_.denominator = vps_list_[active_vps_id_].vps_num_units_in_tick;
        } else if (sps_ptr->vui_parameters.vui_timing_info_present_flag) {
            frame_rate_.numerator = sps_ptr->vui_parameters.vui_time_scale;
            frame_rate_.denominator = sps_ptr->vui_parameters.vui_num_units_in_tick;
        } else {
            frame_rate_.numerator = 0;
            frame_rate_.denominator = 0;
        }
    }

    if (!p_slice_header->first_slice_segment_in_pic_flag) {
        if (pps_ptr->dependent_slice_segments_enabled_flag) {
            temp_sh.dependent_slice_segment_flag = p_slice_header->dependent_slice_segment_flag = Parser::GetBit(nalu, offset);
        }
        int bits_slice_segment_address = (int)ceilf(log2f((float)pic_size_in_ctbs_y_));
        temp_sh.slice_segment_address = p_slice_header->slice_segment_address = Parser::ReadBits(nalu, offset, bits_slice_segment_address);
        //todo:: check for max slice_segment_address and error if exceeds max
    }

    if (!p_slice_header->dependent_slice_segment_flag) {
        memset(&slice_header_copy_, 0, sizeof(slice_header_copy_));
        for (int i = 0; i < pps_ptr->num_extra_slice_header_bits; i++) {
            p_slice_header->slice_reserved_flag[i] = Parser::GetBit(nalu, offset);
        }
        p_slice_header->slice_type = Parser::ExpGolomb::ReadUe(nalu, offset);
        CHECK_ALLOWED_MAX("slice_type", p_slice_header->slice_type, 2);

        if (pps_ptr->output_flag_present_flag) {
            p_slice_header->pic_output_flag = Parser::GetBit(nalu, offset);
        } else {
            p_slice_header->pic_output_flag = 1;  // default value
        }
        if (sps_ptr->separate_colour_plane_flag) {
            p_slice_header->colour_plane_id = Parser::ReadBits(nalu, offset, 2);
        }

        if (slice_nal_unit_header_.nal_unit_type != NAL_UNIT_CODED_SLICE_IDR_W_RADL && slice_nal_unit_header_.nal_unit_type != NAL_UNIT_CODED_SLICE_IDR_N_LP) {
            //length of slice_pic_order_cnt_lsb is log2_max_pic_order_cnt_lsb_minus4 + 4 bits.
            p_slice_header->slice_pic_order_cnt_lsb = Parser::ReadBits(nalu, offset, (sps_ptr->log2_max_pic_order_cnt_lsb_minus4 + 4));

            p_slice_header->short_term_ref_pic_set_sps_flag = Parser::GetBit(nalu, offset);
            int32_t pos = offset;
            if (!p_slice_header->short_term_ref_pic_set_sps_flag) {
                if ( ParseShortTermRefPicSet(sps_ptr, &p_slice_header->st_rps, sps_ptr->num_short_term_ref_pic_sets, sps_ptr->num_short_term_ref_pic_sets, sps_ptr->st_rps, nalu, size, offset) != PARSER_OK) {
                    return PARSER_WRONG_STATE;
                }
            } else {
                if (sps_ptr->num_short_term_ref_pic_sets > 1) {
                    int num_bits = 0;
                    while ((1 << num_bits) < sps_ptr->num_short_term_ref_pic_sets) {
                        num_bits++;
                    }
                    if (num_bits > 0) {
                        p_slice_header->short_term_ref_pic_set_idx = Parser::ReadBits(nalu, offset, num_bits);
                    }
                } else {
                    p_slice_header->short_term_ref_pic_set_idx = 0;
                }

                // Copy the SPS RPS to slice RPS
                p_slice_header->st_rps = sps_ptr->st_rps[p_slice_header->short_term_ref_pic_set_idx];
            }
            p_slice_header->short_term_ref_pic_set_size = offset - pos;

            if (sps_ptr->long_term_ref_pics_present_flag) {
                if (sps_ptr->num_long_term_ref_pics_sps > 0) {
                    p_slice_header->num_long_term_sps = Parser::ExpGolomb::ReadUe(nalu, offset);
                    CHECK_ALLOWED_RANGE("num_long_term_sps", p_slice_header->num_long_term_sps, 0, sps_ptr->num_long_term_ref_pics_sps);
                }
                p_slice_header->num_long_term_pics = Parser::ExpGolomb::ReadUe(nalu, offset);
                CHECK_ALLOWED_MAX("num_long_term_pics", p_slice_header->num_long_term_pics, HEVC_MAX_DPB_FRAMES - 1);

                int bits_for_ltrp_in_sps = 0;
                while (sps_ptr->num_long_term_ref_pics_sps > (1 << bits_for_ltrp_in_sps)) {
                    bits_for_ltrp_in_sps++;
                }
                p_slice_header->lt_rps.num_of_pics = p_slice_header->num_long_term_sps + p_slice_header->num_long_term_pics;
                for (int i = 0; i < (p_slice_header->num_long_term_sps + p_slice_header->num_long_term_pics); i++) {
                    if (i < p_slice_header->num_long_term_sps) {
                        if (sps_ptr->num_long_term_ref_pics_sps > 1) {
                            if( bits_for_ltrp_in_sps > 0) {
                                p_slice_header->lt_idx_sps[i] = Parser::ReadBits(nalu, offset, bits_for_ltrp_in_sps);
                                p_slice_header->lt_rps.pocs[i] = sps_ptr->lt_rps.pocs[p_slice_header->lt_idx_sps[i]];  // PocLsbLt[]
                                p_slice_header->lt_rps.used_by_curr_pic[i] = sps_ptr->lt_rps.used_by_curr_pic[p_slice_header->lt_idx_sps[i]];  // UsedByCurrPicLt[]
                            }
                        }
                    } else {
                        p_slice_header->poc_lsb_lt[i] = Parser::ReadBits(nalu, offset, (sps_ptr->log2_max_pic_order_cnt_lsb_minus4 + 4));
                        p_slice_header->used_by_curr_pic_lt_flag[i] = Parser::GetBit(nalu, offset);
                        p_slice_header->lt_rps.pocs[i] = p_slice_header->poc_lsb_lt[i];  // PocLsbLt[]
                        p_slice_header->lt_rps.used_by_curr_pic[i] = p_slice_header->used_by_curr_pic_lt_flag[i];  // UsedByCurrPicLt[]
                    }
                    int delta_poc_msb_cycle_lt = 0;
                    p_slice_header->delta_poc_msb_present_flag[i] = Parser::GetBit(nalu, offset);
                    if (p_slice_header->delta_poc_msb_present_flag[i]) {
                        // Store DeltaPocMsbCycleLt in delta_poc_msb_cycle_lt for later use
                        delta_poc_msb_cycle_lt = Parser::ExpGolomb::ReadUe(nalu, offset);
                    }
                    if ( i == 0 || i == p_slice_header->num_long_term_sps) {
                        p_slice_header->delta_poc_msb_cycle_lt[i] = delta_poc_msb_cycle_lt;
                    } else {
                        p_slice_header->delta_poc_msb_cycle_lt[i] = delta_poc_msb_cycle_lt + p_slice_header->delta_poc_msb_cycle_lt[i - 1];
                    }
                }
            }
            if (p_slice_header->slice_type != HEVC_SLICE_TYPE_I) {
                CHECK_ALLOWED_RANGE("num_of_delta_pocs + num_long_term_pics", p_slice_header->st_rps.num_of_delta_pocs + p_slice_header->num_long_term_pics, 1, dpb_buffer_.dpb_size - 1);
            }
            if (sps_ptr->sps_temporal_mvp_enabled_flag) {
                p_slice_header->slice_temporal_mvp_enabled_flag = Parser::GetBit(nalu, offset);
            }
        }

        int chroma_array_type = sps_ptr->separate_colour_plane_flag ? 0 : sps_ptr->chroma_format_idc;  // ChromaArrayType
        if (sps_ptr->sample_adaptive_offset_enabled_flag) {
            p_slice_header->slice_sao_luma_flag = Parser::GetBit(nalu, offset);
            if (chroma_array_type)
            {
                p_slice_header->slice_sao_chroma_flag = Parser::GetBit(nalu, offset);
            }
        }

        if (p_slice_header->slice_type != HEVC_SLICE_TYPE_I) {
            p_slice_header->num_ref_idx_active_override_flag = Parser::GetBit(nalu, offset);
            if (p_slice_header->num_ref_idx_active_override_flag) {
                p_slice_header->num_ref_idx_l0_active_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
                CHECK_ALLOWED_MAX("num_ref_idx_l0_active_minus1", p_slice_header->num_ref_idx_l0_active_minus1, std::min(14, static_cast<int>(dpb_buffer_.dpb_size - 1)));
                if (p_slice_header->slice_type == HEVC_SLICE_TYPE_B) {
                    p_slice_header->num_ref_idx_l1_active_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
                    CHECK_ALLOWED_MAX("num_ref_idx_l1_active_minus1", p_slice_header->num_ref_idx_l1_active_minus1, std::min(14, static_cast<int>(dpb_buffer_.dpb_size - 1)));
                }
            } else {
                p_slice_header->num_ref_idx_l0_active_minus1 = pps_ptr->num_ref_idx_l0_default_active_minus1;
                if (p_slice_header->slice_type == HEVC_SLICE_TYPE_B) {
                    p_slice_header->num_ref_idx_l1_active_minus1 = pps_ptr->num_ref_idx_l1_default_active_minus1;
                }
            }

            // 7.3.6.2 Reference picture list modification
            // Calculate NumPicTotalCurr
            num_pic_total_curr_ = 0;
            HevcShortTermRps *st_rps_ptr = &p_slice_header->st_rps;
            for (int i = 0; i < st_rps_ptr->num_negative_pics; i++) {
                if (st_rps_ptr->used_by_curr_pic_s0[i]) {
                    num_pic_total_curr_++;
                }
            }
            for (int i = 0; i < st_rps_ptr->num_positive_pics; i++) {
                if (st_rps_ptr->used_by_curr_pic_s1[i]) {
                    num_pic_total_curr_++;
                }
            }

            HevcLongTermRps *lt_rps_ptr = &p_slice_header->lt_rps;
            // Check the combined list
            for (int i = 0; i < lt_rps_ptr->num_of_pics; i++) {
                if (lt_rps_ptr->used_by_curr_pic[i]) {
                    num_pic_total_curr_++;
                }
            }

            if (pps_ptr->lists_modification_present_flag && num_pic_total_curr_ > 1)
            {
                int list_entry_bits = 0;
                while ((1 << list_entry_bits) < num_pic_total_curr_) {
                    list_entry_bits++;
                }

                p_slice_header->ref_pic_list_modification_flag_l0 = Parser::GetBit(nalu, offset);
                if (p_slice_header->ref_pic_list_modification_flag_l0) {
                    for (int i = 0; i <= p_slice_header->num_ref_idx_l0_active_minus1; i++) {
                        p_slice_header->list_entry_l0[i] = Parser::ReadBits(nalu, offset, list_entry_bits);
                        //todo;; check_max
                    }
                }

                if (p_slice_header->slice_type == HEVC_SLICE_TYPE_B) {
                    p_slice_header->ref_pic_list_modification_flag_l1 = Parser::GetBit(nalu, offset);
                    if (p_slice_header->ref_pic_list_modification_flag_l1) {
                        for (int i = 0; i <= p_slice_header->num_ref_idx_l1_active_minus1; i++) {
                            p_slice_header->list_entry_l1[i] = Parser::ReadBits(nalu, offset, list_entry_bits);
                            //todo::checkmax
                        }
                    }
                }
            }

            if (p_slice_header->slice_type == HEVC_SLICE_TYPE_B) {
                p_slice_header->mvd_l1_zero_flag = Parser::GetBit(nalu, offset);
            }
            if (pps_ptr->cabac_init_present_flag) {
                p_slice_header->cabac_init_flag = Parser::GetBit(nalu, offset);
            }

            if (p_slice_header->slice_temporal_mvp_enabled_flag) {
                if (p_slice_header->slice_type == HEVC_SLICE_TYPE_B) {
                    p_slice_header->collocated_from_l0_flag = Parser::GetBit(nalu, offset);
                }
                if ((p_slice_header->collocated_from_l0_flag && p_slice_header->num_ref_idx_l0_active_minus1 > 0) || (!p_slice_header->collocated_from_l0_flag && p_slice_header->num_ref_idx_l1_active_minus1 > 0)) {
                    p_slice_header->collocated_ref_idx = Parser::ExpGolomb::ReadUe(nalu, offset);
                    if (p_slice_header->slice_type == HEVC_SLICE_TYPE_P || (p_slice_header->slice_type == HEVC_SLICE_TYPE_B && p_slice_header->collocated_from_l0_flag == 1)) {
                        CHECK_ALLOWED_MAX("collocated_ref_idx", p_slice_header->collocated_ref_idx, p_slice_header->num_ref_idx_l0_active_minus1);
                    } else {
                        CHECK_ALLOWED_MAX("collocated_ref_idx", p_slice_header->collocated_ref_idx, p_slice_header->num_ref_idx_l1_active_minus1);
                    }
                }
            }

            if ((pps_ptr->weighted_pred_flag && p_slice_header->slice_type == HEVC_SLICE_TYPE_P) || (pps_ptr->weighted_bipred_flag && p_slice_header->slice_type == HEVC_SLICE_TYPE_B)) {
                ParserResult ret;
                if ((ret = ParsePredWeightTable(p_slice_header, chroma_array_type, nalu, offset)) != PARSER_OK) {
                    return ret;
                }
            }
            p_slice_header->five_minus_max_num_merge_cand = Parser::ExpGolomb::ReadUe(nalu, offset);
        }

        p_slice_header->slice_qp_delta = Parser::ExpGolomb::ReadSe(nalu, offset);
        if (pps_ptr->pps_slice_chroma_qp_offsets_present_flag) {
            p_slice_header->slice_cb_qp_offset = Parser::ExpGolomb::ReadSe(nalu, offset);
            CHECK_ALLOWED_RANGE("slice_cb_qp_offset", p_slice_header->slice_cb_qp_offset, -12, 12);
            p_slice_header->slice_cr_qp_offset = Parser::ExpGolomb::ReadSe(nalu, offset);
            CHECK_ALLOWED_RANGE("slice_cr_qp_offset", p_slice_header->slice_cr_qp_offset, -12, 12);
        }
        if (pps_ptr->chroma_qp_offset_list_enabled_flag) {
            p_slice_header->cu_chroma_qp_offset_enabled_flag = Parser::GetBit(nalu, offset);
        }
        if (pps_ptr->deblocking_filter_override_enabled_flag) {
            p_slice_header->deblocking_filter_override_flag = Parser::GetBit(nalu, offset);
        }
        if (p_slice_header->deblocking_filter_override_flag) {
            p_slice_header->slice_deblocking_filter_disabled_flag = Parser::GetBit(nalu, offset);
            if ( !p_slice_header->slice_deblocking_filter_disabled_flag ) {
                p_slice_header->slice_beta_offset_div2 = Parser::ExpGolomb::ReadSe(nalu, offset);
                CHECK_ALLOWED_RANGE("slice_beta_offset_div2", p_slice_header->slice_beta_offset_div2, -6, 6);
                p_slice_header->slice_tc_offset_div2 = Parser::ExpGolomb::ReadSe(nalu, offset);
                CHECK_ALLOWED_RANGE("slice_tc_offset_div2", p_slice_header->slice_tc_offset_div2, -6, 6);
            }
        }

        if (pps_ptr->pps_loop_filter_across_slices_enabled_flag && (p_slice_header->slice_sao_luma_flag || p_slice_header->slice_sao_chroma_flag ||
!p_slice_header->slice_deblocking_filter_disabled_flag)) {
            p_slice_header->slice_loop_filter_across_slices_enabled_flag = Parser::GetBit(nalu, offset);
        }

        p_slice_header->is_received = 1;
        memcpy(&slice_header_copy_, p_slice_header, sizeof(HevcSliceSegHeader));
    } else {
        //dependant slice
        if (!slice_header_copy_.is_received) {
            return PARSER_WRONG_STATE;
        }
        memcpy(p_slice_header, &slice_header_copy_, sizeof(HevcSliceSegHeader));
        p_slice_header->first_slice_segment_in_pic_flag = temp_sh.first_slice_segment_in_pic_flag;
        p_slice_header->no_output_of_prior_pics_flag = temp_sh.no_output_of_prior_pics_flag;
        p_slice_header->slice_pic_parameter_set_id = temp_sh.slice_pic_parameter_set_id;
        p_slice_header->dependent_slice_segment_flag = temp_sh.dependent_slice_segment_flag;
        p_slice_header->slice_segment_address = temp_sh.slice_segment_address;
    }
    if (pps_ptr->tiles_enabled_flag || pps_ptr->entropy_coding_sync_enabled_flag) {
        int max_num_entry_point_offsets;  // 7.4.7.1
        if (!pps_ptr->tiles_enabled_flag && pps_ptr->entropy_coding_sync_enabled_flag) {
            max_num_entry_point_offsets = pic_height_in_ctbs_y_ - 1;
        } else if (pps_ptr->tiles_enabled_flag && !pps_ptr->entropy_coding_sync_enabled_flag) {
            max_num_entry_point_offsets = (pps_ptr->num_tile_columns_minus1 + 1) * (pps_ptr->num_tile_rows_minus1 + 1) - 1;
        } else {
            max_num_entry_point_offsets = (pps_ptr->num_tile_columns_minus1 + 1) * pic_height_in_ctbs_y_ - 1;
        }
        p_slice_header->num_entry_point_offsets = Parser::ExpGolomb::ReadUe(nalu, offset);
        if (p_slice_header->num_entry_point_offsets > max_num_entry_point_offsets) {
            p_slice_header->num_entry_point_offsets = max_num_entry_point_offsets;
        }

 #if 0 // do not parse syntax parameters that are not used by HW decode
       if (p_slice_header->num_entry_point_offsets) {
            p_slice_header->offset_len_minus1 = Parser::ExpGolomb::ReadUe(nalu, offset);
            for (int i = 0; i < p_slice_header->num_entry_point_offsets; i++) {
                p_slice_header->entry_point_offset_minus1[i] = Parser::ReadBits(nalu, offset, p_slice_header->offset_len_minus1 + 1);
            }
        }
#endif
    }

#if 0 // do not parse syntax parameters that are not used by HW decode
    if (pps_ptr->slice_segment_header_extension_present_flag) {
        p_slice_header->slice_segment_header_extension_length = Parser::ExpGolomb::ReadUe(nalu, offset);
        for (int i = 0; i < p_slice_header->slice_segment_header_extension_length; i++) {
            p_slice_header->slice_segment_header_extension_data_byte[i] = Parser::ReadBits(nalu, offset, 8);
        }
    }
#endif

#if DBGINFO
    PrintSliceSegHeader(p_slice_header);
#endif // DBGINFO

    return PARSER_OK;
}

bool HevcVideoParser::IsIdrPic(HevcNalUnitHeader *nal_header_ptr) {
    return (nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_IDR_W_RADL || nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_IDR_N_LP);
}

bool HevcVideoParser::IsBlaPic(HevcNalUnitHeader *nal_header_ptr) {
    return (nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_BLA_W_LP || nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_BLA_W_RADL || nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_BLA_N_LP);
}

bool HevcVideoParser::IsCraPic(HevcNalUnitHeader *nal_header_ptr) {
    return (nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_CRA_NUT);
}

bool HevcVideoParser::IsRaslPic(HevcNalUnitHeader *nal_header_ptr) {
    return (nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_RASL_N || nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_RASL_R);
}

bool HevcVideoParser::IsRadlPic(HevcNalUnitHeader *nal_header_ptr) {
    return (nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_RADL_N || nal_header_ptr->nal_unit_type == NAL_UNIT_CODED_SLICE_RADL_R);
}

bool HevcVideoParser::IsIrapPic(HevcNalUnitHeader *nal_header_ptr) {
    return (nal_header_ptr->nal_unit_type >= NAL_UNIT_CODED_SLICE_BLA_W_LP && nal_header_ptr->nal_unit_type <= NAL_UNIT_RESERVED_IRAP_VCL23);
}

bool HevcVideoParser::IsRefPic(HevcNalUnitHeader *nal_header_ptr) {
    if (((nal_header_ptr->nal_unit_type <= NAL_UNIT_RESERVED_VCL_R15) && ((nal_header_ptr->nal_unit_type % 2) != 0)) ||
         ((nal_header_ptr->nal_unit_type >= NAL_UNIT_CODED_SLICE_BLA_W_LP) && (nal_header_ptr->nal_unit_type <= NAL_UNIT_RESERVED_IRAP_VCL23))) {
        return true;
    } else {
        return false;
    }
}

void HevcVideoParser::CalculateCurrPoc() {
    HevcSliceSegHeader *p_slice_header = &slice_info_list_[0].slice_header;
    // Record decode order count
    curr_pic_info_.decode_order_count = pic_count_;
    if (IsIdrPic(&slice_nal_unit_header_)) {
        curr_pic_info_.pic_order_cnt = 0;
        curr_pic_info_.prev_poc_lsb = 0;
        curr_pic_info_.prev_poc_msb = 0;
        curr_pic_info_.slice_pic_order_cnt_lsb = 0;
    } else {
        int max_poc_lsb = 1 << (sps_list_[active_sps_id_].log2_max_pic_order_cnt_lsb_minus4 + 4);  // MaxPicOrderCntLsb
        int poc_msb;  // PicOrderCntMsb
        // If the current picture is an IRAP picture with NoRaslOutputFlag equal to 1, PicOrderCntMsb is set equal to 0.
        if (IsIrapPic(&slice_nal_unit_header_) && no_rasl_output_flag_ == 1) {
            poc_msb = 0;
        } else {
            if ((p_slice_header->slice_pic_order_cnt_lsb < curr_pic_info_.prev_poc_lsb) && ((curr_pic_info_.prev_poc_lsb - p_slice_header->slice_pic_order_cnt_lsb) >= (max_poc_lsb / 2))) {
                poc_msb = curr_pic_info_.prev_poc_msb + max_poc_lsb;
            } else if ((p_slice_header->slice_pic_order_cnt_lsb > curr_pic_info_.prev_poc_lsb) && ((p_slice_header->slice_pic_order_cnt_lsb - curr_pic_info_.prev_poc_lsb) > (max_poc_lsb / 2))) {
                poc_msb = curr_pic_info_.prev_poc_msb - max_poc_lsb;
            } else {
                poc_msb = curr_pic_info_.prev_poc_msb;
            }
        }

        curr_pic_info_.pic_order_cnt = poc_msb + p_slice_header->slice_pic_order_cnt_lsb;
        curr_pic_info_.slice_pic_order_cnt_lsb = p_slice_header->slice_pic_order_cnt_lsb;
        if ((slice_nal_unit_header_.nuh_temporal_id_plus1 - 1) == 0 && IsRefPic(&slice_nal_unit_header_) && !IsRaslPic(&slice_nal_unit_header_) && !IsRadlPic(&slice_nal_unit_header_)) {
            curr_pic_info_.prev_poc_lsb = p_slice_header->slice_pic_order_cnt_lsb;
            curr_pic_info_.prev_poc_msb = poc_msb;
        }
    }
}

void HevcVideoParser::DecodeRps() {
    int i, j, k;
    int curr_delta_poc_msb_present_flag[HEVC_MAX_NUM_REF_PICS] = {0}; // CurrDeltaPocMsbPresentFlag
    int foll_delta_poc_msb_present_flag[HEVC_MAX_NUM_REF_PICS] = {0}; // FollDeltaPocMsbPresentFlag
    int max_poc_lsb = 1 << (sps_list_[active_sps_id_].log2_max_pic_order_cnt_lsb_minus4 + 4);  // MaxPicOrderCntLsb
    HevcSliceSegHeader *p_slice_header = &slice_info_list_[0].slice_header;

    // When the current picture is an IRAP picture with NoRaslOutputFlag equal to 1, all reference pictures with
    // nuh_layer_id equal to currPicLayerId currently in the DPB (if any) are marked as "unused for reference".
    if (IsIrapPic(&slice_nal_unit_header_) && no_rasl_output_flag_ == 1) {
        for (i = 0; i < HEVC_MAX_DPB_FRAMES; i++) {
            dpb_buffer_.frame_buffer_list[i].is_reference = kUnusedForReference;
        }
    }

    if (IsIdrPic(&slice_nal_unit_header_)) {
        num_poc_st_curr_before_ = 0;
        num_poc_st_curr_after_ = 0;
        num_poc_st_foll_ = 0;
        num_poc_lt_curr_ = 0;
        num_poc_lt_foll_ = 0;
        memset(poc_st_curr_before_, 0, sizeof(int32_t) * HEVC_MAX_NUM_REF_PICS);
        memset(poc_st_curr_after_, 0, sizeof(int32_t) * HEVC_MAX_NUM_REF_PICS);
        memset(poc_st_foll_, 0, sizeof(int32_t) * HEVC_MAX_NUM_REF_PICS);
        memset(poc_lt_curr_, 0, sizeof(int32_t) * HEVC_MAX_NUM_REF_PICS);
        memset(poc_lt_foll_, 0, sizeof(int32_t) * HEVC_MAX_NUM_REF_PICS);
    } else {
        HevcShortTermRps *rps_ptr = &p_slice_header->st_rps;
        for (i = 0, j = 0, k = 0; i < rps_ptr->num_negative_pics; i++) {
            if (rps_ptr->used_by_curr_pic_s0[i]) {
                poc_st_curr_before_[j++] = curr_pic_info_.pic_order_cnt + rps_ptr->delta_poc_s0[i];
            } else {
                poc_st_foll_[k++] = curr_pic_info_.pic_order_cnt + rps_ptr->delta_poc_s0[i];
            }
        }
        num_poc_st_curr_before_ = j;

        for (i = 0, j = 0; i < rps_ptr->num_positive_pics; i++ ) {
            if (rps_ptr->used_by_curr_pic_s1[i]) {
                poc_st_curr_after_[j++] = curr_pic_info_.pic_order_cnt + rps_ptr->delta_poc_s1[i];
            } else {
                poc_st_foll_[k++] = curr_pic_info_.pic_order_cnt + rps_ptr->delta_poc_s1[i];
            }
        }
        num_poc_st_curr_after_ = j;
        num_poc_st_foll_ = k;

        HevcLongTermRps *lt_rps_ptr = &p_slice_header->lt_rps;
        for (i = 0, j = 0, k = 0; i < lt_rps_ptr->num_of_pics; i++) {
            uint32_t poc_lt = lt_rps_ptr->pocs[i];  // oocLt
            if (p_slice_header->delta_poc_msb_present_flag[i]) {
                poc_lt += curr_pic_info_.pic_order_cnt - p_slice_header->delta_poc_msb_cycle_lt[i] * max_poc_lsb - (curr_pic_info_.pic_order_cnt & (max_poc_lsb - 1));
            }

            if (lt_rps_ptr->used_by_curr_pic[i]) {
                poc_lt_curr_[j] = poc_lt;
                curr_delta_poc_msb_present_flag[j++] = p_slice_header->delta_poc_msb_present_flag[i];
            } else {
                poc_lt_foll_[k] = poc_lt;
                foll_delta_poc_msb_present_flag[k++] = p_slice_header->delta_poc_msb_present_flag[i];
            }
        }
        num_poc_lt_curr_ = j;
        num_poc_lt_foll_ = k;

        /*
        * RPS derivation and picture marking
        */
        // Init to a valid index value to take care of undecodable RASL pictures, whose reference pictures are emptied after a CRA.
        for (i = 0; i < HEVC_MAX_NUM_REF_PICS; i++) {
            ref_pic_set_st_curr_before_[i] = 0;
            ref_pic_set_st_curr_after_[i] = 0;
            ref_pic_set_st_foll_[i] = 0;
            ref_pic_set_lt_curr_[i] = 0;
            ref_pic_set_lt_foll_[i] = 0;
        }

        // Mark all in DPB as unused. We will mark them back while we go through the ref lists. The rest will be actually unused.
        for (i = 0; i < HEVC_MAX_DPB_FRAMES; i++) {
            dpb_buffer_.frame_buffer_list[i].is_reference = kUnusedForReference;
        }

        /// Short term reference pictures
        for (i = 0; i < num_poc_st_curr_before_; i++) {
            for (j = 0; j < HEVC_MAX_DPB_FRAMES; j++) {
                if (poc_st_curr_before_[i] == dpb_buffer_.frame_buffer_list[j].pic_order_cnt && dpb_buffer_.frame_buffer_list[j].use_status != kNotUsed) {
                    ref_pic_set_st_curr_before_[i] = j;  // RefPicSetStCurrBefore. Use DPB buffer index for now
                    dpb_buffer_.frame_buffer_list[j].is_reference = kUsedForShortTerm;
                    break;
                }
            }
        }

        for (i = 0; i < num_poc_st_curr_after_; i++) {
            for (j = 0; j < HEVC_MAX_DPB_FRAMES; j++) {
                if (poc_st_curr_after_[i] == dpb_buffer_.frame_buffer_list[j].pic_order_cnt && dpb_buffer_.frame_buffer_list[j].use_status != kNotUsed) {
                    ref_pic_set_st_curr_after_[i] = j;  // RefPicSetStCurrAfter
                    dpb_buffer_.frame_buffer_list[j].is_reference = kUsedForShortTerm;
                    break;
                }
            }
        }

        for ( i = 0; i < num_poc_st_foll_; i++ ) {
            for (j = 0; j < HEVC_MAX_DPB_FRAMES; j++) {
                if (poc_st_foll_[i] == dpb_buffer_.frame_buffer_list[j].pic_order_cnt && dpb_buffer_.frame_buffer_list[j].use_status != kNotUsed) {
                    ref_pic_set_st_foll_[i] = j;  // RefPicSetStFoll
                    dpb_buffer_.frame_buffer_list[j].is_reference = kUsedForShortTerm;
                    break;
                }
            }
        }

        /// Long term reference pictures
        for (i = 0; i < num_poc_lt_curr_; i++) {
            for (j = 0; j < HEVC_MAX_DPB_FRAMES; j++) {
                if(!curr_delta_poc_msb_present_flag[i]) {
                    if (poc_lt_curr_[i] == (dpb_buffer_.frame_buffer_list[j].pic_order_cnt & (max_poc_lsb - 1)) && dpb_buffer_.frame_buffer_list[j].use_status) {
                        ref_pic_set_lt_curr_[i] = j;  // RefPicSetLtCurr
                        dpb_buffer_.frame_buffer_list[j].is_reference = kUsedForLongTerm;
                        break;
                    }
                } else {
                    if (poc_lt_curr_[i] == dpb_buffer_.frame_buffer_list[j].pic_order_cnt) {
                        ref_pic_set_lt_curr_[i] = j;  // RefPicSetLtCurr
                        dpb_buffer_.frame_buffer_list[j].is_reference = kUsedForLongTerm;
                        break;
                    }
                }
            }
        }

        for (i = 0; i < num_poc_lt_foll_; i++) {
            for (j = 0; j < HEVC_MAX_DPB_FRAMES; j++) {
                if(!foll_delta_poc_msb_present_flag[i]) {
                    if (poc_lt_foll_[i] == (dpb_buffer_.frame_buffer_list[j].pic_order_cnt & (max_poc_lsb - 1)) && dpb_buffer_.frame_buffer_list[j].use_status) {
                        ref_pic_set_lt_foll_[i] = j;  // RefPicSetLtFoll
                        dpb_buffer_.frame_buffer_list[j].is_reference = kUsedForLongTerm;
                        break;
                    }
                } else {
                    if (poc_lt_foll_[i] == dpb_buffer_.frame_buffer_list[j].pic_order_cnt) {
                        ref_pic_set_lt_foll_[i] = j;  // RefPicSetLtFoll
                        dpb_buffer_.frame_buffer_list[j].is_reference = kUsedForLongTerm;
                        break;
                    }
                }
            }
        }
    }
}

void HevcVideoParser::ConstructRefPicLists(HevcSliceInfo *p_slice_info) {
    HevcSliceSegHeader *p_slice_header = &p_slice_info->slice_header;
    uint32_t num_rps_curr_temp_list; // NumRpsCurrTempList0 or NumRpsCurrTempList1;
    int i;
    int rIdx;
    uint32_t ref_pic_list_temp[HEVC_MAX_NUM_REF_PICS] = {0};  // RefPicListTemp0 or RefPicListTemp1

    /// List 0
    rIdx = 0;
    num_rps_curr_temp_list = std::max(p_slice_header->num_ref_idx_l0_active_minus1 + 1, num_pic_total_curr_);

    while ((num_poc_st_curr_before_ | num_poc_st_curr_after_ | num_poc_lt_curr_) && (rIdx < num_rps_curr_temp_list)) {
        for (i = 0; i < num_poc_st_curr_before_ && rIdx < num_rps_curr_temp_list; rIdx++, i++) {
            ref_pic_list_temp[rIdx] = ref_pic_set_st_curr_before_[i];
        }

        for (i = 0; i < num_poc_st_curr_after_ && rIdx < num_rps_curr_temp_list; rIdx++, i++) {
            ref_pic_list_temp[rIdx] = ref_pic_set_st_curr_after_[i];
        }

        for (i = 0; i < num_poc_lt_curr_ && rIdx < num_rps_curr_temp_list; rIdx++, i++) {
            ref_pic_list_temp[rIdx] = ref_pic_set_lt_curr_[i];
        }
    }

    for( rIdx = 0; rIdx <= p_slice_header->num_ref_idx_l0_active_minus1; rIdx++) {
        p_slice_info->ref_pic_list_0_[rIdx] = p_slice_header->ref_pic_list_modification_flag_l0 ? ref_pic_list_temp[p_slice_header->list_entry_l0[rIdx]] : ref_pic_list_temp[rIdx];
    }

    /// List 1
    if (p_slice_header->slice_type == HEVC_SLICE_TYPE_B) {
        rIdx = 0;
        num_rps_curr_temp_list = std::max(p_slice_header->num_ref_idx_l1_active_minus1 + 1, num_pic_total_curr_);

        while ((num_poc_st_curr_after_ | num_poc_st_curr_before_ | num_poc_lt_curr_) && (rIdx < num_rps_curr_temp_list)) {
            for (i = 0; i < num_poc_st_curr_after_ && rIdx < num_rps_curr_temp_list; rIdx++, i++) {
                ref_pic_list_temp[rIdx] = ref_pic_set_st_curr_after_[i];
            }

            for (i = 0; i < num_poc_st_curr_before_ && rIdx < num_rps_curr_temp_list; rIdx++, i++ ) {
                ref_pic_list_temp[rIdx] = ref_pic_set_st_curr_before_[i];
            }

            for (i = 0; i < num_poc_lt_curr_ && rIdx < num_rps_curr_temp_list; rIdx++, i++ ) {
                ref_pic_list_temp[rIdx] = ref_pic_set_lt_curr_[i];
            }
        }

        for( rIdx = 0; rIdx <= p_slice_header->num_ref_idx_l1_active_minus1; rIdx++) {
            p_slice_info->ref_pic_list_1_[rIdx] = p_slice_header->ref_pic_list_modification_flag_l1 ? ref_pic_list_temp[p_slice_header->list_entry_l1[rIdx]] : ref_pic_list_temp[rIdx];
        }
    }
}

void HevcVideoParser::InitDpb() {
    memset(&dpb_buffer_, 0, sizeof(DecodedPictureBuffer));
    for (int i = 0; i < HEVC_MAX_DPB_FRAMES; i++) {
        dpb_buffer_.frame_buffer_list[i].pic_idx = i;
        dpb_buffer_.frame_buffer_list[i].is_reference = kUnusedForReference;
        dpb_buffer_.frame_buffer_list[i].pic_output_flag = 0;
        dpb_buffer_.frame_buffer_list[i].use_status = kNotUsed;
    }
    dpb_buffer_.dpb_size = 0;
    dpb_buffer_.dpb_fullness = 0;
    dpb_buffer_.num_pics_needed_for_output = 0;
}

void HevcVideoParser::EmptyDpb() {
    for (int i = 0; i < HEVC_MAX_DPB_FRAMES; i++) {
        dpb_buffer_.frame_buffer_list[i].is_reference = kUnusedForReference;
        dpb_buffer_.frame_buffer_list[i].pic_output_flag = 0;
        dpb_buffer_.frame_buffer_list[i].use_status = kNotUsed;
        decode_buffer_pool_[dpb_buffer_.frame_buffer_list[i].dec_buf_idx].use_status = kNotUsed;
    }
    dpb_buffer_.dpb_fullness = 0;
    dpb_buffer_.num_pics_needed_for_output = 0;
    num_output_pics_ = 0;
}

int HevcVideoParser::FlushDpb() {
    // Bump the remaining pictures
    while (dpb_buffer_.num_pics_needed_for_output) {
        if (BumpPicFromDpb() != PARSER_OK) {
            return PARSER_FAIL;
        }
    }
    if (pfn_display_picture_cb_ && num_output_pics_ > 0) {
        if (OutputDecodedPictures(true) != PARSER_OK) {
            return PARSER_FAIL;
        }
    }

    return PARSER_OK;
}

int HevcVideoParser::MarkOutputPictures() {
    int i;

    if (IsIrapPic(&slice_nal_unit_header_) && no_rasl_output_flag_ == 1 && pic_count_ != 0) {
        if (IsCraPic(&slice_nal_unit_header_)) {
            no_output_of_prior_pics_flag = 1;
        } else {
            no_output_of_prior_pics_flag = slice_info_list_[0].slice_header.no_output_of_prior_pics_flag;
        }

        if (!no_output_of_prior_pics_flag) {
            // Bump the remaining pictures
            if (FlushDpb() != PARSER_OK) {
                return PARSER_FAIL;
            }
        }

        // We can still have remaining undisplayed frames due to display delay feature
        if (num_output_pics_) {
            OutputDecodedPictures(true);
        }
        EmptyDpb();
    } else {
        for (i = 0; i < HEVC_MAX_DPB_FRAMES; i++) {
            if (dpb_buffer_.frame_buffer_list[i].is_reference == kUnusedForReference && dpb_buffer_.frame_buffer_list[i].pic_output_flag == 0 && dpb_buffer_.frame_buffer_list[i].use_status) {
                dpb_buffer_.frame_buffer_list[i].use_status = kNotUsed;
                decode_buffer_pool_[dpb_buffer_.frame_buffer_list[i].dec_buf_idx].use_status &= ~kFrameUsedForDecode;
                if (dpb_buffer_.dpb_fullness > 0) {
                    dpb_buffer_.dpb_fullness--;
                } else {
                    logger_.ErrorLog(MakeMsg("Invalid DPB buffer fullness:" + TOSTR(dpb_buffer_.dpb_fullness)));
                    return PARSER_FAIL;
                }
            }
        }

        HevcSeqParamSet *sps_ptr = &sps_list_[active_sps_id_];
        uint32_t highest_tid = sps_ptr->sps_max_sub_layers_minus1; // HighestTid
        uint32_t max_num_reorder_pics = sps_ptr->sps_max_num_reorder_pics[highest_tid];
        uint32_t max_dec_pic_buffering = sps_ptr->sps_max_dec_pic_buffering_minus1[highest_tid] + 1;

        while (dpb_buffer_.dpb_fullness >= max_dec_pic_buffering) {
            if (BumpPicFromDpb() != PARSER_OK) {
                return PARSER_FAIL;
            }
        }

        while (dpb_buffer_.num_pics_needed_for_output > max_num_reorder_pics) {
            if (BumpPicFromDpb() != PARSER_OK) {
                return PARSER_FAIL;
            }
        }

        // Skip SpsMaxLatencyPictures check as SpsMaxLatencyPictures >= sps_max_num_reorder_pics.
    }

    return PARSER_OK;
}

ParserResult HevcVideoParser::FindFreeInDecBufPool() {
    int dec_buf_index;

    // Find a free buffer in decode buffer pool
    for (dec_buf_index = 0; dec_buf_index < dec_buf_pool_size_; dec_buf_index++) {
        if (decode_buffer_pool_[dec_buf_index].use_status == kNotUsed) {
            break;
        }
    }
    if (dec_buf_index == dec_buf_pool_size_) {
        logger_.ErrorLog(MakeMsg("Could not find a free buffer in decode buffer pool."));
        return PARSER_NOT_FOUND;
    }
    curr_pic_info_.dec_buf_idx = dec_buf_index;
    return PARSER_OK;
}

ParserResult HevcVideoParser::FindFreeInDpbAndMark() {
    int i;

    // Look for an empty buffer in DPB with longest decode history (lowest decode count)
    uint32_t min_decode_order_count = 0xFFFFFFFF;
    int index = dpb_buffer_.dpb_size;
    for (i = 0; i < dpb_buffer_.dpb_size; i++) {
        if (dpb_buffer_.frame_buffer_list[i].use_status == kNotUsed) {
            if (dpb_buffer_.frame_buffer_list[i].decode_order_count < min_decode_order_count) {
                min_decode_order_count = dpb_buffer_.frame_buffer_list[i].decode_order_count;
                index = i;
            }
        }
    }
    if (index == dpb_buffer_.dpb_size) {
        logger_.ErrorLog(MakeMsg("Error! DPB buffer overflow! Fullness = " + TOSTR(dpb_buffer_.dpb_fullness)));
        return PARSER_NOT_FOUND;
    }

    curr_pic_info_.pic_idx = index;
    curr_pic_info_.is_reference = kUsedForShortTerm;
    // dpb_buffer_.frame_buffer_list[i].pic_idx is already set in InitDpb().
    dpb_buffer_.frame_buffer_list[index].dec_buf_idx = curr_pic_info_.dec_buf_idx;
    dpb_buffer_.frame_buffer_list[index].pic_order_cnt = curr_pic_info_.pic_order_cnt;
    dpb_buffer_.frame_buffer_list[index].prev_poc_lsb = curr_pic_info_.prev_poc_lsb;
    dpb_buffer_.frame_buffer_list[index].prev_poc_msb = curr_pic_info_.prev_poc_msb;
    dpb_buffer_.frame_buffer_list[index].slice_pic_order_cnt_lsb = curr_pic_info_.slice_pic_order_cnt_lsb;
    dpb_buffer_.frame_buffer_list[index].decode_order_count = curr_pic_info_.decode_order_count;
    dpb_buffer_.frame_buffer_list[index].pic_output_flag = curr_pic_info_.pic_output_flag;
    dpb_buffer_.frame_buffer_list[index].is_reference = kUsedForShortTerm;
    dpb_buffer_.frame_buffer_list[index].use_status = kFrameUsedForDecode;

    if (dpb_buffer_.frame_buffer_list[index].pic_output_flag) {
        dpb_buffer_.num_pics_needed_for_output++;
    }
    dpb_buffer_.dpb_fullness++;

    // Mark as used in decode buffer pool
    decode_buffer_pool_[curr_pic_info_.dec_buf_idx].use_status |= kFrameUsedForDecode;
    if (pfn_display_picture_cb_ && curr_pic_info_.pic_output_flag) {
        decode_buffer_pool_[curr_pic_info_.dec_buf_idx].use_status |= kFrameUsedForDisplay;
    }
    decode_buffer_pool_[curr_pic_info_.dec_buf_idx].pic_order_cnt = curr_pic_info_.pic_order_cnt;
    decode_buffer_pool_[curr_pic_info_.dec_buf_idx].pts = curr_pts_;

    HevcSeqParamSet *sps_ptr = &sps_list_[active_sps_id_];
    uint32_t highest_tid = sps_ptr->sps_max_sub_layers_minus1; // HighestTid
    uint32_t max_num_reorder_pics = sps_ptr->sps_max_num_reorder_pics[highest_tid];

    while ( dpb_buffer_.num_pics_needed_for_output > max_num_reorder_pics) {
        if (BumpPicFromDpb() != PARSER_OK) {
            return PARSER_FAIL;
        }
    }

    // Skip SpsMaxLatencyPictures check as SpsMaxLatencyPictures >= sps_max_num_reorder_pics.

    return PARSER_OK;
}

int HevcVideoParser::BumpPicFromDpb() {
    int32_t min_poc = 0x7FFFFFFF;  // largest possible POC value 2^31 - 1
    int min_poc_pic_idx = HEVC_MAX_DPB_FRAMES;
    int i;

    for (i = 0; i < HEVC_MAX_DPB_FRAMES; i++) {
        if (dpb_buffer_.frame_buffer_list[i].pic_output_flag && dpb_buffer_.frame_buffer_list[i].use_status) {
            if (dpb_buffer_.frame_buffer_list[i].pic_order_cnt < min_poc) {
                min_poc = dpb_buffer_.frame_buffer_list[i].pic_order_cnt;
                min_poc_pic_idx = i;
            }
        }
    }
    if (min_poc_pic_idx >= HEVC_MAX_DPB_FRAMES) {
        // No picture that is needed for ouput is found
        return PARSER_OK;
    }

    // Mark as "not needed for output"
    dpb_buffer_.frame_buffer_list[min_poc_pic_idx].pic_output_flag = 0;
    if (dpb_buffer_.num_pics_needed_for_output > 0) {
        dpb_buffer_.num_pics_needed_for_output--;
    }

    // If it is not used for reference, empty it.
    if (dpb_buffer_.frame_buffer_list[min_poc_pic_idx].is_reference == kUnusedForReference) {
        dpb_buffer_.frame_buffer_list[min_poc_pic_idx].use_status = kNotUsed;
        decode_buffer_pool_[dpb_buffer_.frame_buffer_list[min_poc_pic_idx].dec_buf_idx].use_status &= ~kFrameUsedForDecode;
        if (dpb_buffer_.dpb_fullness > 0 ) {
            dpb_buffer_.dpb_fullness--;
        }
    }

    // Insert into output/display picture list
    if (pfn_display_picture_cb_) {
        if (num_output_pics_ >= dec_buf_pool_size_) {
            logger_.ErrorLog(MakeMsg("Error! Decode buffer pool overflow!"));
            return PARSER_OUT_OF_RANGE;
        } else {
            output_pic_list_[num_output_pics_] = dpb_buffer_.frame_buffer_list[min_poc_pic_idx].dec_buf_idx;
            num_output_pics_++;
        }
    }

    return PARSER_OK;
}

#if DBGINFO
void HevcVideoParser::PrintVps(HevcVideoParamSet *vps_ptr) {
    MSG("=== hevc_video_parameter_set_t ===");
    MSG("vps_video_parameter_set_id               = " <<  vps_ptr->vps_video_parameter_set_id);
    MSG("vps_base_layer_internal_flag             = " <<  vps_ptr->vps_base_layer_internal_flag);
    MSG("vps_base_layer_available_flag            = " <<  vps_ptr->vps_base_layer_available_flag);
    MSG("vps_max_layers_minus1                    = " <<  vps_ptr->vps_max_layers_minus1);
    MSG("vps_max_sub_layers_minus1                = " <<  vps_ptr->vps_max_sub_layers_minus1);
    MSG("vps_temporal_id_nesting_flag             = " <<  vps_ptr->vps_temporal_id_nesting_flag);
    MSG("vps_reserved_0xffff_16bits               = " <<  vps_ptr->vps_reserved_0xffff_16bits);

    MSG("Profile tier level:");
    MSG("general_profile_space                    = " <<  vps_ptr->profile_tier_level.general_profile_space);
    MSG("general_tier_flag                        = " <<  vps_ptr->profile_tier_level.general_tier_flag);
    MSG("general_profile_idc                      = " <<  vps_ptr->profile_tier_level.general_profile_idc);
    MSG_NO_NEWLINE("general_profile_compatibility_flag[32]: ");
    for(int i = 0; i < 32; i++) {
        MSG_NO_NEWLINE(" " << vps_ptr->profile_tier_level.general_profile_compatibility_flag[i]);
    }
    MSG("");
    MSG("general_progressive_source_flag          = " <<  vps_ptr->profile_tier_level.general_progressive_source_flag);
    MSG("general_interlaced_source_flag           = " <<  vps_ptr->profile_tier_level.general_interlaced_source_flag);
    MSG("general_non_packed_constraint_flag       = " <<  vps_ptr->profile_tier_level.general_non_packed_constraint_flag);
    MSG("general_frame_only_constraint_flag       = " <<  vps_ptr->profile_tier_level.general_frame_only_constraint_flag);
    MSG("general_reserved_zero_44bits             = " <<  vps_ptr->profile_tier_level.general_reserved_zero_44bits);
    MSG("general_level_idc                        = " <<  vps_ptr->profile_tier_level.general_level_idc);

    MSG("vps_sub_layer_ordering_info_present_flag = " <<  vps_ptr->vps_sub_layer_ordering_info_present_flag);
    MSG_NO_NEWLINE("vps_max_dec_pic_buffering_minus1[]: ");
    for(int i = 0; i < 7; i++) {
        MSG_NO_NEWLINE(" " << vps_ptr->vps_max_dec_pic_buffering_minus1[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("vps_max_num_reorder_pics[]: ");
    for(int i = 0; i < 7; i++) {
        MSG_NO_NEWLINE(" " << vps_ptr->vps_max_num_reorder_pics[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("vps_max_latency_increase_plus1[]: ");
    for(int i = 0; i < 7; i++) {
        MSG_NO_NEWLINE(" " << vps_ptr->vps_max_latency_increase_plus1[i]);
    }
    MSG("");
    MSG("vps_max_layer_id                         = " <<  vps_ptr->vps_max_layer_id);
    MSG("vps_num_layer_sets_minus1                = " <<  vps_ptr->vps_num_layer_sets_minus1);
    MSG("vps_timing_info_present_flag             = " <<  vps_ptr->vps_timing_info_present_flag);
    MSG("vps_num_hrd_parameters                   = " <<  vps_ptr->vps_num_hrd_parameters);
    MSG("vps_extension_flag                       = " <<  vps_ptr->vps_extension_flag);
    MSG("vps_extension_data_flag                  = " <<  vps_ptr->vps_extension_data_flag);
    MSG("");
}

void HevcVideoParser::PrintSps(HevcSeqParamSet *sps_ptr) {
    MSG("=== hevc_sequence_parameter_set_t ===");
    MSG("sps_video_parameter_set_id                = " <<  sps_ptr->sps_video_parameter_set_id);
    MSG("sps_max_sub_layers_minus1                 = " <<  sps_ptr->sps_max_sub_layers_minus1);
    MSG("sps_temporal_id_nesting_flag              = " <<  sps_ptr->sps_temporal_id_nesting_flag);

    MSG("Profile tier level:");
    MSG("general_profile_space                     = " <<  sps_ptr->profile_tier_level.general_profile_space);
    MSG("general_tier_flag                         = " <<  sps_ptr->profile_tier_level.general_tier_flag);
    MSG("general_profile_idc                       = " <<  sps_ptr->profile_tier_level.general_profile_idc);
    MSG("general_profile_compatibility_flag[32]:");
    for(int i = 0; i < 32; i++) {
        MSG_NO_NEWLINE(" " << sps_ptr->profile_tier_level.general_profile_compatibility_flag[i]);
    }
    MSG("");
    MSG("general_progressive_source_flag           = " <<  sps_ptr->profile_tier_level.general_progressive_source_flag);
    MSG("general_interlaced_source_flag            = " <<  sps_ptr->profile_tier_level.general_interlaced_source_flag);
    MSG("general_non_packed_constraint_flag        = " <<  sps_ptr->profile_tier_level.general_non_packed_constraint_flag);
    MSG("general_frame_only_constraint_flag        = " <<  sps_ptr->profile_tier_level.general_frame_only_constraint_flag);
    MSG("general_reserved_zero_44bits              = " <<  sps_ptr->profile_tier_level.general_reserved_zero_44bits);
    MSG("general_level_idc                         = " <<  sps_ptr->profile_tier_level.general_level_idc);

    MSG("sps_seq_parameter_set_id                  = " <<  sps_ptr->sps_seq_parameter_set_id);
    MSG("chroma_format_idc                         = " <<  sps_ptr->chroma_format_idc);
    MSG("separate_colour_plane_flag                = " <<  sps_ptr->separate_colour_plane_flag);
    MSG("pic_width_in_luma_samples                 = " <<  sps_ptr->pic_width_in_luma_samples);
    MSG("pic_height_in_luma_samples                = " <<  sps_ptr->pic_height_in_luma_samples);
    MSG("conformance_window_flag                   = " <<  sps_ptr->conformance_window_flag);
    MSG("conf_win_left_offset                      = " <<  sps_ptr->conf_win_left_offset);
    MSG("conf_win_right_offset                     = " <<  sps_ptr->conf_win_right_offset);
    MSG("conf_win_top_offset                       = " <<  sps_ptr->conf_win_top_offset);
    MSG("conf_win_bottom_offset                    = " <<  sps_ptr->conf_win_bottom_offset);
    MSG("bit_depth_luma_minus8                     = " <<  sps_ptr->bit_depth_luma_minus8);
    MSG("bit_depth_chroma_minus8                   = " <<  sps_ptr->bit_depth_chroma_minus8);
    MSG("log2_max_pic_order_cnt_lsb_minus4         = " <<  sps_ptr->log2_max_pic_order_cnt_lsb_minus4);
    MSG("sps_sub_layer_ordering_info_present_flag  = " <<  sps_ptr->sps_sub_layer_ordering_info_present_flag);
    MSG_NO_NEWLINE("sps_max_dec_pic_buffering_minus1[]:");
    for(int i = 0; i < 7; i++) {
        MSG_NO_NEWLINE(" " << sps_ptr->sps_max_dec_pic_buffering_minus1[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("sps_max_num_reorder_pics[]:");
    for(int i = 0; i < 7; i++) {
        MSG_NO_NEWLINE(" " << sps_ptr->sps_max_num_reorder_pics[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("sps_max_latency_increase_plus1[]:");
    for(int i = 0; i < 7; i++) {
        MSG_NO_NEWLINE(" " << sps_ptr->sps_max_latency_increase_plus1[i]);
    }
    MSG("");
    MSG("log2_min_luma_coding_block_size_minus3    = " <<  sps_ptr->log2_min_luma_coding_block_size_minus3);
    MSG("log2_diff_max_min_luma_coding_block_size  = " <<  sps_ptr->log2_diff_max_min_luma_coding_block_size);
    MSG("log2_min_luma_transform_block_size_minus2      = " <<  sps_ptr->log2_min_luma_transform_block_size_minus2);
    MSG("log2_diff_max_min_luma_transform_block_size    = " <<  sps_ptr->log2_diff_max_min_luma_transform_block_size);
    MSG("max_transform_hierarchy_depth_inter       = " <<  sps_ptr->max_transform_hierarchy_depth_inter);
    MSG("max_transform_hierarchy_depth_intra       = " <<  sps_ptr->max_transform_hierarchy_depth_intra);
    MSG("scaling_list_enabled_flag                 = " <<  sps_ptr->scaling_list_enabled_flag);
    MSG("sps_scaling_list_data_present_flag        = " <<  sps_ptr->sps_scaling_list_data_present_flag);
    MSG("Scaling list:");
    for (int i = 0; i < HEVC_SCALING_LIST_SIZE_NUM; i++) {
        for (int j = 0; j < HEVC_SCALING_LIST_NUM; j++) {
            MSG_NO_NEWLINE("scaling_list[" << i <<"][" << j << "][]:")
            for (int k = 0; k < HEVC_SCALING_LIST_MAX_INDEX; k++) {
                MSG_NO_NEWLINE(" " << sps_ptr->scaling_list_data.scaling_list[i][j][k]);
            }
            MSG("");
        }
    }

    MSG("amp_enabled_flag                          = " <<  sps_ptr->amp_enabled_flag);
    MSG("sample_adaptive_offset_enabled_flag       = " <<  sps_ptr->sample_adaptive_offset_enabled_flag);
    MSG("pcm_enabled_flag                          = " <<  sps_ptr->pcm_enabled_flag);
    MSG("pcm_sample_bit_depth_luma_minus1          = " <<  sps_ptr->pcm_sample_bit_depth_luma_minus1);
    MSG("pcm_sample_bit_depth_chroma_minus1        = " <<  sps_ptr->pcm_sample_bit_depth_chroma_minus1);
    MSG("log2_min_pcm_luma_coding_block_size_minus3 = " <<  sps_ptr->log2_min_pcm_luma_coding_block_size_minus3);
    MSG("log2_diff_max_min_pcm_luma_coding_block_size = " <<  sps_ptr->log2_diff_max_min_pcm_luma_coding_block_size);
    MSG("pcm_loop_filter_disabled_flag             = " <<  sps_ptr->pcm_loop_filter_disabled_flag);
    MSG("num_short_term_ref_pic_sets               = " <<  sps_ptr->num_short_term_ref_pic_sets);

    if (sps_ptr->num_short_term_ref_pic_sets) {
        for(int i = 0; i < sps_ptr->num_short_term_ref_pic_sets; i++) {
            PrintStRps(&sps_ptr->st_rps[i]);
        }
    }

    MSG("long_term_ref_pics_present_flag           = " <<  sps_ptr->long_term_ref_pics_present_flag);
    MSG("num_long_term_ref_pics_sps                = " <<  sps_ptr->num_long_term_ref_pics_sps);
    
    if (sps_ptr->num_long_term_ref_pics_sps) {
        MSG("lt_ref_pic_poc_lsb_sps[%u]:  " <<  sps_ptr->num_long_term_ref_pics_sps);
        for(int i = 0; i < sps_ptr->num_long_term_ref_pics_sps; i++) {
            MSG_NO_NEWLINE(" " << sps_ptr->lt_ref_pic_poc_lsb_sps[i]);
        }
        MSG("");
        MSG("used_by_curr_pic_lt_sps_flag[%u]:  " <<  sps_ptr->num_long_term_ref_pics_sps);
        for(int i = 0; i < sps_ptr->num_long_term_ref_pics_sps; i++) {
            MSG_NO_NEWLINE(" " << sps_ptr->used_by_curr_pic_lt_sps_flag[i]);
        }
        MSG("");
    }

    PrintLtRefInfo(&sps_ptr->lt_rps);

    MSG("sps_temporal_mvp_enabled_flag             = " <<  sps_ptr->sps_temporal_mvp_enabled_flag);
    MSG("strong_intra_smoothing_enabled_flag       = " <<  sps_ptr->strong_intra_smoothing_enabled_flag);
    MSG("vui_parameters_present_flag               = " <<  sps_ptr->vui_parameters_present_flag);

    MSG("sps_extension_present_flag                = " <<  sps_ptr->sps_extension_flag);
    MSG("");
}

void HevcVideoParser::PrintPps(HevcPicParamSet *pps_ptr) {
    MSG("=== hevc_picture_parameter_set_t ===");
    MSG("pps_pic_parameter_set_id                    = " <<  pps_ptr->pps_pic_parameter_set_id);
    MSG("pps_seq_parameter_set_id                    = " <<  pps_ptr->pps_seq_parameter_set_id);
    MSG("dependent_slice_segments_enabled_flag       = " <<  pps_ptr->dependent_slice_segments_enabled_flag);
    MSG("output_flag_present_flag                    = " <<  pps_ptr->output_flag_present_flag);
    MSG("num_extra_slice_header_bits                 = " <<  pps_ptr->num_extra_slice_header_bits);
    MSG("sign_data_hiding_enabled_flag               = " <<  pps_ptr->sign_data_hiding_enabled_flag);
    MSG("cabac_init_present_flag                     = " <<  pps_ptr->cabac_init_present_flag);
    MSG("num_ref_idx_l0_default_active_minus1        = " <<  pps_ptr->num_ref_idx_l0_default_active_minus1);
    MSG("num_ref_idx_l1_default_active_minus1        = " <<  pps_ptr->num_ref_idx_l1_default_active_minus1);
    MSG("init_qp_minus26                             = " <<  pps_ptr->init_qp_minus26);
    MSG("constrained_intra_pred_flag                 = " <<  pps_ptr->constrained_intra_pred_flag);
    MSG("transform_skip_enabled_flag                 = " <<  pps_ptr->transform_skip_enabled_flag);
    MSG("cu_qp_delta_enabled_flag                    = " <<  pps_ptr->cu_qp_delta_enabled_flag);
    MSG("diff_cu_qp_delta_depth                      = " <<  pps_ptr->diff_cu_qp_delta_depth);
    MSG("pps_cb_qp_offset                            = " <<  pps_ptr->pps_cb_qp_offset);
    MSG("pps_cr_qp_offset                            = " <<  pps_ptr->pps_cr_qp_offset);
    MSG("pps_slice_chroma_qp_offsets_present_flag    = " <<  pps_ptr->pps_slice_chroma_qp_offsets_present_flag);
    MSG("weighted_pred_flag                          = " <<  pps_ptr->weighted_pred_flag);
    MSG("weighted_bipred_flag                        = " <<  pps_ptr->weighted_bipred_flag);
    MSG("transquant_bypass_enabled_flag              = " <<  pps_ptr->transquant_bypass_enabled_flag);
    MSG("tiles_enabled_flag                          = " <<  pps_ptr->tiles_enabled_flag);
    MSG("entropy_coding_sync_enabled_flag            = " <<  pps_ptr->entropy_coding_sync_enabled_flag);
    MSG("num_tile_columns_minus1                     = " <<  pps_ptr->num_tile_columns_minus1);
    MSG("num_tile_rows_minus1                        = " <<  pps_ptr->num_tile_rows_minus1);
    MSG("uniform_spacing_flag                        = " <<  pps_ptr->uniform_spacing_flag);
    if (!pps_ptr->uniform_spacing_flag) {
        MSG_NO_NEWLINE("column_width_minus1[" << pps_ptr->num_tile_columns_minus1 << "]");
        for (int i = 0; i < pps_ptr->num_tile_columns_minus1; i++) {
            MSG_NO_NEWLINE(" " << pps_ptr->column_width_minus1[i]);
        }
        MSG("");
        MSG_NO_NEWLINE("row_height_minus1[" << pps_ptr->num_tile_rows_minus1 << "]");
        for (int i = 0; i < pps_ptr->num_tile_rows_minus1; i++) {
            MSG_NO_NEWLINE(" " << pps_ptr->row_height_minus1[i]);
        }
        MSG("");
    }
    MSG("loop_filter_across_tiles_enabled_flag       = " <<  pps_ptr->loop_filter_across_tiles_enabled_flag);
    MSG("pps_loop_filter_across_slices_enabled_flag  = " <<  pps_ptr->pps_loop_filter_across_slices_enabled_flag);
    MSG("deblocking_filter_control_present_flag      = " <<  pps_ptr->deblocking_filter_control_present_flag);
    MSG("deblocking_filter_override_enabled_flag     = " <<  pps_ptr->deblocking_filter_override_enabled_flag);
    MSG("pps_deblocking_filter_disabled_flag         = " <<  pps_ptr->pps_deblocking_filter_disabled_flag);
    MSG("pps_beta_offset_div2                        = " <<  pps_ptr->pps_beta_offset_div2);
    MSG("pps_tc_offset_div2                          = " <<  pps_ptr->pps_tc_offset_div2);
    MSG("pps_scaling_list_data_present_flag          = " <<  pps_ptr->pps_scaling_list_data_present_flag);
    MSG("Scaling list:");
    for (int i = 0; i < HEVC_SCALING_LIST_SIZE_NUM; i++) {
        for (int j = 0; j < HEVC_SCALING_LIST_NUM; j++) {
            MSG_NO_NEWLINE("scaling_list[" << i <<"][" << j << "][]:")
            for (int k = 0; k < HEVC_SCALING_LIST_MAX_INDEX; k++) {
                MSG_NO_NEWLINE(" " << pps_ptr->scaling_list_data.scaling_list[i][j][k]);
            }
            MSG("");
        }
    }

    MSG("lists_modification_present_flag             = " <<  pps_ptr->lists_modification_present_flag);
    MSG("log2_parallel_merge_level_minus2            = " <<  pps_ptr->log2_parallel_merge_level_minus2);
    MSG("slice_segment_header_extension_present_flag = " <<  pps_ptr->slice_segment_header_extension_present_flag);
    MSG("pps_extension_present_flag                  = " <<  pps_ptr->pps_extension_present_flag);
    MSG("");
}

void HevcVideoParser::PrintSliceSegHeader(HevcSliceSegHeader *slice_header_ptr) {
    MSG("=== hevc_slice_segment_header_t ===");
    MSG("first_slice_segment_in_pic_flag             = " <<  slice_header_ptr->first_slice_segment_in_pic_flag);
    MSG("no_output_of_prior_pics_flag                = " <<  slice_header_ptr->no_output_of_prior_pics_flag);
    MSG("slice_pic_parameter_set_id                  = " <<  slice_header_ptr->slice_pic_parameter_set_id);
    MSG("dependent_slice_segment_flag                = " <<  slice_header_ptr->dependent_slice_segment_flag);
    MSG("slice_segment_address                       = " <<  slice_header_ptr->slice_segment_address);
    MSG("slice_type                                  = " <<  slice_header_ptr->slice_type);
    MSG("pic_output_flag                             = " <<  slice_header_ptr->pic_output_flag);
    MSG("colour_plane_id                             = " <<  slice_header_ptr->colour_plane_id);
    MSG("slice_pic_order_cnt_lsb                     = " <<  slice_header_ptr->slice_pic_order_cnt_lsb);
    MSG("short_term_ref_pic_set_sps_flag             = " <<  slice_header_ptr->short_term_ref_pic_set_sps_flag);
    MSG("short_term_ref_pic_set_idx                  = " <<  slice_header_ptr->short_term_ref_pic_set_idx);

    PrintStRps(&slice_header_ptr->st_rps);

    MSG("num_long_term_sps                           = " <<  slice_header_ptr->num_long_term_sps);
    MSG("num_long_term_pics                          = " <<  slice_header_ptr->num_long_term_pics);
    MSG_NO_NEWLINE("lt_idx_sps[]:");
    for(int i = 0; i < 32; i++) {
        MSG_NO_NEWLINE(" " << slice_header_ptr->lt_idx_sps[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("poc_lsb_lt[]:");
    for(int i = 0; i < 32; i++) {
        MSG_NO_NEWLINE(" " << slice_header_ptr->poc_lsb_lt[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("used_by_curr_pic_lt_flag[]:");
    for(int i = 0; i < 32; i++) {
        MSG_NO_NEWLINE(" " << slice_header_ptr->used_by_curr_pic_lt_flag[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("delta_poc_msb_present_flag[]:");
    for(int i = 0; i < 32; i++) {
        MSG_NO_NEWLINE(" " << slice_header_ptr->delta_poc_msb_present_flag[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("delta_poc_msb_cycle_lt[]:");
    for(int i = 0; i < 32; i++) {
        MSG_NO_NEWLINE(" " << slice_header_ptr->delta_poc_msb_cycle_lt[i]);
    }
    MSG("");

    PrintLtRefInfo(&slice_header_ptr->lt_rps);

    MSG("slice_temporal_mvp_enabled_flag             = " <<  slice_header_ptr->slice_temporal_mvp_enabled_flag);
    MSG("slice_sao_luma_flag                         = " <<  slice_header_ptr->slice_sao_luma_flag);
    MSG("slice_sao_chroma_flag                       = " <<  slice_header_ptr->slice_sao_chroma_flag);

    MSG("num_ref_idx_active_override_flag            = " <<  slice_header_ptr->num_ref_idx_active_override_flag);
    MSG("num_ref_idx_l0_active_minus1                = " <<  slice_header_ptr->num_ref_idx_l0_active_minus1);
    MSG("num_ref_idx_l1_active_minus1                = " <<  slice_header_ptr->num_ref_idx_l1_active_minus1);
    MSG("ref_pic_list_modification_flag_l0           = " <<  slice_header_ptr->ref_pic_list_modification_flag_l0);
    MSG("ref_pic_list_modification_flag_l1           = " <<  slice_header_ptr->ref_pic_list_modification_flag_l1);
    MSG_NO_NEWLINE("list_entry_l0[]:");
    for(int i = 0; i < 16; i++) {
        MSG_NO_NEWLINE(" " << slice_header_ptr->list_entry_l0[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("list_entry_l1[]:");
    for(int i = 0; i < 16; i++) {
        MSG_NO_NEWLINE(" " << slice_header_ptr->list_entry_l1[i]);
    }
    MSG("");
    MSG("mvd_l1_zero_flag                            = " <<  slice_header_ptr->mvd_l1_zero_flag);
    MSG("cabac_init_flag                             = " <<  slice_header_ptr->cabac_init_flag);
    MSG("collocated_from_l0_flag                     = " <<  slice_header_ptr->collocated_from_l0_flag);
    MSG("collocated_ref_idx                          = " <<  slice_header_ptr->collocated_ref_idx);
    MSG("five_minus_max_num_merge_cand               = " <<  slice_header_ptr->five_minus_max_num_merge_cand);
    MSG("slice_qp_delta                              = " <<  slice_header_ptr->slice_qp_delta);
    MSG("slice_cb_qp_offset                          = " <<  slice_header_ptr->slice_cb_qp_offset);
    MSG("slice_cr_qp_offset                          = " <<  slice_header_ptr->slice_cr_qp_offset);
    MSG("cu_chroma_qp_offset_enabled_flag            = " <<  static_cast<uint32_t>(slice_header_ptr->cu_chroma_qp_offset_enabled_flag));
    MSG("deblocking_filter_override_flag             = " <<  slice_header_ptr->deblocking_filter_override_flag);
    MSG("slice_deblocking_filter_disabled_flag       = " <<  slice_header_ptr->slice_deblocking_filter_disabled_flag);
    MSG("slice_beta_offset_div2                      = " <<  slice_header_ptr->slice_beta_offset_div2);
    MSG("slice_tc_offset_div2                        = " <<  slice_header_ptr->slice_tc_offset_div2);
    MSG("slice_loop_filter_across_slices_enabled_flag = " <<  slice_header_ptr->slice_loop_filter_across_slices_enabled_flag);
    MSG("num_entry_point_offsets                     = " <<  slice_header_ptr->num_entry_point_offsets);
    MSG("offset_len_minus1                           = " <<  slice_header_ptr->offset_len_minus1);
    MSG("slice_segment_header_extension_length       = " <<  slice_header_ptr->slice_segment_header_extension_length);
    MSG("");
}

void HevcVideoParser::PrintStRps(HevcShortTermRps *rps_ptr) {
    MSG("==== Short-term reference picture set =====")
    MSG("inter_ref_pic_set_prediction_flag           = " <<  static_cast<uint32_t>(rps_ptr->inter_ref_pic_set_prediction_flag));
    MSG("delta_idx_minus1                            = " <<  rps_ptr->delta_idx_minus1);
    MSG("delta_rps_sign                              = " <<  static_cast<uint32_t>(rps_ptr->delta_rps_sign));
    MSG("abs_delta_rps_minus1                        = " <<  rps_ptr->abs_delta_rps_minus1);
    MSG_NO_NEWLINE("rps->used_by_curr_pic_flag[]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(rps_ptr->used_by_curr_pic_flag[j]));
    }
    MSG("");
    MSG_NO_NEWLINE("use_delta_flag[]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(rps_ptr->use_delta_flag[j]));
    }
    MSG("");
    MSG("num_negative_pics                           = " <<  rps_ptr->num_negative_pics);
    MSG("num_positive_pics                           = " <<  rps_ptr->num_positive_pics);
    MSG("num_of_delta_pocs                           = " <<  rps_ptr->num_of_delta_pocs);

    MSG_NO_NEWLINE("delta_poc_s0_minus1[]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << rps_ptr->delta_poc_s0_minus1[j]);
    }
    MSG("");
    MSG_NO_NEWLINE("used_by_curr_pic_s0_flag[]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(rps_ptr->used_by_curr_pic_s0_flag[j]));
    }
    MSG("");
    MSG_NO_NEWLINE("delta_poc_s1_minus1[]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << rps_ptr->delta_poc_s1_minus1[j]);
    }
    MSG("");
    MSG_NO_NEWLINE("used_by_curr_pic_s1_flag[]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(rps_ptr->used_by_curr_pic_s1_flag[j]));
    }
    MSG("");

    MSG_NO_NEWLINE("delta_poc_s0[16]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << rps_ptr->delta_poc_s0[j]);
    }
    MSG("");
   MSG_NO_NEWLINE("delta_poc_s1[16]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << rps_ptr->delta_poc_s1[j]);
    }
    MSG("");
    MSG_NO_NEWLINE("used_by_curr_pic_s0[16]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(rps_ptr->used_by_curr_pic_s0[j]));
    }
    MSG("");
    MSG_NO_NEWLINE("used_by_curr_pic_s1[16]:");
    for(int j = 0; j < 16; j++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(rps_ptr->used_by_curr_pic_s1[j]));
    }
    MSG("");
}

void HevcVideoParser::PrintLtRefInfo(HevcLongTermRps *lt_info_ptr) {
    MSG("==== Long-term reference picture info =====");
    MSG("num_of_pics                 = " <<  lt_info_ptr->num_of_pics);
    MSG_NO_NEWLINE("pocs[]:");
    for(int j = 0; j < 32; j++) {
        MSG_NO_NEWLINE(" " << lt_info_ptr->pocs[j]);
    }
    MSG("");
    MSG_NO_NEWLINE("used_by_curr_pic[]:");
    for(int j = 0; j < 32; j++) {
        MSG_NO_NEWLINE(" " << lt_info_ptr->used_by_curr_pic[j]);
    }
    MSG("");
}

void HevcVideoParser::PrintDpb() {
    uint32_t i;

    MSG("=======================");
    MSG("DPB buffer content: ");
    MSG("=======================");
    MSG("dpb_size = " << dpb_buffer_.dpb_size);
    MSG("num_pics_needed_for_output = " << dpb_buffer_.num_pics_needed_for_output);
    MSG("dpb_fullness = " << dpb_buffer_.dpb_fullness);
    MSG("Frame buffer store:");
    for (i = 0; i < HEVC_MAX_DPB_FRAMES; i++) {
        HevcPicInfo *p_buf = &dpb_buffer_.frame_buffer_list[i];
        MSG("Frame buffer " << i << ": pic_idx = " << p_buf->pic_idx << ", dec_buf_idx = " << p_buf->dec_buf_idx << ", pic_order_cnt = " << p_buf->pic_order_cnt << ", slice_pic_order_cnt_lsb = " << p_buf->slice_pic_order_cnt_lsb << ", decode_order_count = " << p_buf->decode_order_count << ", is_reference = " << p_buf->is_reference << ", use_status = " << p_buf->use_status << ", pic_output_flag = " << p_buf->pic_output_flag);
    }
    MSG("");

    MSG("Decode buffer pool:");
    for(i = 0; i < dec_buf_pool_size_; i++) {
        DecodeFrameBuffer *p_dec_buf = &decode_buffer_pool_[i];
        MSG("Decode buffer " << i << ": use_status = " << p_dec_buf->use_status << ", pic_order_cnt = " << p_dec_buf->pic_order_cnt << ", pts = " << p_dec_buf->pts);
    }
    MSG("num_output_pics_ = " << num_output_pics_);
    if (num_output_pics_) {
        MSG("output_pic_list:");
        for (i = 0; i < num_output_pics_; i++) {
            MSG_NO_NEWLINE(output_pic_list_[i] << ", ");
        }
        MSG("");
    }
}

void HevcVideoParser::PrintVappiBufInfo() {
    RocdecHevcPicParams *p_pic_param = &dec_pic_params_.pic_params.hevc;
    MSG("=======================");
    MSG("VAAPI Buffer Info: ");
    MSG("=======================");
    MSG("Current buffer:");
    MSG_NO_NEWLINE("pic_idx = " << p_pic_param->curr_pic.pic_idx << ", poc = " << p_pic_param->curr_pic.poc);
    MSG(", flags = 0x" << std::hex << p_pic_param->curr_pic.flags);
    MSG(std::dec);

    MSG("Reference pictures:");
    for (int i = 0; i < 15; i++) {
        RocdecHevcPicture *p_ref_pic = &p_pic_param->ref_frames[i];
        MSG_NO_NEWLINE("Ref pic " << i << ": " << "pic_idx = " << p_ref_pic->pic_idx << ", poc = " << p_ref_pic->poc);
        MSG(", flags = 0x" << std::hex << p_ref_pic->flags);
        MSG_NO_NEWLINE(std::dec);
    }

    MSG("Slice ref lists:")
    for (int slice_index = 0; slice_index < num_slices_; slice_index++) {
        RocdecHevcSliceParams *p_slice_param = &slice_param_list_[slice_index];
        HevcSliceInfo *p_slice_info = &slice_info_list_[slice_index];
        MSG("Slice " << slice_index << " ref list 0:");
        for (int i = 0; i <= p_slice_info->slice_header.num_ref_idx_l0_active_minus1; i++) {
            MSG("Index " << i << ": " << static_cast<uint32_t>(p_slice_param->ref_pic_list[0][i]));
        }
        if (p_slice_info->slice_header.slice_type == HEVC_SLICE_TYPE_B) {
            MSG("Slice " << slice_index << " ref list 1: ");
            for (int i = 0; i <= p_slice_info->slice_header.num_ref_idx_l1_active_minus1; i++) {
                MSG("Index " << i << ": " << static_cast<uint32_t>(p_slice_param->ref_pic_list[1][i]));
            }
        }
        MSG("");
    }
}
#endif // DBGINFO
