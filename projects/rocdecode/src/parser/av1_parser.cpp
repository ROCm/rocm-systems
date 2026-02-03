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

#include <algorithm>
#include "av1_parser.h"

Av1VideoParser::Av1VideoParser() {
    seen_frame_header_ = 0;
    tile_param_list_.assign(INIT_SLICE_LIST_NUM, {0});
    memset(&curr_pic_, 0, sizeof(Av1Picture));
    memset(&dpb_buffer_, 0, sizeof(DecodedPictureBuffer));
    memset(&seq_header_, 0, sizeof(Av1SequenceHeader));
    memset(&frame_header_, 0, sizeof(Av1FrameHeader));
    memset(&tile_group_data_, 0, sizeof(Av1TileGroupDataInfo));
    InitDpb();
}

Av1VideoParser::~Av1VideoParser() {
}

rocDecStatus Av1VideoParser::Initialize(RocdecParserParams *p_params) {
    rocDecStatus ret;
    if ((ret = RocVideoParser::Initialize(p_params)) != ROCDEC_SUCCESS) {
        return ret;
    }
    // Set display delay to at least DECODE_BUF_POOL_EXTENSION (2) to prevent synchronous submission
    if (parser_params_.max_display_delay < DECODE_BUF_POOL_EXTENSION) {
        parser_params_.max_display_delay = DECODE_BUF_POOL_EXTENSION;
    }
    CheckAndAdjustDecBufPoolSize(BUFFER_POOL_MAX_SIZE);
    return ROCDEC_SUCCESS;
}

rocDecStatus Av1VideoParser::UnInitialize() {
    return ROCDEC_SUCCESS;
}

rocDecStatus Av1VideoParser::ParseVideoData(RocdecSourceDataPacket *p_data) { 
    if (p_data->payload && p_data->payload_size) {
        curr_pts_ = p_data->pts;
        if (ParsePictureData(p_data->payload, p_data->payload_size) != PARSER_OK) {
            logger_.ErrorLog(MakeMsg("Error occurred in picture data parsing."));
            return ROCDEC_RUNTIME_ERROR;
        }
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

ParserResult Av1VideoParser::ParsePictureData(const uint8_t *p_stream, uint32_t pic_data_size) {
    ParserResult ret = PARSER_OK;
    pic_data_buffer_ptr_ = (uint8_t*)p_stream;
    pic_data_size_ = pic_data_size;
    curr_byte_offset_ = 0;

    while (ReadObuHeaderAndSize() != PARSER_EOF) {
        switch (obu_header_.obu_type) {
            case kObuTemporalDelimiter: {
                seen_frame_header_ = 0;
                break;
            }
            case kObuSequenceHeader: {
                ParseSequenceHeaderObu(pic_data_buffer_ptr_ + obu_byte_offset_, obu_size_);
                break;
            }
            case kObuFrameHeader: {
                if (seen_frame_header_ != 0) {
                    logger_.ErrorLog(MakeMsg("If obu_type is equal to OBU_FRAME_HEADER, it is a requirement of bitstream conformance that SeenFrameHeader is equal to 0."));
                    return PARSER_INVALID_ARG;
                }
                int bytes_parsed;
                if ((ret = ParseFrameHeaderObu(pic_data_buffer_ptr_ + obu_byte_offset_, obu_size_, &bytes_parsed)) != PARSER_OK) {
                    return ret;
                }
                break;
            }
            case kObuRedundantFrameHeader: {
                if (seen_frame_header_ != 1) {
                    logger_.ErrorLog(MakeMsg("If obu_type is equal to OBU_REDUNDANT_FRAME_HEADER, it is a requirement of bitstream conformance that SeenFrameHeader is equal to 1."));
                    return PARSER_INVALID_ARG;
                }
                int bytes_parsed;
                if ((ret = ParseFrameHeaderObu(pic_data_buffer_ptr_ + obu_byte_offset_, obu_size_, &bytes_parsed)) != PARSER_OK) {
                    return ret;
                }
                break;
            }
            case kObuFrame: {
                int bytes_parsed;
                if ((ret = ParseFrameHeaderObu(pic_data_buffer_ptr_ + obu_byte_offset_, obu_size_, &bytes_parsed)) != PARSER_OK) {
                    return ret;
                }
                obu_byte_offset_ += bytes_parsed;
                if (obu_size_ > bytes_parsed) {
                    obu_size_ -= bytes_parsed;
                    if (ParseTileGroupObu(pic_data_buffer_ptr_ + obu_byte_offset_, obu_size_) != PARSER_OK) {
                        logger_.ErrorLog(MakeMsg("Error occurred in ParseTileGroupObu(). Skip this OBU."));
                        break;
                    }
                } else {
                    logger_.ErrorLog(MakeMsg("Frame OBU size error."));
                    return PARSER_OUT_OF_RANGE;
                }
                break;
            }
            case kObuTileGroup: {
                if (ParseTileGroupObu(pic_data_buffer_ptr_ + obu_byte_offset_, obu_size_) != PARSER_OK) {
                    logger_.ErrorLog(MakeMsg("Error occurred in ParseTileGroupObu(). Skip this OBU."));
                }
                break;
            }
            default:
                break;
        }

        // Init Roc decoder for the first time or reconfigure the existing decoder
        if (new_seq_activated_) {
            if ((ret = NotifyNewSequence(&seq_header_, &frame_header_)) != PARSER_OK) {
                return ret;
            }
            new_seq_activated_ = false;
        }

        // Submit decode when we have the entire frame data, or display an existing frame
        if (frame_header_.show_existing_frame) {
            int disp_idx = dpb_buffer_.virtual_buffer_index[frame_header_.frame_to_show_map_idx];
            if (disp_idx == INVALID_INDEX) {
                logger_.ErrorLog(MakeMsg("Invalid existing frame index to show."));
                return PARSER_INVALID_ARG;
            }
            if (pfn_display_picture_cb_) {
                if (seq_header_.film_grain_params_present && frame_header_.film_grain_params.apply_grain) {
                    disp_idx = dpb_buffer_.frame_store[disp_idx].fg_buf_idx;
                } else {
                    disp_idx = dpb_buffer_.frame_store[disp_idx].dec_buf_idx;
                }
                decode_buffer_pool_[disp_idx].use_status |= kFrameUsedForDisplay;
                decode_buffer_pool_[disp_idx].pts = curr_pts_;
                // Insert into output/display picture list
                if (num_output_pics_ >= dec_buf_pool_size_) {
                    logger_.ErrorLog(MakeMsg("Display list size larger than decode buffer pool size!"));
                    return PARSER_OUT_OF_RANGE;
                } else {
                    output_pic_list_[num_output_pics_] = disp_idx;
                    num_output_pics_++;
                }
            }
            if ((ret = DecodeFrameWrapup()) != PARSER_OK) {
                return ret;
            }
        } else if (tile_group_data_.num_tiles_parsed && tile_group_data_.num_tiles_parsed == tile_group_data_.num_tiles) {
            if ((ret = FindFreeInDecBufPool()) != PARSER_OK) {
                return ret;
            }
            if ((ret = FindFreeInDpbAndMark()) != PARSER_OK) {
                return ret;
            }
            if ((ret = SendPicForDecode()) != PARSER_OK) {
                logger_.ErrorLog(MakeMsg(STR("Failed to decode!")));
                return ret;
            }
            pic_count_++;
            dpb_buffer_.dec_ref_count[curr_pic_.pic_idx]--;
            memset(&tile_group_data_, 0, sizeof(Av1TileGroupDataInfo));
            if ((ret = DecodeFrameWrapup()) != PARSER_OK) {
                return ret;
            }
            CheckAndUpdateDecStatus();
        }
    };
    return PARSER_OK;
}

ParserResult Av1VideoParser::NotifyNewSequence(Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    video_format_params_.codec = rocDecVideoCodec_AV1;
    video_format_params_.frame_rate.numerator = frame_rate_.numerator;
    video_format_params_.frame_rate.denominator = frame_rate_.denominator;
    video_format_params_.bit_depth_luma_minus8 = p_seq_header->color_config.bit_depth - 8;
    video_format_params_.bit_depth_chroma_minus8 = p_seq_header->color_config.bit_depth - 8;
    video_format_params_.progressive_sequence = 1;
    video_format_params_.min_num_decode_surfaces = dec_buf_pool_size_;
    video_format_params_.coded_width = pic_width_;
    video_format_params_.coded_height = pic_height_;

    // 6.4.2. Color config semantics
    if (p_seq_header->color_config.mono_chrome == 1 && p_seq_header->color_config.subsampling_x == 1 && p_seq_header->color_config.subsampling_y == 1) {
            video_format_params_.chroma_format = rocDecVideoChromaFormat_Monochrome;
    } else if (p_seq_header->color_config.mono_chrome == 0 && p_seq_header->color_config.subsampling_x == 1 && p_seq_header->color_config.subsampling_y == 1) {
        video_format_params_.chroma_format = rocDecVideoChromaFormat_420;
    } else if (p_seq_header->color_config.mono_chrome == 0 && p_seq_header->color_config.subsampling_x == 1 && p_seq_header->color_config.subsampling_y == 0) {
        video_format_params_.chroma_format = rocDecVideoChromaFormat_422;
    } else if (p_seq_header->color_config.mono_chrome == 0 && p_seq_header->color_config.subsampling_x == 0 && p_seq_header->color_config.subsampling_y == 0) {
        video_format_params_.chroma_format = rocDecVideoChromaFormat_444;
    } else {
        logger_.ErrorLog(MakeMsg("Incorrect chroma format."));
        return PARSER_INVALID_FORMAT;
    }

    video_format_params_.display_area.left = 0;
    video_format_params_.display_area.top = 0;
    video_format_params_.display_area.right = p_frame_header->render_size.render_width;
    video_format_params_.display_area.bottom = p_frame_header->render_size.render_height;
    video_format_params_.bitrate = 0;

    // Dispaly aspect ratio
    int disp_width = (video_format_params_.display_area.right - video_format_params_.display_area.left);
    int disp_height = (video_format_params_.display_area.bottom - video_format_params_.display_area.top);
    int gcd = std::__gcd(disp_width, disp_height); // greatest common divisor
    if (gcd) {
        video_format_params_.display_aspect_ratio.x = disp_width / gcd;
        video_format_params_.display_aspect_ratio.y = disp_height / gcd;
    }

    video_format_params_.reconfig_options = ROCDEC_RECONFIG_NEW_SURFACES;
    video_format_params_.video_signal_description = {0};
    video_format_params_.seqhdr_data_length = 0;

    // callback function with RocdecVideoFormat params filled out
    if (pfn_sequence_cb_(parser_params_.user_data, &video_format_params_) == 0) {
        logger_.ErrorLog(MakeMsg("Sequence callback function failed."));
        return PARSER_FAIL;
    } else {
        return PARSER_OK;
    }
    
    return PARSER_OK;
}

ParserResult Av1VideoParser::SendPicForDecode() {
    int i, j;
    Av1SequenceHeader *p_seq_header = &seq_header_;
    Av1FrameHeader *p_frame_header = &frame_header_;
    dec_pic_params_ = {0};

    dec_pic_params_.pic_width = pic_width_;
    dec_pic_params_.pic_height = pic_height_;
    dec_pic_params_.curr_pic_idx = curr_pic_.dec_buf_idx;
    dec_pic_params_.field_pic_flag = 0;
    dec_pic_params_.bottom_field_flag = 0;
    dec_pic_params_.second_field = 0;

    dec_pic_params_.bitstream_data_len = pic_stream_data_size_;
    dec_pic_params_.bitstream_data = pic_stream_data_ptr_;
    dec_pic_params_.num_slices = tile_group_data_.num_tiles;

    dec_pic_params_.ref_pic_flag = 1;
    dec_pic_params_.intra_pic_flag = p_frame_header->frame_is_intra;

    // Set up the picture parameter buffer
    RocdecAv1PicParams *p_pic_param = &dec_pic_params_.pic_params.av1;
    p_pic_param->profile = p_seq_header->seq_profile;
    p_pic_param->order_hint_bits_minus_1 = p_seq_header->order_hint_bits_minus_1;
    p_pic_param->bit_depth_idx = p_seq_header->color_config.bit_depth == 8 ? 0 : (p_seq_header->color_config.bit_depth == 10 ? 1 : 2);
    p_pic_param->matrix_coefficients = p_seq_header->color_config.matrix_coefficients;
    p_pic_param->seq_info_fields.fields.still_picture = p_seq_header->still_picture;
    p_pic_param->seq_info_fields.fields.use_128x128_superblock = p_seq_header->use_128x128_superblock;
    p_pic_param->seq_info_fields.fields.enable_filter_intra = p_seq_header->enable_filter_intra;
    p_pic_param->seq_info_fields.fields.enable_intra_edge_filter = p_seq_header->enable_intra_edge_filter;
    p_pic_param->seq_info_fields.fields.enable_interintra_compound = p_seq_header->enable_interintra_compound;
    p_pic_param->seq_info_fields.fields.enable_masked_compound = p_seq_header->enable_masked_compound;
    p_pic_param->seq_info_fields.fields.enable_dual_filter = p_seq_header->enable_dual_filter;
    p_pic_param->seq_info_fields.fields.enable_order_hint = p_seq_header->enable_order_hint;
    p_pic_param->seq_info_fields.fields.enable_jnt_comp = p_seq_header->enable_jnt_comp;
    p_pic_param->seq_info_fields.fields.enable_cdef = p_seq_header->enable_cdef;
    p_pic_param->seq_info_fields.fields.mono_chrome = p_seq_header->color_config.mono_chrome;
    p_pic_param->seq_info_fields.fields.color_range = p_seq_header->color_config.color_range;
    p_pic_param->seq_info_fields.fields.subsampling_x = p_seq_header->color_config.subsampling_x;
    p_pic_param->seq_info_fields.fields.subsampling_y = p_seq_header->color_config.subsampling_y;
    p_pic_param->seq_info_fields.fields.chroma_sample_position = p_seq_header->color_config.chroma_sample_position;
    p_pic_param->seq_info_fields.fields.film_grain_params_present = p_seq_header->film_grain_params_present;

    p_pic_param->current_frame = curr_pic_.dec_buf_idx;
    p_pic_param->current_display_picture = curr_pic_.fg_buf_idx;
    p_pic_param->anchor_frames_num = 0;
    p_pic_param->anchor_frames_list = nullptr;

    p_pic_param->frame_width_minus1 = p_frame_header->frame_size.frame_width_minus_1;
    p_pic_param->frame_height_minus1 = p_frame_header->frame_size.frame_height_minus_1;
    p_pic_param->output_frame_width_in_tiles_minus_1 = 0; // Todo for large scale tile
    p_pic_param->output_frame_height_in_tiles_minus_1 = 0; // Todo for large scale tile
    
    for (i = 0; i < NUM_REF_FRAMES; i++) {
        if (dpb_buffer_.virtual_buffer_index[i] != INVALID_INDEX) {
            p_pic_param->ref_frame_map[i] = dpb_buffer_.frame_store[dpb_buffer_.virtual_buffer_index[i]].dec_buf_idx;
        } else {
            p_pic_param->ref_frame_map[i] = 0xFF;
        }
    }
    for (i = 0; i < REFS_PER_FRAME; i++) {
        p_pic_param->ref_frame_idx[i] = frame_header_.ref_frame_idx[i];
    }
    p_pic_param->primary_ref_frame = p_frame_header->primary_ref_frame;
    p_pic_param->order_hint = p_frame_header->order_hint;

    p_pic_param->seg_info.segment_info_fields.bits.enabled = p_frame_header->segmentation_params.segmentation_enabled;
    p_pic_param->seg_info.segment_info_fields.bits.update_map = p_frame_header->segmentation_params.segmentation_update_map;
    p_pic_param->seg_info.segment_info_fields.bits.temporal_update = p_frame_header->segmentation_params.segmentation_temporal_update;
    p_pic_param->seg_info.segment_info_fields.bits.update_data = p_frame_header->segmentation_params.segmentation_update_data;
    for (i = 0; i < MAX_SEGMENTS; i++) {
        for (j = 0; j < SEG_LVL_MAX; j++) {
            p_pic_param->seg_info.feature_data[i][j] = p_frame_header->segmentation_params.feature_data[i][j];
            p_pic_param->seg_info.feature_mask[i] |= p_frame_header->segmentation_params.feature_enabled_flags[i][j] << j;
        }
    }

    p_pic_param->film_grain_info.film_grain_info_fields.bits.apply_grain = p_frame_header->film_grain_params.apply_grain;
    p_pic_param->film_grain_info.film_grain_info_fields.bits.chroma_scaling_from_luma = p_frame_header->film_grain_params.chroma_scaling_from_luma;
    p_pic_param->film_grain_info.film_grain_info_fields.bits.grain_scaling_minus_8 = p_frame_header->film_grain_params.grain_scaling_minus_8;
    p_pic_param->film_grain_info.film_grain_info_fields.bits.ar_coeff_lag = p_frame_header->film_grain_params.ar_coeff_lag;
    p_pic_param->film_grain_info.film_grain_info_fields.bits.ar_coeff_shift_minus_6 = p_frame_header->film_grain_params.ar_coeff_shift_minus_6;
    p_pic_param->film_grain_info.film_grain_info_fields.bits.grain_scale_shift = p_frame_header->film_grain_params.grain_scale_shift;
    p_pic_param->film_grain_info.film_grain_info_fields.bits.overlap_flag = p_frame_header->film_grain_params.overlap_flag;
    p_pic_param->film_grain_info.film_grain_info_fields.bits.clip_to_restricted_range = p_frame_header->film_grain_params.clip_to_restricted_range;
    p_pic_param->film_grain_info.grain_seed = p_frame_header->film_grain_params.grain_seed;
    p_pic_param->film_grain_info.num_y_points = p_frame_header->film_grain_params.num_y_points;
    p_pic_param->film_grain_info.num_cb_points = p_frame_header->film_grain_params.num_cb_points;
    p_pic_param->film_grain_info.num_cr_points = p_frame_header->film_grain_params.num_cr_points;
    if (p_frame_header->film_grain_params.apply_grain) {
        for (i = 0; i < p_pic_param->film_grain_info.num_y_points; i++) {
            p_pic_param->film_grain_info.point_y_value[i] = p_frame_header->film_grain_params.point_y_value[i];
            p_pic_param->film_grain_info.point_y_scaling[i] = p_frame_header->film_grain_params.point_y_scaling[i];
        }
        for (i = 0; i < p_pic_param->film_grain_info.num_cb_points; i++) {
            p_pic_param->film_grain_info.point_cb_value[i] = p_frame_header->film_grain_params.point_cb_value[i];
            p_pic_param->film_grain_info.point_cb_scaling[i] = p_frame_header->film_grain_params.point_cb_scaling[i];
        }
        for (i = 0; i < p_pic_param->film_grain_info.num_cr_points; i++) {
            p_pic_param->film_grain_info.point_cr_value[i] = p_frame_header->film_grain_params.point_cr_value[i];
            p_pic_param->film_grain_info.point_cr_scaling[i] = p_frame_header->film_grain_params.point_cr_scaling[i];
        }
        for (i = 0; i < 24; i++) {
            p_pic_param->film_grain_info.ar_coeffs_y[i] = p_frame_header->film_grain_params.ar_coeffs_y_plus_128[i] - 128;
        }
        for (i = 0; i < 25; i++) {
            p_pic_param->film_grain_info.ar_coeffs_cb[i] = p_frame_header->film_grain_params.ar_coeffs_cb_plus_128[i] - 128;
            p_pic_param->film_grain_info.ar_coeffs_cr[i] = p_frame_header->film_grain_params.ar_coeffs_cr_plus_128[i] - 128;
        }
    }
    p_pic_param->film_grain_info.cb_mult = p_frame_header->film_grain_params.cb_mult;
    p_pic_param->film_grain_info.cb_luma_mult = p_frame_header->film_grain_params.cb_luma_mult;
    p_pic_param->film_grain_info.cb_offset = p_frame_header->film_grain_params.cb_offset;
    p_pic_param->film_grain_info.cr_mult = p_frame_header->film_grain_params.cr_mult;
    p_pic_param->film_grain_info.cr_luma_mult = p_frame_header->film_grain_params.cr_luma_mult;
    p_pic_param->film_grain_info.cr_offset = p_frame_header->film_grain_params.cr_offset;

    p_pic_param->tile_cols = p_frame_header->tile_info.tile_cols;
    p_pic_param->tile_rows = p_frame_header->tile_info.tile_rows;
    for (i = 0; i < p_pic_param->tile_cols; i++) {
        p_pic_param->width_in_sbs_minus_1[i] = p_frame_header->tile_info.width_in_sbs_minus_1[i];
    }
    for (i = 0; i < p_pic_param->tile_rows; i++) {
        p_pic_param->height_in_sbs_minus_1[i] = p_frame_header->tile_info.height_in_sbs_minus_1[i];
    }
    p_pic_param->tile_count_minus_1 = 0; // Todo for large scale tile
    p_pic_param->context_update_tile_id = p_frame_header->tile_info.context_update_tile_id;

    p_pic_param->pic_info_fields.bits.frame_type = p_frame_header->frame_type;
    p_pic_param->pic_info_fields.bits.show_frame = p_frame_header->show_frame;
    p_pic_param->pic_info_fields.bits.showable_frame = p_frame_header->showable_frame;
    p_pic_param->pic_info_fields.bits.error_resilient_mode = p_frame_header->error_resilient_mode;
    p_pic_param->pic_info_fields.bits.disable_cdf_update = p_frame_header->disable_cdf_update;
    p_pic_param->pic_info_fields.bits.allow_screen_content_tools = p_frame_header->allow_screen_content_tools;
    p_pic_param->pic_info_fields.bits.force_integer_mv = p_frame_header->force_integer_mv;
    p_pic_param->pic_info_fields.bits.allow_intrabc = p_frame_header->allow_intrabc;
    p_pic_param->pic_info_fields.bits.use_superres = p_frame_header->frame_size.superres_params.use_superres;
    p_pic_param->pic_info_fields.bits.allow_high_precision_mv = p_frame_header->allow_high_precision_mv;
    p_pic_param->pic_info_fields.bits.is_motion_mode_switchable = p_frame_header->is_motion_mode_switchable;
    p_pic_param->pic_info_fields.bits.use_ref_frame_mvs = p_frame_header->use_ref_frame_mvs;
    p_pic_param->pic_info_fields.bits.disable_frame_end_update_cdf = p_frame_header->disable_frame_end_update_cdf;
    p_pic_param->pic_info_fields.bits.uniform_tile_spacing_flag = p_frame_header->tile_info.uniform_tile_spacing_flag;
    p_pic_param->pic_info_fields.bits.allow_warped_motion = p_frame_header->allow_warped_motion;
    p_pic_param->pic_info_fields.bits.large_scale_tile = 0;

    p_pic_param->superres_scale_denominator = p_frame_header->frame_size.superres_params.super_res_denom;
    p_pic_param->interp_filter = p_frame_header->interpolation_filter;
    p_pic_param->filter_level[0] = p_frame_header->loop_filter_params.loop_filter_level[0];
    p_pic_param->filter_level[1] = p_frame_header->loop_filter_params.loop_filter_level[1];
    p_pic_param->filter_level_u = p_frame_header->loop_filter_params.loop_filter_level[2];
    p_pic_param->filter_level_v = p_frame_header->loop_filter_params.loop_filter_level[3];
    p_pic_param->loop_filter_info_fields.bits.sharpness_level = p_frame_header->loop_filter_params.loop_filter_sharpness;
    p_pic_param->loop_filter_info_fields.bits.mode_ref_delta_enabled = p_frame_header->loop_filter_params.loop_filter_delta_enabled;
    p_pic_param->loop_filter_info_fields.bits.mode_ref_delta_update = p_frame_header->loop_filter_params.loop_filter_delta_update;
    for (i = 0; i < TOTAL_REFS_PER_FRAME; i++) {
        p_pic_param->ref_deltas[i] = p_frame_header->loop_filter_params.loop_filter_ref_deltas[i];
    }
    for (i = 0; i < 2; i++) {
        p_pic_param->mode_deltas[i] = p_frame_header->loop_filter_params.loop_filter_mode_deltas[i];
    }

    p_pic_param->base_qindex = p_frame_header->quantization_params.base_q_idx;
    p_pic_param->y_dc_delta_q = p_frame_header->quantization_params.delta_q_y_dc;
    p_pic_param->u_dc_delta_q = p_frame_header->quantization_params.delta_q_u_dc;
    p_pic_param->u_ac_delta_q = p_frame_header->quantization_params.delta_q_u_ac;
    p_pic_param->v_dc_delta_q = p_frame_header->quantization_params.delta_q_v_dc;
    p_pic_param->v_ac_delta_q = p_frame_header->quantization_params.delta_q_v_ac;
    p_pic_param->qmatrix_fields.bits.using_qmatrix = p_frame_header->quantization_params.using_qmatrix;
    p_pic_param->qmatrix_fields.bits.qm_y = p_frame_header->quantization_params.qm_y;
    p_pic_param->qmatrix_fields.bits.qm_u = p_frame_header->quantization_params.qm_u;
    p_pic_param->qmatrix_fields.bits.qm_v = p_frame_header->quantization_params.qm_v;

    p_pic_param->mode_control_fields.bits.delta_q_present_flag = p_frame_header->delta_q_params.delta_q_present;
    p_pic_param->mode_control_fields.bits.log2_delta_q_res = p_frame_header->delta_q_params.delta_q_res;
    p_pic_param->mode_control_fields.bits.delta_lf_present_flag = p_frame_header->delta_lf_params.delta_lf_present;
    p_pic_param->mode_control_fields.bits.log2_delta_lf_res = p_frame_header->delta_lf_params.delta_lf_res;
    p_pic_param->mode_control_fields.bits.delta_lf_multi = p_frame_header->delta_lf_params.delta_lf_multi;
    p_pic_param->mode_control_fields.bits.tx_mode = p_frame_header->tx_mode.tx_mode;
    p_pic_param->mode_control_fields.bits.reference_select = p_frame_header->frame_reference_mode.reference_select;
    p_pic_param->mode_control_fields.bits.reduced_tx_set_used = p_frame_header->reduced_tx_set;
    p_pic_param->mode_control_fields.bits.skip_mode_present = p_frame_header->skip_mode_params.skip_mode_present;

    p_pic_param->cdef_damping_minus_3 = p_frame_header->cdef_params.cdef_damping_minus_3;
    p_pic_param->cdef_bits = p_frame_header->cdef_params.cdef_bits;
    for (int i = 0; i < (1 << p_frame_header->cdef_params.cdef_bits); i++) {
        p_pic_param->cdef_y_strengths[i] = (p_frame_header->cdef_params.cdef_y_pri_strength[i] << 2) | (p_frame_header->cdef_params.cdef_y_sec_strength[i] & 0x03);
        p_pic_param->cdef_uv_strengths[i] = (p_frame_header->cdef_params.cdef_uv_pri_strength[i] << 2) | (p_frame_header->cdef_params.cdef_uv_sec_strength[i] & 0x03);
    }

    p_pic_param->loop_restoration_fields.bits.yframe_restoration_type = p_frame_header->lr_params.frame_restoration_type[0];
    p_pic_param->loop_restoration_fields.bits.cbframe_restoration_type = p_frame_header->lr_params.frame_restoration_type[1];
    p_pic_param->loop_restoration_fields.bits.crframe_restoration_type = p_frame_header->lr_params.frame_restoration_type[2];
    p_pic_param->loop_restoration_fields.bits.lr_unit_shift = p_frame_header->lr_params.lr_unit_shift;
    p_pic_param->loop_restoration_fields.bits.lr_uv_shift = p_frame_header->lr_params.lr_uv_shift;

    for (i = kLastFrame; i <= kAltRefFrame; i++) {
        p_pic_param->wm[i - 1].invalid = p_frame_header->global_motion_params.gm_invalid[i];
        p_pic_param->wm[i - 1].wmtype = static_cast<RocdecAv1TransformationType>(p_frame_header->global_motion_params.gm_type[i]);
        for (int j = 0; j < 6; j++) {
            p_pic_param->wm[i - 1].wmmat[j] = p_frame_header->global_motion_params.gm_params[i][j];
        }
    }

    // Set up tile parameter buffers
    if (tile_group_data_.num_tiles > tile_param_list_.size()) {
        tile_param_list_.resize(tile_group_data_.num_tiles, {0});
    }
    for (i = 0; i < tile_group_data_.num_tiles; i++) {
        RocdecAv1SliceParams *p_tile_param = &tile_param_list_[i];
        Av1TileDataInfo *p_tile_info = &tile_group_data_.tile_data_info[i];
        p_tile_param->slice_data_size = p_tile_info->tile_size;
        p_tile_param->slice_data_offset = p_tile_info->tile_offset;
        p_tile_param->slice_data_flag = 0; // VA_SLICE_DATA_FLAG_ALL;
        p_tile_param->tile_row = p_tile_info->tile_row;
        p_tile_param->tile_column = p_tile_info->tile_col;
        p_tile_param->tg_start = tile_group_data_.tg_start;
        p_tile_param->tg_end = tile_group_data_.tg_end;
        p_tile_param->anchor_frame_idx = 0; // Todo for large scale tile
        p_tile_param->tile_idx_in_tile_list = 0; // Todo large scale tile 
    }
    dec_pic_params_.slice_params.av1 = tile_param_list_.data();

#if DBGINFO
    PrintVaapiParams();
#endif // DBGINFO

    if (pfn_decode_picture_cb_(parser_params_.user_data, &dec_pic_params_) == 0) {
        logger_.ErrorLog(MakeMsg("Decode error occurred."));
        return PARSER_FAIL;
    } else {
        return PARSER_OK;
    }
}

void Av1VideoParser::UpdateRefFrames() {
    for (int i = 0; i < NUM_REF_FRAMES; i++) {
        if ((frame_header_.refresh_frame_flags >> i) & 1) {
            dpb_buffer_.ref_valid[i] = 1;
            dpb_buffer_.ref_frame_id[i] = frame_header_.current_frame_id;
            dpb_buffer_.ref_frame_type[i] = frame_header_.frame_type;
            dpb_buffer_.ref_upscaled_width[i] = frame_header_.frame_size.upscaled_width;
            dpb_buffer_.ref_frame_width[i] = frame_header_.frame_size.frame_width;
            dpb_buffer_.ref_frame_height[i] = frame_header_.frame_size.frame_height;
            dpb_buffer_.ref_render_width[i] = frame_header_.render_size.render_width;
            dpb_buffer_.ref_render_height[i] = frame_header_.render_size.render_height;
            dpb_buffer_.ref_order_hint[i] = frame_header_.order_hint;
            for (int j = 0; j < REFS_PER_FRAME; j++) {
                dpb_buffer_.saved_order_hints[i][j + kLastFrame] = frame_header_.order_hints[j + kLastFrame];
            }
            for (int ref = kLastFrame; ref <= kAltRefFrame; ref++) {
                for (int j = 0; j < 6; j++) {
                    dpb_buffer_.saved_gm_params[i][ref][j] = frame_header_.global_motion_params.gm_params[ref][j];
                }
            }
            for (int j = 0; j < TOTAL_REFS_PER_FRAME; j++) {
                dpb_buffer_.saved_loop_filter_ref_deltas[i][j] = frame_header_.loop_filter_params.loop_filter_ref_deltas[j];
            }
            dpb_buffer_.saved_loop_filter_mode_deltas[i][0] = frame_header_.loop_filter_params.loop_filter_mode_deltas[0];
            dpb_buffer_.saved_loop_filter_mode_deltas[i][1] = frame_header_.loop_filter_params.loop_filter_mode_deltas[1];
            for (int j = 0; j < MAX_SEGMENTS; j++) {
                for (int k = 0; k < SEG_LVL_MAX; k++) {
                    dpb_buffer_.saved_feature_enabled[i][j][k] = frame_header_.segmentation_params.feature_enabled_flags[j][k];
                    dpb_buffer_.saved_feature_data[i][j][k] = frame_header_.segmentation_params.feature_data[j][k];
                }
            }
            if (seq_header_.film_grain_params_present) {
                dpb_buffer_.saved_film_grain_params[i] = frame_header_.film_grain_params; //save_grain_params()
            }

            if (dpb_buffer_.virtual_buffer_index[i] != INVALID_INDEX) {
                dpb_buffer_.dec_ref_count[dpb_buffer_.virtual_buffer_index[i]]--;
            }
            dpb_buffer_.virtual_buffer_index[i] = curr_pic_.pic_idx;
            dpb_buffer_.dec_ref_count[curr_pic_.pic_idx]++;
        }
    }
}

void Av1VideoParser::LoadRefFrame() {
    int ref_idx = frame_header_.frame_to_show_map_idx;
    frame_header_.current_frame_id = dpb_buffer_.ref_frame_id[ref_idx];
    frame_header_.frame_size.upscaled_width = dpb_buffer_.ref_upscaled_width[ref_idx];
    frame_header_.frame_size.frame_width = dpb_buffer_.ref_frame_width[ref_idx];
    frame_header_.frame_size.frame_height = dpb_buffer_.ref_frame_height[ref_idx];
    frame_header_.render_size.render_width = dpb_buffer_.ref_render_width[ref_idx];
    frame_header_.render_size.render_height = dpb_buffer_.ref_render_height[ref_idx];
    frame_header_.order_hint = dpb_buffer_.ref_order_hint[ref_idx];
    for (int j = 0; j < REFS_PER_FRAME; j++) {
        frame_header_.order_hints[j + kLastFrame] = dpb_buffer_.saved_order_hints[ref_idx][j + kLastFrame];
    }
    for (int ref = kLastFrame; ref <= kAltRefFrame; ref++) {
        for (int i = 0; i < 6; i++) {
            frame_header_.global_motion_params.gm_params[ref][i] = dpb_buffer_.saved_gm_params[ref_idx][ref][i];
        }
    }
    if (seq_header_.film_grain_params_present) {
        frame_header_.film_grain_params = dpb_buffer_.saved_film_grain_params[ref_idx];
    }
    for (int i = 0; i < TOTAL_REFS_PER_FRAME; i++) {
        frame_header_.loop_filter_params.loop_filter_ref_deltas[i] = dpb_buffer_.saved_loop_filter_ref_deltas[ref_idx][i];
    }
    frame_header_.loop_filter_params.loop_filter_mode_deltas[0] = dpb_buffer_.saved_loop_filter_mode_deltas[ref_idx][0];
    frame_header_.loop_filter_params.loop_filter_mode_deltas[1] = dpb_buffer_.saved_loop_filter_mode_deltas[ref_idx][1];
    for (int j = 0; j < MAX_SEGMENTS; j++) {
        for (int k = 0; k < SEG_LVL_MAX; k++) {
            frame_header_.segmentation_params.feature_enabled_flags[j][k] = dpb_buffer_.saved_feature_enabled[ref_idx][j][k];
            frame_header_.segmentation_params.feature_data[j][k] = dpb_buffer_.saved_feature_data[ref_idx][j][k];
        }
    }
}

ParserResult Av1VideoParser::DecodeFrameWrapup() {
    ParserResult ret = PARSER_OK;
    if (frame_header_.show_existing_frame) {
        // If the existing frame is key frame, load and update ref frames. Note refresh_frame_flags is set to allFrames (0xFF)
        if ( frame_header_.frame_type == kKeyFrame) {
            LoadRefFrame();
            UpdateRefFrames();
        }
    } else {
        // For show_existing_frame = 0 case, post processing filtering is done in HW
        UpdateRefFrames();
    }
#if DBGINFO
    PrintDpb();
#endif // DBGINFO

    // Output decoded pictures from DPB if any are ready
    if (pfn_display_picture_cb_ && num_output_pics_ > 0) {
        if ((ret = OutputDecodedPictures(false)) != PARSER_OK) {
            return ret;
        }
    }
    memset(&frame_header_, 0, sizeof(Av1FrameHeader));
    return ret;
}

void Av1VideoParser::InitDpb() {
    int i;
    memset(&dpb_buffer_, 0, sizeof(DecodedPictureBuffer));
    for (i = 0; i < BUFFER_POOL_MAX_SIZE; i++) {
        dpb_buffer_.frame_store[i].pic_idx = i;
        dpb_buffer_.frame_store[i].use_status = kNotUsed;
        dpb_buffer_.dec_ref_count[i] = 0;
    }
    for (i = 0; i < NUM_REF_FRAMES; i++) {
        dpb_buffer_.virtual_buffer_index[i] = INVALID_INDEX;
    }
}

ParserResult Av1VideoParser::FlushDpb() {
    if (pfn_display_picture_cb_ && num_output_pics_ > 0) {
        if (OutputDecodedPictures(true) != PARSER_OK) {
            return PARSER_FAIL;
        }
    }
    return PARSER_OK;
}

ParserResult Av1VideoParser::FindFreeInDecBufPool() {
    int dec_buf_index;
    // Find a free buffer in decode/display buffer pool to store the decoded image
    for (dec_buf_index = 0; dec_buf_index < dec_buf_pool_size_; dec_buf_index++) {
        if (decode_buffer_pool_[dec_buf_index].use_status == kNotUsed) {
            break;
        }
    }
    if (dec_buf_index == dec_buf_pool_size_) {
        logger_.ErrorLog(MakeMsg("Could not find a free buffer in decode buffer pool for decoded image."));
        return PARSER_NOT_FOUND;
    }
    curr_pic_.dec_buf_idx = dec_buf_index;
    decode_buffer_pool_[dec_buf_index].use_status |= kFrameUsedForDecode;
    decode_buffer_pool_[dec_buf_index].pic_order_cnt = curr_pic_.order_hint;
    decode_buffer_pool_[dec_buf_index].pts = curr_pts_;
    // Find a free buffer in decode/display buffer pool to store FG output
    if (seq_header_.film_grain_params_present && frame_header_.film_grain_params.apply_grain) {
        for (dec_buf_index = 0; dec_buf_index < dec_buf_pool_size_; dec_buf_index++) {
            if (decode_buffer_pool_[dec_buf_index].use_status == kNotUsed) {
                break;
            }
        }
        if (dec_buf_index == dec_buf_pool_size_) {
            logger_.ErrorLog(MakeMsg("Could not find a free buffer in decode buffer pool for FG output."));
            return PARSER_NOT_FOUND;
        }
        curr_pic_.fg_buf_idx = dec_buf_index;
        decode_buffer_pool_[dec_buf_index].use_status |= kFrameUsedForDisplay;
        decode_buffer_pool_[dec_buf_index].pic_order_cnt = curr_pic_.order_hint;
        decode_buffer_pool_[dec_buf_index].pts = curr_pts_;
    } else {
        curr_pic_.fg_buf_idx = curr_pic_.dec_buf_idx;
    }
    return PARSER_OK;
}

ParserResult Av1VideoParser::FindFreeInDpbAndMark() {
    int i;
    for (i = 0; i < BUFFER_POOL_MAX_SIZE; i++ ) {
        if (dpb_buffer_.dec_ref_count[i] == 0) {
            break;
        }
    }
    if (i == BUFFER_POOL_MAX_SIZE) {
        logger_.ErrorLog(MakeMsg("DPB buffer overflow!"));
        return PARSER_NOT_FOUND;
    }
    curr_pic_.pic_idx = i;
    curr_pic_.use_status = kFrameUsedForDecode;
    dpb_buffer_.frame_store[curr_pic_.pic_idx] = curr_pic_;
    dpb_buffer_.dec_ref_count[curr_pic_.pic_idx]++;
    // Mark as used in decode/display buffer pool
    if (pfn_display_picture_cb_ && curr_pic_.show_frame) {
        int disp_idx = 0xFF;
        if (seq_header_.film_grain_params_present && frame_header_.film_grain_params.apply_grain) {
            disp_idx = curr_pic_.fg_buf_idx;
        } else {
            disp_idx = curr_pic_.dec_buf_idx;
        }
        decode_buffer_pool_[disp_idx].use_status |= kFrameUsedForDisplay;
        decode_buffer_pool_[disp_idx].pts = curr_pts_;
        // Insert into output/display picture list
        if (num_output_pics_ >= dec_buf_pool_size_) {
            logger_.ErrorLog(MakeMsg("Display list size larger than decode buffer pool size!"));
            return PARSER_OUT_OF_RANGE;
        } else {
            output_pic_list_[num_output_pics_] = disp_idx;
            num_output_pics_++;
        }
    }

    return PARSER_OK;
}

void Av1VideoParser::CheckAndUpdateDecStatus() {
    for (int i = 0; i < BUFFER_POOL_MAX_SIZE; i++) {
        if (dpb_buffer_.frame_store[i].use_status != kNotUsed && dpb_buffer_.dec_ref_count[i] == 0) {
            dpb_buffer_.frame_store[i].use_status = kNotUsed;
            decode_buffer_pool_[dpb_buffer_.frame_store[i].dec_buf_idx].use_status &= ~kFrameUsedForDecode;
        }
    }
}

ParserResult Av1VideoParser::ParseObuHeader(const uint8_t *p_stream) {
    size_t offset = 0;
    obu_header_.size = 1;
    if (Parser::GetBit(p_stream, offset) != 0) {
        logger_.ErrorLog(MakeMsg("Syntax error: obu_forbidden_bit must be set to 0."));
        return PARSER_INVALID_ARG;
    }
    obu_header_.obu_type = Parser::ReadBits(p_stream, offset, 4);
    obu_header_.obu_extension_flag = Parser::GetBit(p_stream, offset);
    obu_header_.obu_has_size_field = Parser::GetBit(p_stream, offset);
    if (!obu_header_.obu_has_size_field) {
        logger_.ErrorLog(MakeMsg("Syntax error: Section 5.2: obu_has_size_field must be equal to 1."));
        return PARSER_INVALID_ARG;
    }
    if (Parser::GetBit(p_stream, offset) != 0) {
        logger_.ErrorLog(MakeMsg("Syntax error: obu_reserved_1bit must be set to 0."));
        return PARSER_INVALID_ARG;
    }
    if (obu_header_.obu_extension_flag) {
        obu_header_.size += 1;
        obu_header_.temporal_id = Parser::ReadBits(p_stream, offset, 3);
        obu_header_.spatial_id = Parser::ReadBits(p_stream, offset, 2);
        if (Parser::ReadBits(p_stream, offset, 3) != 0) {
            logger_.ErrorLog(MakeMsg("Syntax error: extension_header_reserved_3bits must be set to 0."));
            return PARSER_INVALID_ARG;
        }
    }
    return PARSER_OK;
}

ParserResult Av1VideoParser::ReadObuHeaderAndSize() {
    if (curr_byte_offset_ >= pic_data_size_) {
        return PARSER_EOF;
    }
    uint8_t *p_stream = pic_data_buffer_ptr_ + curr_byte_offset_;
    if (ParseObuHeader(p_stream) != PARSER_OK) {
        logger_.ErrorLog(MakeMsg("Syntax error(s) found in OBU header."));
    }
    curr_byte_offset_ += obu_header_.size;
    p_stream += obu_header_.size;

    uint32_t bytes_read;
    obu_size_ = ReadLeb128(p_stream, &bytes_read);
    obu_byte_offset_ = curr_byte_offset_ + bytes_read;
    curr_byte_offset_ = obu_byte_offset_ + obu_size_;
    if (curr_byte_offset_ > pic_data_size_) {
        logger_.ErrorLog(MakeMsg("Invalid obu_size value."));
        return PARSER_EOF;
    } else {
        return PARSER_OK;
    }
}

ParserResult Av1VideoParser::ParseSequenceHeaderObu(uint8_t *p_stream, size_t size) {
    Av1SequenceHeader *p_seq_header = &seq_header_;
    size_t offset = 0;  // current bit offset

    memset(p_seq_header, 0, sizeof(Av1SequenceHeader));
    p_seq_header->seq_profile = Parser::ReadBits(p_stream, offset, 3);
    CHECK_ALLOWED_MAX("seq_profile", p_seq_header->seq_profile, 2);
    p_seq_header->still_picture = Parser::GetBit(p_stream, offset);
    p_seq_header->reduced_still_picture_header = Parser::GetBit(p_stream, offset);
    if (p_seq_header->reduced_still_picture_header == 1 && p_seq_header->still_picture != 1) {
        logger_.ErrorLog(MakeMsg("If reduced_still_picture_header is 1, still_picture is required to be 1."));
        return PARSER_WRONG_STATE;
    }

    if (p_seq_header->reduced_still_picture_header) {
        p_seq_header->timing_info_present_flag = 0;
        p_seq_header->decoder_model_info_present_flag = 0;
        p_seq_header->initial_display_delay_present_flag = 0;
        p_seq_header->operating_points_cnt_minus_1 = 0;
        p_seq_header->operating_point_idc[0] = 0;
        p_seq_header->seq_level_idx[0] = Parser::ReadBits(p_stream, offset, 5);
        p_seq_header->seq_tier[0] = 0;
        p_seq_header->decoder_model_present_for_this_op[0] = 0;
        p_seq_header->initial_display_delay_present_for_this_op[0] = 0;
    } else {
        p_seq_header->timing_info_present_flag = Parser::GetBit(p_stream, offset);
        if (p_seq_header->timing_info_present_flag) {
            // timing_info()
            p_seq_header->timing_info.num_units_in_display_tick = Parser::ReadBits(p_stream, offset, 32);
            if (p_seq_header->timing_info.num_units_in_display_tick == 0) {
                logger_.ErrorLog(MakeMsg("num_units_in_display_tick is 0."));
                return PARSER_WRONG_STATE;
            }
            p_seq_header->timing_info.time_scale = Parser::ReadBits(p_stream, offset, 32);
            if (p_seq_header->timing_info.time_scale == 0) {
                logger_.ErrorLog(MakeMsg("time_scale is 0."));
                return PARSER_WRONG_STATE;
            }
            p_seq_header->timing_info.equal_picture_interval = Parser::GetBit(p_stream, offset);
            if (p_seq_header->timing_info.equal_picture_interval) {
                p_seq_header->timing_info.num_ticks_per_picture_minus_1 = ReadUVLC(p_stream, offset);
            }

            p_seq_header->decoder_model_info_present_flag = Parser::GetBit(p_stream, offset);
            if (p_seq_header->decoder_model_info_present_flag) {
                p_seq_header->decoder_model_info.buffer_delay_length_minus_1 = Parser::ReadBits(p_stream, offset, 5);
                p_seq_header->decoder_model_info.num_units_in_decoding_tick = Parser::ReadBits(p_stream, offset, 32);
                p_seq_header->decoder_model_info.buffer_removal_time_length_minus_1 = Parser::ReadBits(p_stream, offset, 5);
                p_seq_header->decoder_model_info.frame_presentation_time_length_minus_1 = Parser::ReadBits(p_stream, offset, 5);
            }
        } else {
            p_seq_header->decoder_model_info_present_flag = 0;
        }

        p_seq_header->initial_display_delay_present_flag = Parser::GetBit(p_stream, offset);
        p_seq_header->operating_points_cnt_minus_1 = Parser::ReadBits(p_stream, offset, 5);
        for (int i = 0; i < p_seq_header->operating_points_cnt_minus_1 + 1; i++) {
            p_seq_header->operating_point_idc[i] = Parser::ReadBits(p_stream, offset, 12);
            if (i > 0) {
                for (int j = 0; j < i - 1; j++) {
                    if (p_seq_header->operating_point_idc[i] == p_seq_header->operating_point_idc[j]) {
                        logger_.ErrorLog(MakeMsg("operating_point_idc[" + TOSTR(i) + "] is equal to operating_point_idc[" + TOSTR(j) + "]"));
                        return PARSER_WRONG_STATE;
                    }
                }
            }
            if (p_seq_header->operating_point_idc[i] && obu_header_.obu_extension_flag != 1) {
                logger_.ErrorLog(MakeMsg("When operating_point_idc is not 0, obu_extension_flag is required to be 1."));
                return PARSER_WRONG_STATE;
            }
            p_seq_header->seq_level_idx[i] = Parser::ReadBits(p_stream, offset, 5);
            if (p_seq_header->seq_level_idx[i] > 7) {
                p_seq_header->seq_tier[i] = Parser::GetBit(p_stream, offset);
            } else {
                p_seq_header->seq_tier[i] = 0;
            }

            if (p_seq_header->decoder_model_info_present_flag) {
                p_seq_header->decoder_model_present_for_this_op[i] = Parser::GetBit(p_stream, offset);
                if (p_seq_header->decoder_model_present_for_this_op[i]) {
                    p_seq_header->operating_parameters_info[i].decoder_buffer_delay = Parser::ReadBits(p_stream, offset, p_seq_header->decoder_model_info.buffer_delay_length_minus_1 + 1);
                    p_seq_header->operating_parameters_info[i].encoder_buffer_delay = Parser::ReadBits(p_stream, offset, p_seq_header->decoder_model_info.buffer_delay_length_minus_1 + 1);
                    p_seq_header->operating_parameters_info[i].low_delay_mode_flag = Parser::GetBit(p_stream, offset);
                }
            } else {
                p_seq_header->decoder_model_present_for_this_op[i] = 0;
            }

            if (p_seq_header->initial_display_delay_present_flag) {
                p_seq_header->initial_display_delay_present_for_this_op[i] = Parser::GetBit(p_stream, offset);
                if (p_seq_header->initial_display_delay_present_for_this_op[i]) {
                    p_seq_header->initial_display_delay_minus_1[i] = Parser::ReadBits(p_stream, offset, 4);
                }
            }
        }
    }

    // Todo: Choose operating point.

    p_seq_header->frame_width_bits_minus_1 = Parser::ReadBits(p_stream, offset, 4);
    p_seq_header->frame_height_bits_minus_1 = Parser::ReadBits(p_stream, offset, 4);
    p_seq_header->max_frame_width_minus_1 = Parser::ReadBits(p_stream, offset, p_seq_header->frame_width_bits_minus_1 + 1);
    p_seq_header->max_frame_height_minus_1 = Parser::ReadBits(p_stream, offset, p_seq_header->frame_height_bits_minus_1 + 1);
    if (p_seq_header->reduced_still_picture_header) {
        p_seq_header->frame_id_numbers_present_flag = 0;
    } else {
        p_seq_header->frame_id_numbers_present_flag = Parser::GetBit(p_stream, offset);
    }
    if (p_seq_header->frame_id_numbers_present_flag) {
        p_seq_header->delta_frame_id_length_minus_2 = Parser::ReadBits(p_stream, offset, 4);
        p_seq_header->additional_frame_id_length_minus_1 = Parser::ReadBits(p_stream, offset, 3);
    }
    p_seq_header->use_128x128_superblock = Parser::GetBit(p_stream, offset);
    p_seq_header->enable_filter_intra = Parser::GetBit(p_stream, offset);
    p_seq_header->enable_intra_edge_filter = Parser::GetBit(p_stream, offset);

    if (p_seq_header->reduced_still_picture_header) {
        p_seq_header->enable_interintra_compound = 0;
        p_seq_header->enable_masked_compound = 0;
        p_seq_header->enable_warped_motion = 0;
        p_seq_header->enable_dual_filter = 0;
        p_seq_header->enable_order_hint = 0;
        p_seq_header->enable_jnt_comp = 0;
        p_seq_header->enable_ref_frame_mvs = 0;
        p_seq_header->seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        p_seq_header->seq_force_integer_mv = SELECT_INTEGER_MV;
        p_seq_header->order_hint_bits = 0;
    } else {
        p_seq_header->enable_interintra_compound = Parser::GetBit(p_stream, offset);
        p_seq_header->enable_masked_compound = Parser::GetBit(p_stream, offset);
        p_seq_header->enable_warped_motion = Parser::GetBit(p_stream, offset);
        p_seq_header->enable_dual_filter = Parser::GetBit(p_stream, offset);
        p_seq_header->enable_order_hint = Parser::GetBit(p_stream, offset);
        if (p_seq_header->enable_order_hint) {
            p_seq_header->enable_jnt_comp = Parser::GetBit(p_stream, offset);
            p_seq_header->enable_ref_frame_mvs = Parser::GetBit(p_stream, offset);
        } else {
            p_seq_header->enable_jnt_comp = 0;
            p_seq_header->enable_ref_frame_mvs = 0;
        }

        p_seq_header->seq_choose_screen_content_tools = Parser::GetBit(p_stream, offset);
        if (p_seq_header->seq_choose_screen_content_tools) {
            p_seq_header->seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        } else {
            p_seq_header->seq_force_screen_content_tools = Parser::GetBit(p_stream, offset);
        }
        if (p_seq_header->seq_force_screen_content_tools > 0) {
            p_seq_header->seq_choose_integer_mv = Parser::GetBit(p_stream, offset);
            if (p_seq_header->seq_choose_integer_mv) {
                p_seq_header->seq_force_integer_mv = SELECT_INTEGER_MV;
            } else {
                p_seq_header->seq_force_integer_mv = Parser::GetBit(p_stream, offset);
            }
        } else {
            p_seq_header->seq_force_integer_mv = SELECT_INTEGER_MV;
        }

        if (p_seq_header->enable_order_hint) {
            p_seq_header->order_hint_bits_minus_1 = Parser::ReadBits(p_stream, offset, 3);
            p_seq_header->order_hint_bits = p_seq_header->order_hint_bits_minus_1 + 1;
        } else {
            p_seq_header->order_hint_bits = 0;
        }
    }

    p_seq_header->enable_superres = Parser::GetBit(p_stream, offset);
    p_seq_header->enable_cdef = Parser::GetBit(p_stream, offset);
    p_seq_header->enable_restoration = Parser::GetBit(p_stream, offset);

    ParseColorConfig(p_stream, offset, p_seq_header);

    p_seq_header->film_grain_params_present = Parser::GetBit(p_stream, offset);
    // Increase decode/display pool size for film grain synthesis output store
    if (p_seq_header->film_grain_params_present) {
        CheckAndAdjustDecBufPoolSize(BUFFER_POOL_MAX_SIZE * 2);
    }
    p_seq_header->is_received = 1;
    return PARSER_OK;
}

ParserResult Av1VideoParser::ParseFrameHeaderObu(uint8_t *p_stream, size_t size, int *p_bytes_parsed) {
    if (seen_frame_header_ == 1) {
        // frame_header_copy(). Use the existing frame_header_obu
    } else {
        seen_frame_header_ = 1;
        ParserResult ret;
        if ((ret = ParseUncompressedHeader(p_stream, size, p_bytes_parsed)) != PARSER_OK) {
            return ret;
        }
        if (frame_header_.show_existing_frame) {
            seen_frame_header_ = 0;
        } else {
            seen_frame_header_ = 1;
        }
    }
    curr_pic_.show_frame = frame_header_.show_frame;
    curr_pic_.current_frame_id = frame_header_.current_frame_id;
    curr_pic_.order_hint = frame_header_.order_hint;
    curr_pic_.frame_type = frame_header_.frame_type;
    return PARSER_OK;
}

ParserResult Av1VideoParser::ParseUncompressedHeader(uint8_t *p_stream, size_t size, int *p_bytes_parsed) {
    ParserResult ret = PARSER_OK;
    size_t offset = 0;  // current bit offset
    Av1SequenceHeader *p_seq_header = &seq_header_;
    Av1FrameHeader *p_frame_header = &frame_header_;
    uint32_t frame_id_len = 0;
    uint32_t all_frames = (1 << NUM_REF_FRAMES) - 1;
    int i;

    p_frame_header->is_received = 0;
    if (p_seq_header->is_received == 0) {
        logger_.ErrorLog(MakeMsg("No valid sequence header received before frame header."));
        return PARSER_WRONG_STATE;
    }

    if (p_seq_header->frame_id_numbers_present_flag) {
        frame_id_len = p_seq_header->additional_frame_id_length_minus_1 + p_seq_header-> delta_frame_id_length_minus_2 + 3;
    }

    if (p_seq_header->reduced_still_picture_header) {
        p_frame_header->show_existing_frame = 0;
        p_frame_header->frame_type = kKeyFrame;
        p_frame_header->frame_is_intra = 1;
        p_frame_header->show_frame = 1;
        p_frame_header->showable_frame = 0;
    } else {
        p_frame_header->show_existing_frame = Parser::GetBit(p_stream, offset);
        if (p_frame_header->show_existing_frame == 1) {
            if (obu_header_.obu_type == kObuFrame) {
                logger_.ErrorLog(MakeMsg("show_existing_frame is 1 when obu_type is equal to OBU_FRAME."));
                return PARSER_WRONG_STATE;
            }
            p_frame_header->frame_to_show_map_idx = Parser::ReadBits(p_stream, offset, 3);
            if (p_seq_header->decoder_model_info_present_flag && !p_seq_header->timing_info.equal_picture_interval) {
                // temporal_point_info()
                p_frame_header->temporal_point_info.frame_presentation_time = Parser::ReadBits(p_stream, offset, p_seq_header->decoder_model_info.frame_presentation_time_length_minus_1 + 1);
            }
            p_frame_header->refresh_frame_flags = 0;
            if (p_seq_header->frame_id_numbers_present_flag) {
                CHECK_ALLOWED_MAX("display_frame_id length", frame_id_len, 16);
                p_frame_header->display_frame_id = Parser::ReadBits(p_stream, offset, frame_id_len);
            }
            p_frame_header->frame_type = dpb_buffer_.ref_frame_type[p_frame_header->frame_to_show_map_idx];
            if (p_frame_header->frame_type == kKeyFrame) {
                p_frame_header->refresh_frame_flags = all_frames;
            }
            if (p_seq_header->film_grain_params_present) {
                p_frame_header->film_grain_params = dpb_buffer_.saved_film_grain_params[p_frame_header->frame_to_show_map_idx]; // load_grain_params();
            }
            return PARSER_OK;
        }

        p_frame_header->frame_type = Parser::ReadBits(p_stream, offset, 2);
        p_frame_header->frame_is_intra = (p_frame_header->frame_type == kIntraOnlyFrame) || (p_frame_header->frame_type == kKeyFrame);
        p_frame_header->show_frame = Parser::GetBit(p_stream, offset);
        if (p_frame_header->show_frame && p_seq_header->decoder_model_info_present_flag && !p_seq_header->timing_info.equal_picture_interval) {
            // temporal_point_info()
            p_frame_header->temporal_point_info.frame_presentation_time = Parser::ReadBits(p_stream, offset, p_seq_header->decoder_model_info.frame_presentation_time_length_minus_1 + 1);
        }
        if (p_frame_header->show_frame) {
            p_frame_header->showable_frame = p_frame_header->frame_type != kKeyFrame;
        } else {
            p_frame_header->showable_frame = Parser::GetBit(p_stream, offset);
        }
        if (p_frame_header->frame_type == kSwitchFrame || (p_frame_header->frame_type == kKeyFrame && p_frame_header->show_frame)) {
            p_frame_header->error_resilient_mode = 1;
        } else {
            p_frame_header->error_resilient_mode = Parser::GetBit(p_stream, offset);
        }
    }

    if (p_frame_header->frame_type == kKeyFrame && p_frame_header->show_frame) {
        for (i = 0; i < NUM_REF_FRAMES; i++) {
            dpb_buffer_.ref_valid[i] = 0;
            dpb_buffer_.ref_order_hint[i] = 0;
        }
        for (i = 0; i < REFS_PER_FRAME; i++) {
            p_frame_header->order_hints[kLastFrame + i] = 0;
        }
    }

    p_frame_header->disable_cdf_update = Parser::GetBit(p_stream, offset);
    if (p_seq_header->seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS) {
        p_frame_header->allow_screen_content_tools = Parser::GetBit(p_stream, offset);
    } else {
        p_frame_header->allow_screen_content_tools = p_seq_header->seq_force_screen_content_tools;
    }

    if (p_frame_header->allow_screen_content_tools) {
        if (p_seq_header->seq_force_integer_mv == SELECT_INTEGER_MV) {
            p_frame_header->force_integer_mv = Parser::GetBit(p_stream, offset);
        } else {
            p_frame_header->force_integer_mv = p_seq_header->seq_force_integer_mv;
        }
    } else {
        p_frame_header->force_integer_mv = 0;
    }
    if (p_frame_header->frame_is_intra) {
        p_frame_header->force_integer_mv = 1;
    }

    if (p_seq_header->frame_id_numbers_present_flag) {
        p_frame_header->prev_frame_id = p_frame_header->current_frame_id;
        p_frame_header->current_frame_id = Parser::ReadBits(p_stream, offset, frame_id_len);
        MarkRefFrames(p_seq_header, p_frame_header, frame_id_len);
    } else {
        p_frame_header->current_frame_id = 0;
    }

    if (p_frame_header->frame_type == kSwitchFrame) {
        p_frame_header->frame_size_override_flag = 1;
    } else if (p_seq_header->reduced_still_picture_header) {
        p_frame_header->frame_size_override_flag = 0;
    } else {
        p_frame_header->frame_size_override_flag = Parser::GetBit(p_stream, offset);
    }

    p_frame_header->order_hint = Parser::ReadBits(p_stream, offset, p_seq_header->order_hint_bits);
    if (p_frame_header->frame_is_intra || p_frame_header->error_resilient_mode) {
        p_frame_header->primary_ref_frame = PRIMARY_REF_NONE;
    } else {
        p_frame_header->primary_ref_frame = Parser::ReadBits(p_stream, offset, 3);
    }

    if (p_seq_header->decoder_model_info_present_flag) {
        p_frame_header->buffer_removal_time_present_flag = Parser::GetBit(p_stream, offset);
        if (p_frame_header->buffer_removal_time_present_flag) {
            for (int op_num = 0; op_num <= p_seq_header->operating_points_cnt_minus_1; op_num++) {
                if (p_seq_header->decoder_model_present_for_this_op[op_num]) {
                    uint32_t op_pt_idc = p_seq_header->operating_point_idc[op_num];
                    uint32_t in_temporal_layer = (op_pt_idc >> temporal_id_) & 1;
                    uint32_t in_spatial_layer = (op_pt_idc >> (spatial_id_ + 8)) & 1;
                    if (op_pt_idc == 0 || (in_temporal_layer && in_spatial_layer)) {
                        p_frame_header->buffer_removal_time[op_num] = Parser::ReadBits(p_stream, offset, p_seq_header->decoder_model_info.buffer_removal_time_length_minus_1 + 1);
                    }
                }
            }
        }
    }

    p_frame_header->allow_high_precision_mv = 0;
    p_frame_header->use_ref_frame_mvs = 0;
    p_frame_header->allow_intrabc = 0;

    if (p_frame_header->frame_type == kSwitchFrame || (p_frame_header->frame_type == kKeyFrame && p_frame_header->show_frame)) {
        p_frame_header->refresh_frame_flags = all_frames;
    } else {
        p_frame_header->refresh_frame_flags = Parser::ReadBits(p_stream, offset, 8);
        if (p_frame_header->frame_type == kIntraOnlyFrame && p_frame_header->refresh_frame_flags == 0xFF) {
            logger_.ErrorLog(MakeMsg("When frame_type is equal to INTRA_ONLY_FRAME, it is a requirement of bitstream conformance that refresh_frame_flags is not equal to 0xff."));
            return PARSER_WRONG_STATE;
        }
    }
    if (!p_frame_header->frame_is_intra || p_frame_header->refresh_frame_flags != all_frames) {
        if (p_frame_header->error_resilient_mode && p_seq_header->enable_order_hint) {
            for (i = 0; i < NUM_REF_FRAMES; i++) {
                p_frame_header->ref_order_hint[i] = Parser::ReadBits(p_stream, offset, p_seq_header->order_hint_bits);
                if (p_frame_header->ref_order_hint[i] != dpb_buffer_.ref_order_hint[i]) {
                    dpb_buffer_.ref_valid[i] = 0;
                }
            }
        }
    }

    if (p_frame_header->frame_is_intra) {
        if ((ret = FrameSize(p_stream, offset, p_seq_header, p_frame_header)) != PARSER_OK) {
            return ret;
        }
        RenderSize(p_stream, offset, p_frame_header);
        if (p_frame_header->allow_screen_content_tools && p_frame_header->frame_size.upscaled_width == p_frame_header->frame_size.frame_width) {
            p_frame_header->allow_intrabc = Parser::GetBit(p_stream, offset);
        }
    } else {
        if (!p_seq_header->enable_order_hint) {
            p_frame_header->frame_refs_short_signaling = 0;
        } else {
            p_frame_header->frame_refs_short_signaling = Parser::GetBit(p_stream, offset);
            if (p_frame_header->frame_refs_short_signaling) {
                p_frame_header->last_frame_idx = Parser::ReadBits(p_stream, offset, 3);
                p_frame_header->gold_frame_idx = Parser::ReadBits(p_stream, offset, 3);
                // 7.8. Set frame refs process
                SetFrameRefs(p_seq_header, p_frame_header);
            }
        }

        for (int i = 0; i < REFS_PER_FRAME; i++) {
            if (!p_frame_header->frame_refs_short_signaling) {
                p_frame_header->ref_frame_idx[i] = Parser::ReadBits(p_stream, offset, 3);
            }
            if (p_seq_header->frame_id_numbers_present_flag) {
                p_frame_header->delta_frame_id_minus_1 = Parser::ReadBits(p_stream, offset, p_seq_header->delta_frame_id_length_minus_2 + 2);
                uint32_t delta_frame_id = p_frame_header->delta_frame_id_minus_1 + 1;
                p_frame_header->expected_frame_id[i] = ((p_frame_header->current_frame_id + (1 << frame_id_len) - delta_frame_id ) % (1 << frame_id_len));
                if (p_frame_header->expected_frame_id[i] != dpb_buffer_.ref_frame_id[p_frame_header->ref_frame_idx[i]] || dpb_buffer_.ref_valid[p_frame_header->ref_frame_idx[i]] == 0) {
                    logger_.ErrorLog(MakeMsg("Syntax Error: Reference buffer frame ID mismatch.\n"));
                    return PARSER_INVALID_ARG;
                }
            }
        }

        if (p_frame_header->frame_size_override_flag && !p_frame_header->error_resilient_mode) {
            FrameSizeWithRefs(p_stream, offset, p_seq_header, p_frame_header);
        } else {
            if ((ret = FrameSize(p_stream, offset, p_seq_header, p_frame_header)) != PARSER_OK) {
                return ret;
            }
            RenderSize(p_stream, offset, p_frame_header);
        }

        if (p_frame_header->force_integer_mv) {
            p_frame_header->allow_high_precision_mv = 0;
        } else {
            p_frame_header->allow_high_precision_mv = Parser::GetBit(p_stream, offset);
        }

        // read_interpolation_filter()
        p_frame_header->is_filter_switchable = Parser::GetBit(p_stream, offset);
        if (p_frame_header->is_filter_switchable == 1) {
            p_frame_header->interpolation_filter = kSwitchable;
        } else {
            p_frame_header->interpolation_filter = Parser::ReadBits(p_stream, offset, 2);
        }
        p_frame_header->is_motion_mode_switchable = Parser::GetBit(p_stream, offset);
        if (p_frame_header->error_resilient_mode || !p_seq_header->enable_ref_frame_mvs) {
            p_frame_header->use_ref_frame_mvs = 0;
        } else {
            p_frame_header->use_ref_frame_mvs = Parser::GetBit(p_stream, offset);
        }

        for (i = 0; i < REFS_PER_FRAME; i++) {
            uint32_t ref_frame = kLastFrame + i;
            uint32_t hint = dpb_buffer_.ref_order_hint[p_frame_header->ref_frame_idx[i]];
            p_frame_header->order_hints[ref_frame] = hint;
            if (!p_seq_header->enable_order_hint) {
                p_frame_header->ref_frame_sign_bias[ref_frame] = 0;
            } else {
                p_frame_header->ref_frame_sign_bias[ref_frame] = GetRelativeDist(p_seq_header, hint, p_frame_header->order_hint) > 0;
            }
        }
    }

    if ( pic_width_ != p_frame_header->frame_size.frame_width || pic_height_ != p_frame_header->frame_size.frame_height) {
        pic_width_ = p_frame_header->frame_size.upscaled_width;
        pic_height_ = p_frame_header->frame_size.frame_height;
        new_seq_activated_ = true;
    }

    if (p_seq_header->reduced_still_picture_header || p_frame_header->disable_cdf_update) {
        p_frame_header->disable_frame_end_update_cdf = 1;
    } else {
        p_frame_header->disable_frame_end_update_cdf = Parser::GetBit(p_stream, offset);
    }

    if (p_frame_header->primary_ref_frame == PRIMARY_REF_NONE) {
        // Todo: check need for implementation
        //init_non_coeff_cdfs();
        SetupPastIndependence(p_frame_header);
    } else {
        // Todo: check need for implementation
        //load_cdfs();
        LoadPrevious(p_frame_header);
    }

    if (p_frame_header->use_ref_frame_mvs == 1) {
        // Todo: check need for implementation
        //motion_field_estimation());
    }

    if ((ret = TileInfo(p_stream, offset, p_seq_header, p_frame_header)) != PARSER_OK) {
        return ret;
    }
    QuantizationParams(p_stream, offset, p_seq_header, p_frame_header);
    SegmentationParams(p_stream, offset, p_frame_header);
    DeltaQParams(p_stream, offset, p_frame_header);
    DeltaLFParams(p_stream, offset, p_frame_header);

    if (p_frame_header->primary_ref_frame == PRIMARY_REF_NONE) {
        // Todo: check need for implementation
        // init_coeff_cdfs();
    } else {
        // Todo: check need for implementation
        //load_previous_segment_ids();
    }

    p_frame_header->coded_lossless = 1;
    for (int segment_id = 0; segment_id < MAX_SEGMENTS; segment_id++) {
        int qindex = GetQIndex(p_frame_header, 1, segment_id);
        p_frame_header->lossless_array[segment_id] = qindex == 0 && p_frame_header->quantization_params.delta_q_y_dc == 0 && p_frame_header->quantization_params.delta_q_u_ac == 0 && p_frame_header->quantization_params.delta_q_u_dc == 0 && p_frame_header->quantization_params.delta_q_v_ac == 0 && p_frame_header->quantization_params.delta_q_v_dc == 0;
        if (!p_frame_header->lossless_array[segment_id]) {
            p_frame_header->coded_lossless = 0;
        }
        if (p_frame_header->quantization_params.using_qmatrix) {
            if (p_frame_header->lossless_array[segment_id]) {
                p_frame_header->seg_qm_level[0][segment_id] = 15;
                p_frame_header->seg_qm_level[1][segment_id] = 15;
                p_frame_header->seg_qm_level[2][segment_id] = 15;
            } else {
                p_frame_header->seg_qm_level[0][segment_id] = p_frame_header->quantization_params.qm_y;
                p_frame_header->seg_qm_level[1][segment_id] = p_frame_header->quantization_params.qm_u;
                p_frame_header->seg_qm_level[2][segment_id] = p_frame_header->quantization_params.qm_v;
            }
        }
    }
    if (p_frame_header->coded_lossless == 1 && p_frame_header->delta_q_params.delta_q_present == 1) {
        logger_.ErrorLog(MakeMsg("It is a requirement of bitstream conformance that delta_q_present is equal to 0 when CodedLossless is equal to 1."));
        return PARSER_WRONG_STATE;
    }

    p_frame_header->all_lossless = p_frame_header->coded_lossless && (p_frame_header->frame_size.frame_width == p_frame_header->frame_size.upscaled_width);

    LoopFilterParams(p_stream, offset, p_seq_header, p_frame_header);
    CdefParams(p_stream, offset, p_seq_header, p_frame_header);
    LrParams(p_stream, offset, p_seq_header, p_frame_header);
    ReadTxMode(p_stream, offset, p_frame_header);

    // frame_reference_mode()
    if (p_frame_header->frame_is_intra) {
        p_frame_header->frame_reference_mode.reference_select = 0;
    } else {
        p_frame_header->frame_reference_mode.reference_select = Parser::GetBit(p_stream, offset);
    }

    SkipModeParams(p_stream, offset, p_seq_header, p_frame_header);

    if (p_frame_header->frame_is_intra || p_frame_header->error_resilient_mode || !p_seq_header->enable_warped_motion) {
        p_frame_header->allow_warped_motion = 0;
    } else {
        p_frame_header->allow_warped_motion = Parser::GetBit(p_stream, offset);
    }

    p_frame_header->reduced_tx_set = Parser::GetBit(p_stream, offset);

    GlobalMotionParams(p_stream, offset, p_frame_header);
    if ((ret = FilmGrainParams(p_stream, offset, p_seq_header, p_frame_header)) != PARSER_OK) {
        return ret;
    }

    *p_bytes_parsed = (offset + 7) >> 3;
    p_frame_header->is_received = 1;
    return PARSER_OK;
}

ParserResult Av1VideoParser::ParseTileGroupObu(uint8_t *p_stream, size_t size) {
    size_t offset = 0;  // current bit offset
    Av1FrameHeader *p_frame_header = &frame_header_;
    Av1TileGroupDataInfo *p_tile_group = &tile_group_data_;
    uint32_t tile_start_and_end_present_flag = 0;
    uint32_t header_bytes = 0;
    uint32_t tile_cols = p_frame_header->tile_info.tile_cols;
    uint32_t tile_rows = p_frame_header->tile_info.tile_rows;
    uint8_t *p_tg_buf = p_stream;
    uint32_t tg_size = size;

    if (p_frame_header->is_received == 0) {
        logger_.ErrorLog(MakeMsg("No valid frame header received before tile group."));
        return PARSER_WRONG_STATE;
    }

    if (p_tile_group->tile_group_num == 0) {
        p_tile_group->buffer_ptr = p_stream;
    }

    // First parse the header
    p_tile_group->num_tiles = tile_cols * tile_rows;
    if (p_tile_group->num_tiles > 1) {
        tile_start_and_end_present_flag = Parser::GetBit(p_stream, offset);
        if (obu_header_.obu_type == kObuFrame && tile_start_and_end_present_flag != 0) {
            logger_.ErrorLog(MakeMsg("If obu_type is equal to OBU_FRAME, it is a requirement of bitstream conformance that the value of tile_start_and_end_present_flag is equal to 0."));
            return PARSER_WRONG_STATE;
        }
    }
    if (p_tile_group->num_tiles == 1 || !tile_start_and_end_present_flag) {
        p_tile_group->tg_start = 0;
        p_tile_group->tg_end = p_tile_group->num_tiles - 1;
    } else {
        uint32_t tile_bits = p_frame_header->tile_info.tile_cols_log2 + p_frame_header->tile_info.tile_rows_log2;
        p_tile_group->tg_start = Parser::ReadBits(p_stream, offset, tile_bits);
        if (p_tile_group->tg_start != p_tile_group->num_tiles_parsed) {
            logger_.ErrorLog(MakeMsg("It is a requirement of bitstream conformance that the value of tg_start (" + TOSTR(p_tile_group->tg_start) + ") is equal to the value of TileNum (" + TOSTR(p_tile_group->num_tiles_parsed) + ") at the point that tile_group_obu is invoked."));
            return PARSER_WRONG_STATE;
        }
        p_tile_group->tg_end = Parser::ReadBits(p_stream, offset, tile_bits);
        if (p_tile_group->tg_end < p_tile_group->tg_start) {
            logger_.ErrorLog(MakeMsg("It is a requirement of bitstream conformance that the value of tg_end (" + TOSTR(p_tile_group->tg_end) + ") is greater than or equal to tg_start (" + TOSTR(p_tile_group->tg_start) + ")."));
            return PARSER_WRONG_STATE;
        }
    }

    header_bytes = ((offset + 7) >> 3);
    p_tg_buf += header_bytes;
    tg_size -= header_bytes;
    for (int tile_num = p_tile_group->tg_start; tile_num <= p_tile_group->tg_end; tile_num++) {
        if (tile_cols == 0) {
            logger_.ErrorLog(MakeMsg("Tile columns is 0."));
            return PARSER_WRONG_STATE;
        }
        p_tile_group->tile_data_info[tile_num].tile_row = tile_num / tile_cols;
        p_tile_group->tile_data_info[tile_num].tile_col = tile_num % tile_cols;
        int last_tile = (tile_num == p_tile_group->tg_end);
        if (last_tile) {
            p_tile_group->tile_data_info[tile_num].tile_size = tg_size;
            p_tile_group->tile_data_info[tile_num].tile_offset = p_tg_buf - p_tile_group->buffer_ptr;
        } else {
            uint32_t tile_size_bytes = p_frame_header->tile_info.tile_size_bytes_minus_1 + 1;
            uint32_t tile_size = ReadLeBytes(p_tg_buf, tile_size_bytes) + 1;
            CHECK_ALLOWED_MAX("Tile size", tile_size, tg_size);
            p_tile_group->tile_data_info[tile_num].tile_size = tile_size;
            p_tile_group->tile_data_info[tile_num].tile_offset = p_tg_buf + tile_size_bytes - p_tile_group->buffer_ptr;
            tg_size -= tile_size + tile_size_bytes;
            p_tg_buf += tile_size + tile_size_bytes;
        }
        p_tile_group->num_tiles_parsed++;
    }
    p_tile_group->tile_group_num++;
    if (p_tile_group->tg_end == p_tile_group->num_tiles - 1) {
        p_tile_group->buffer_size = p_tile_group->tile_data_info[p_tile_group->tg_end].tile_offset + p_tile_group->tile_data_info[p_tile_group->tg_end].tile_size;
        p_tile_group->tg_start = 0;
        pic_stream_data_ptr_ = p_tile_group->buffer_ptr;
        pic_stream_data_size_ = p_tile_group->buffer_size;
        if (!frame_header_.disable_frame_end_update_cdf) {
            //frame_end_update_cdf();
        }
        seen_frame_header_ = 0;
    }
    return PARSER_OK;
}

void Av1VideoParser::ParseColorConfig(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header) {
    p_seq_header->color_config.bit_depth = 8;
    
    p_seq_header->color_config.high_bitdepth = Parser::GetBit(p_stream, offset);
    if (p_seq_header->seq_profile == 2 && p_seq_header->color_config.high_bitdepth) {
        p_seq_header->color_config.twelve_bit = Parser::GetBit(p_stream, offset);
        p_seq_header->color_config.bit_depth = p_seq_header->color_config.twelve_bit ? 12 : 10;
    } else if (p_seq_header->seq_profile <= 2) {
        p_seq_header->color_config.bit_depth = p_seq_header->color_config.high_bitdepth ? 10 : 8;
    }

    if (p_seq_header->seq_profile == 1) {
        p_seq_header->color_config.mono_chrome = 0;
    } else {
        p_seq_header->color_config.mono_chrome = Parser::GetBit(p_stream, offset);
    }
    p_seq_header->color_config.num_planes = p_seq_header->color_config.mono_chrome ? 1 : 3;

    p_seq_header->color_config.color_description_present_flag = Parser::GetBit(p_stream, offset);
    if (p_seq_header->color_config.color_description_present_flag) {
        p_seq_header->color_config.color_primaries = Parser::ReadBits(p_stream, offset, 8);
        p_seq_header->color_config.transfer_characteristics = Parser::ReadBits(p_stream, offset, 8);
        p_seq_header->color_config.matrix_coefficients = Parser::ReadBits(p_stream, offset, 8);
    } else {
        p_seq_header->color_config.color_primaries = CP_UNSPECIFIED;
        p_seq_header->color_config.transfer_characteristics = TC_UNSPECIFIED;
        p_seq_header->color_config.matrix_coefficients = MC_UNSPECIFIED;
    }

    if (p_seq_header->color_config.mono_chrome) {
        p_seq_header->color_config.color_range = Parser::GetBit(p_stream, offset);
        p_seq_header->color_config.subsampling_x = 1;
        p_seq_header->color_config.subsampling_y = 1;
        p_seq_header->color_config.chroma_sample_position = CSP_UNKNOWN;
        p_seq_header->color_config.separate_uv_delta_q = 0;
        return;
    } else if (p_seq_header->color_config.color_primaries == CP_BT_709 && p_seq_header->color_config.transfer_characteristics == TC_SRGB && p_seq_header->color_config.matrix_coefficients == MC_IDENTITY) {
        p_seq_header->color_config.color_range = 1;
        p_seq_header->color_config.subsampling_x = 0;
        p_seq_header->color_config.subsampling_y = 0;
    } else {
        p_seq_header->color_config.color_range = Parser::GetBit(p_stream, offset);
        if (p_seq_header->seq_profile == 0) {
            p_seq_header->color_config.subsampling_x = 1;
            p_seq_header->color_config.subsampling_y = 1;
        } else if (p_seq_header->seq_profile == 1) {
            p_seq_header->color_config.subsampling_x = 0;
            p_seq_header->color_config.subsampling_y = 0;
        } else {
            if (p_seq_header->color_config.bit_depth == 12) {
                p_seq_header->color_config.subsampling_x = Parser::GetBit(p_stream, offset);
                if (p_seq_header->color_config.subsampling_x) {
                    p_seq_header->color_config.subsampling_y = Parser::GetBit(p_stream, offset);
                } else {
                    p_seq_header->color_config.subsampling_y = 0;
                }
            } else {
                p_seq_header->color_config.subsampling_x = 1;
                p_seq_header->color_config.subsampling_y = 0;
            }
        }
        if (p_seq_header->color_config.subsampling_x && p_seq_header->color_config.subsampling_y) {
            p_seq_header->color_config.chroma_sample_position = Parser::ReadBits(p_stream, offset, 2);
        }
    }
    p_seq_header->color_config.separate_uv_delta_q = Parser::GetBit(p_stream, offset);
}

void Av1VideoParser::MarkRefFrames(Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header, uint32_t id_len) {
    int diff_len = p_seq_header->delta_frame_id_length_minus_2 + 2;
    for (int i = 0; i < NUM_REF_FRAMES; i++) {
        if (p_frame_header->frame_type == kKeyFrame && p_frame_header->show_frame) {
            dpb_buffer_.ref_valid[i] = 0;
        } else if (p_frame_header->current_frame_id > (1 << diff_len)) {
            if (dpb_buffer_.ref_frame_id[i] > p_frame_header->current_frame_id || dpb_buffer_.ref_frame_id[i] < (p_frame_header->current_frame_id - (1 << diff_len))) {
                dpb_buffer_.ref_valid[i] = 0;
            }
        } else {
            if (dpb_buffer_.ref_frame_id[i] > p_frame_header->current_frame_id && dpb_buffer_.ref_frame_id[i] < ((1 << id_len) + p_frame_header->current_frame_id - (1 << diff_len))) {
                dpb_buffer_.ref_valid[ i ] = 0;
            }
        }
    }
}

ParserResult Av1VideoParser::FrameSize(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    if (p_frame_header->frame_size_override_flag) {
        p_frame_header->frame_size.frame_width_minus_1 = Parser::ReadBits(p_stream, offset, p_seq_header->frame_width_bits_minus_1 + 1);
        CHECK_ALLOWED_MAX("frame_width_minus_1", p_frame_header->frame_size.frame_width_minus_1, p_seq_header->max_frame_width_minus_1);
        p_frame_header->frame_size.frame_width = p_frame_header->frame_size.frame_width_minus_1 + 1;
        p_frame_header->frame_size.frame_height_minus_1 = Parser::ReadBits(p_stream, offset, p_seq_header->frame_height_bits_minus_1 + 1);
        CHECK_ALLOWED_MAX("frame_height_minus_1", p_frame_header->frame_size.frame_height_minus_1, p_seq_header->max_frame_height_minus_1);
        p_frame_header->frame_size.frame_height = p_frame_header->frame_size.frame_height_minus_1 + 1;
    } else {
        p_frame_header->frame_size.frame_width_minus_1 = p_seq_header->max_frame_width_minus_1;
        p_frame_header->frame_size.frame_height_minus_1 = p_seq_header->max_frame_height_minus_1;
        p_frame_header->frame_size.frame_width = p_seq_header->max_frame_width_minus_1 + 1;
        p_frame_header->frame_size.frame_height = p_seq_header->max_frame_height_minus_1 + 1;
    }
    SuperResParams(p_stream, offset, p_seq_header, p_frame_header);
    ComputeImageSize(p_frame_header);
    return PARSER_OK;
}

void Av1VideoParser::SuperResParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    if (p_seq_header->enable_superres) {
        p_frame_header->frame_size.superres_params.use_superres = Parser::GetBit(p_stream, offset);
    } else {
        p_frame_header->frame_size.superres_params.use_superres = 0;
    }
    if (p_frame_header->frame_size.superres_params.use_superres) {
        p_frame_header->frame_size.superres_params.coded_denom = Parser::ReadBits(p_stream, offset, SUPERRES_DENOM_BITS);
        p_frame_header->frame_size.superres_params.super_res_denom = p_frame_header->frame_size.superres_params.coded_denom + SUPERRES_DENOM_MIN;
    } else {
        p_frame_header->frame_size.superres_params.super_res_denom = SUPERRES_NUM;
    }
    p_frame_header->frame_size.upscaled_width = p_frame_header->frame_size.frame_width;
    p_frame_header->frame_size.frame_width = (p_frame_header->frame_size.upscaled_width * SUPERRES_NUM + (p_frame_header->frame_size.superres_params.super_res_denom / 2)) / p_frame_header->frame_size.superres_params.super_res_denom;
}

void Av1VideoParser::ComputeImageSize(Av1FrameHeader *p_frame_header) {
    p_frame_header->frame_size.mi_cols = 2 * ((p_frame_header->frame_size.frame_width + 7) >> 3);
    p_frame_header->frame_size.mi_rows = 2 * ((p_frame_header->frame_size.frame_height + 7) >> 3);
}

void Av1VideoParser::RenderSize(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header) {
    p_frame_header->render_size.render_and_frame_size_different = Parser::GetBit(p_stream, offset);
    if (p_frame_header->render_size.render_and_frame_size_different) {
        p_frame_header->render_size.render_width_minus_1 = Parser::ReadBits(p_stream, offset, 16);
        p_frame_header->render_size.render_height_minus_1 = Parser::ReadBits(p_stream, offset, 16);
        p_frame_header->render_size.render_width = p_frame_header->render_size.render_width_minus_1 + 1;
        p_frame_header->render_size.render_height = p_frame_header->render_size.render_height_minus_1 + 1;
    } else {
        p_frame_header->render_size.render_width = p_frame_header->frame_size.upscaled_width;
        p_frame_header->render_size.render_height = p_frame_header->frame_size.frame_height;
    }
}

void Av1VideoParser::SetFrameRefs(Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    int i;
    int used_frame[NUM_REF_FRAMES];
    int curr_frame_hint;
    int shifted_order_hints[NUM_REF_FRAMES];
    int latest_order_hint, earliest_order_hint;
    int ref;

    for (i = 0; i < REFS_PER_FRAME; i++) {
        p_frame_header->ref_frame_idx[i] = INVALID_INDEX;
    }
    p_frame_header->ref_frame_idx[kLastFrame - kLastFrame] = p_frame_header->last_frame_idx;
    p_frame_header->ref_frame_idx[kGoldenFrame - kLastFrame] = p_frame_header->gold_frame_idx;
    for (i = 0; i < NUM_REF_FRAMES; i++) {
        used_frame[ i ] = 0;
    }
    used_frame[p_frame_header->last_frame_idx] = 1;
    used_frame[p_frame_header->gold_frame_idx ] = 1;

    curr_frame_hint = 1 << (p_seq_header->order_hint_bits - 1);
    for (i = 0; i < NUM_REF_FRAMES; i++) {
        shifted_order_hints[i] = curr_frame_hint + GetRelativeDist(p_seq_header, dpb_buffer_.ref_order_hint[i], p_frame_header->order_hint);
    }
    latest_order_hint = shifted_order_hints[p_frame_header->last_frame_idx];
    earliest_order_hint = shifted_order_hints[p_frame_header->gold_frame_idx];

    // The kAltRefFrame reference is set to be a backward reference to the frame with highest output order.
    ref = FindLatestBackward(shifted_order_hints, used_frame, curr_frame_hint, latest_order_hint);
    if (ref >= 0) {
        p_frame_header->ref_frame_idx[kAltRefFrame - kLastFrame] = ref;
        used_frame[ref] = 1;
    }
    // The kBwdRefFrame reference is set to be a backward reference to the closest frame.
    ref = FindEarliestBackward(shifted_order_hints, used_frame, curr_frame_hint, earliest_order_hint);
    if ( ref >= 0 )
    {
        p_frame_header->ref_frame_idx[kBwdRefFrame - kLastFrame] = ref;
        used_frame[ref] = 1;
    }
    // The kAltRef2Frame reference is set to the next closest backward reference.
    ref = FindEarliestBackward(shifted_order_hints, used_frame, curr_frame_hint, earliest_order_hint);
    if ( ref >= 0 )
    {
        p_frame_header->ref_frame_idx[kAltRef2Frame - kLastFrame] = ref;
        used_frame[ref] = 1;
    }

    // The remaining references are set to be forward references in anti-chronological order.
    int ref_frame_list[REFS_PER_FRAME - 2] = {kLast2Frame, kLast3Frame, kBwdRefFrame, kAltRef2Frame, kAltRefFrame};
    for (i = 0; i < REFS_PER_FRAME - 2; i++) {
        int refFrame = ref_frame_list[i];
        if (p_frame_header->ref_frame_idx[refFrame - kLastFrame] < 0) {
            ref = FindLatestForward(shifted_order_hints, used_frame, curr_frame_hint, latest_order_hint);
            if (ref >= 0) {
                p_frame_header->ref_frame_idx[refFrame - kLastFrame] = ref;
                used_frame[ref] = 1;
            }
        }
    }

    // Finally, any remaining references are set to the reference frame with smallest output order.
    ref = INVALID_INDEX;
    for (i = 0; i < NUM_REF_FRAMES; i++) {
        int hint = shifted_order_hints[i];
        if (ref < 0 || hint < earliest_order_hint) {
            ref = i;
            earliest_order_hint = hint;
        }
    }
    for (i = 0; i < REFS_PER_FRAME; i++) {
        if (p_frame_header->ref_frame_idx[i] < 0) {
            p_frame_header->ref_frame_idx[i] = ref;
        }
    }
}

int Av1VideoParser::GetRelativeDist(Av1SequenceHeader *p_seq_header, int a, int b) {
    if (!p_seq_header->enable_order_hint) {
        return 0;
    }
    int diff = a - b;
    int m = 1 << (p_seq_header->order_hint_bits - 1);
    diff = (diff & (m - 1)) - (diff & m);
    return diff;
}

int Av1VideoParser::FindLatestBackward(int *shifted_order_hints, int *used_frame, int curr_frame_hint, int &latest_order_hint) {
    int ref = INVALID_INDEX;
    int hint;

    for (int i = 0; i < NUM_REF_FRAMES; i++) {
        hint = shifted_order_hints[i];
        if (!used_frame[i] && hint >= curr_frame_hint && (ref < 0 || hint >= latest_order_hint)) {
            ref = i;
            latest_order_hint = hint;
        }
    }
    return ref;
}

int Av1VideoParser::FindEarliestBackward(int *shifted_order_hints, int *used_frame, int curr_frame_hint, int &earliest_order_hint) {
    int ref = INVALID_INDEX;
    int hint;

    for (int i = 0; i < NUM_REF_FRAMES; i++) {
        hint = shifted_order_hints[i];
        if (!used_frame[i] && hint >= curr_frame_hint && (ref < 0 || hint < earliest_order_hint)) {
            ref = i;
            earliest_order_hint = hint;
        }
    }
    return ref;
}

int Av1VideoParser::FindLatestForward(int *shifted_order_hints, int *used_frame, int curr_frame_hint, int &latest_order_hint) {
    int ref = INVALID_INDEX;
    int hint;

    for (int i = 0; i < NUM_REF_FRAMES; i++) {
        hint = shifted_order_hints[i];
        if (!used_frame[i] && hint < curr_frame_hint && (ref < 0 || hint >= latest_order_hint)) {
            ref = i;
            latest_order_hint = hint;
        }
    }
    return ref;
}

void Av1VideoParser::FrameSizeWithRefs(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    for (int i = 0; i < REFS_PER_FRAME; i++) {
        p_frame_header->found_ref = Parser::GetBit(p_stream, offset);
        if (p_frame_header->found_ref) {
            frame_header_.frame_size.upscaled_width = dpb_buffer_.ref_upscaled_width[frame_header_.ref_frame_idx[i]];
            frame_header_.frame_size.frame_width = frame_header_.frame_size.upscaled_width;
            frame_header_.frame_size.frame_width_minus_1 = frame_header_.frame_size.frame_width - 1;
            frame_header_.frame_size.frame_height = dpb_buffer_.ref_frame_height[frame_header_.ref_frame_idx[i]];
            frame_header_.frame_size.frame_height_minus_1 = frame_header_.frame_size.frame_height - 1;
            frame_header_.render_size.render_width = dpb_buffer_.ref_render_width[frame_header_.ref_frame_idx[i]];
            frame_header_.render_size.render_height = dpb_buffer_.ref_render_height[frame_header_.ref_frame_idx[i]];
            break;
        }
    }

    if (p_frame_header->found_ref == 0) {
        FrameSize(p_stream, offset, p_seq_header, p_frame_header);
        RenderSize(p_stream, offset, p_frame_header);
    } else {
        SuperResParams(p_stream, offset, p_seq_header, p_frame_header);
        ComputeImageSize(p_frame_header);
    }
}

void Av1VideoParser::SetupPastIndependence(Av1FrameHeader *p_frame_header) {
    for (int i = 0; i < MAX_SEGMENTS; i++) {
        for (int j = 0; j < SEG_LVL_MAX; j++) {
            p_frame_header->segmentation_params.feature_data[i][j] = 0;
            p_frame_header->segmentation_params.feature_enabled_flags[i][j] = 0;
        }
    }
    // Block level: PrevSegmentIds[row][col] is set equal to 0 for row = 0..MiRows-1 and col = 0..MiCols-1.
    for (int ref = kLastFrame; ref <= kAltRefFrame; ref++) {
        p_frame_header->global_motion_params.gm_type[ref] = kIdentity;
        for (int i = 0; i < 6; i++) {
            prev_gm_params_[ref][i] = (i % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS : 0;
        }
    }

    p_frame_header->loop_filter_params.loop_filter_delta_enabled = 1;
    p_frame_header->loop_filter_params.loop_filter_ref_deltas[kIntraFrame] = 1;
    p_frame_header->loop_filter_params.loop_filter_ref_deltas[kLastFrame] = 0;
    p_frame_header->loop_filter_params.loop_filter_ref_deltas[kLast2Frame] = 0;
    p_frame_header->loop_filter_params.loop_filter_ref_deltas[kLast3Frame] = 0;
    p_frame_header->loop_filter_params.loop_filter_ref_deltas[kBwdRefFrame] = 0;
    p_frame_header->loop_filter_params.loop_filter_ref_deltas[kGoldenFrame] = -1;
    p_frame_header->loop_filter_params.loop_filter_ref_deltas[kAltRefFrame] = -1;
    p_frame_header->loop_filter_params.loop_filter_ref_deltas[kAltRef2Frame] = -1;
    p_frame_header->loop_filter_params.loop_filter_mode_deltas[0] = 0;
    p_frame_header->loop_filter_params.loop_filter_mode_deltas[1] = 0;
}

void Av1VideoParser::LoadPrevious(Av1FrameHeader *p_frame_header) {
    int prev_frame = p_frame_header->ref_frame_idx[p_frame_header->primary_ref_frame];
    for (int ref = kLastFrame; ref <= kAltRefFrame; ref++) {
        for (int i = 0; i < 6; i++) {
            prev_gm_params_[ref][i] = dpb_buffer_.saved_gm_params[prev_frame][ref][i];
        }
    }
    for (int i = 0; i < TOTAL_REFS_PER_FRAME; i++) {
        p_frame_header->loop_filter_params.loop_filter_ref_deltas[i] = dpb_buffer_.saved_loop_filter_ref_deltas[prev_frame][i];
    }
    p_frame_header->loop_filter_params.loop_filter_mode_deltas[0] = dpb_buffer_.saved_loop_filter_mode_deltas[prev_frame][0];
    p_frame_header->loop_filter_params.loop_filter_mode_deltas[1] = dpb_buffer_.saved_loop_filter_mode_deltas[prev_frame][1];
    for (int j = 0; j < MAX_SEGMENTS; j++) {
        for (int k = 0; k < SEG_LVL_MAX; k++) {
            p_frame_header->segmentation_params.feature_enabled_flags[j][k] = dpb_buffer_.saved_feature_enabled[prev_frame][j][k];
            p_frame_header->segmentation_params.feature_data[j][k] = dpb_buffer_.saved_feature_data[prev_frame][j][k];
        }
    }
}

ParserResult Av1VideoParser::TileInfo(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    int32_t sb_cols;
    int32_t sb_rows;
    int32_t sb_shift;
    int32_t sb_size;
    int32_t max_tile_width_sb;
    int32_t max_tile_height_sb;
    int32_t max_tile_area_sb;
    int32_t min_log2_tile_cols;
    int32_t min_log2_tile_rows;
    int32_t max_log2_tile_cols;
    int32_t max_log2_tile_rows;
    int32_t min_log2_tiles;
    int32_t tile_width_sb;
    int32_t tile_height_sb;
    int32_t max_width;
    int32_t max_height;
    int32_t size_sb;
    int start_sb;
    int i;

    sb_cols = p_seq_header->use_128x128_superblock ? ((p_frame_header->frame_size.mi_cols + 31) >> 5) : ((p_frame_header->frame_size.mi_cols + 15) >> 4);
    sb_rows = p_seq_header->use_128x128_superblock ? ((p_frame_header->frame_size.mi_rows + 31) >> 5) : ((p_frame_header->frame_size.mi_rows + 15 ) >> 4);
    sb_shift = p_seq_header->use_128x128_superblock ? 5 : 4;
    sb_size = sb_shift + 2;
    max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    min_log2_tile_cols = TileLog2(max_tile_width_sb, sb_cols);
    max_log2_tile_cols = TileLog2(1, std::min(sb_cols, MAX_TILE_COLS));
    max_log2_tile_rows = TileLog2(1, std::min(sb_rows, MAX_TILE_ROWS));
    min_log2_tiles = std::max(min_log2_tile_cols, static_cast<int>(TileLog2(max_tile_area_sb, sb_rows * sb_cols)));

    p_frame_header->tile_info.uniform_tile_spacing_flag = Parser::GetBit(p_stream, offset);
    if (p_frame_header->tile_info.uniform_tile_spacing_flag) {
        p_frame_header->tile_info.tile_cols_log2 = min_log2_tile_cols;
        while (p_frame_header->tile_info.tile_cols_log2 < max_log2_tile_cols) {
            p_frame_header->tile_info.increment_tile_cols_log2 = Parser::GetBit(p_stream, offset);
            if (p_frame_header->tile_info.increment_tile_cols_log2 == 1) {
                p_frame_header->tile_info.tile_cols_log2++;
            } else {
                break;
            }
        }
        tile_width_sb = (sb_cols + (1 << p_frame_header->tile_info.tile_cols_log2) - 1) >> p_frame_header->tile_info.tile_cols_log2;
        CHECK_ALLOWED_MAX("tileWidthSb", tile_width_sb, max_tile_width_sb);
        i = 0;
        for (start_sb = 0; start_sb < sb_cols; start_sb += tile_width_sb) {
            p_frame_header->tile_info.mi_col_starts[i] = start_sb << sb_shift;
            i += 1;
        }
        p_frame_header->tile_info.mi_col_starts[i] = p_frame_header->frame_size.mi_cols;
        p_frame_header->tile_info.tile_cols = i;
        CHECK_ALLOWED_MAX("TileCols", p_frame_header->tile_info.tile_cols, MAX_TILE_COLS);

        min_log2_tile_rows = std::max(min_log2_tiles - p_frame_header->tile_info.tile_cols_log2, 0);
        p_frame_header->tile_info.tile_rows_log2 = min_log2_tile_rows;
        while (p_frame_header->tile_info.tile_rows_log2 < max_log2_tile_rows) {
            p_frame_header->tile_info.increment_tile_rows_log2 = Parser::GetBit(p_stream, offset);
            if (p_frame_header->tile_info.increment_tile_rows_log2 == 1) {
                p_frame_header->tile_info.tile_rows_log2++;
            } else {
                break;
            }
        }
        tile_height_sb = (sb_rows + (1 << p_frame_header->tile_info.tile_rows_log2) - 1) >> p_frame_header->tile_info.tile_rows_log2;
        CHECK_ALLOWED_MAX("tileWidthSb * tileHeightSb", tile_width_sb * tile_height_sb, max_tile_area_sb);
        i = 0;
        for (start_sb = 0; start_sb < sb_rows; start_sb += tile_height_sb ) {
            p_frame_header->tile_info.mi_row_starts[i] = start_sb << sb_shift;
            i += 1;
        }
        p_frame_header->tile_info.mi_row_starts[i] = p_frame_header->frame_size.mi_rows;
        p_frame_header->tile_info.tile_rows = i;
        CHECK_ALLOWED_MAX("TileRows", p_frame_header->tile_info.tile_rows, MAX_TILE_ROWS);

        for (i = 0; i < p_frame_header->tile_info.tile_cols - 1; i++) {
            p_frame_header->tile_info.width_in_sbs_minus_1[i] = tile_width_sb - 1;
        }
        p_frame_header->tile_info.width_in_sbs_minus_1[i] = sb_cols - (p_frame_header->tile_info.tile_cols - 1) * tile_width_sb - 1;
        for (i = 0; i < p_frame_header->tile_info.tile_rows - 1; i++) {
            p_frame_header->tile_info.height_in_sbs_minus_1[i] = tile_height_sb - 1;
        }
        p_frame_header->tile_info.height_in_sbs_minus_1[i] = sb_rows - (p_frame_header->tile_info.tile_rows - 1) * tile_height_sb - 1;
    } else {
        int widest_tile_sb = 0;
        start_sb = 0;
        for (i = 0; start_sb < sb_cols; i++) {
            p_frame_header->tile_info.mi_col_starts[i] = start_sb << sb_shift;
            max_width = std::min(sb_cols - start_sb, max_tile_width_sb);
            p_frame_header->tile_info.width_in_sbs_minus_1[i] = ReadUnsignedNonSymmetic(p_stream, offset, max_width);
            size_sb = p_frame_header->tile_info.width_in_sbs_minus_1[i] + 1;
            widest_tile_sb = std::max(size_sb, widest_tile_sb);
            start_sb += size_sb;
        }
        p_frame_header->tile_info.mi_col_starts[i] = p_frame_header->frame_size.mi_cols;
        p_frame_header->tile_info.tile_cols = i;
        CHECK_ALLOWED_MAX("TileCols", p_frame_header->tile_info.tile_cols, MAX_TILE_COLS);
        p_frame_header->tile_info.tile_cols_log2 = TileLog2(1, p_frame_header->tile_info.tile_cols);

        if (min_log2_tiles > 0) {
            max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
        } else {
            max_tile_area_sb = sb_rows * sb_cols;
        }
        max_tile_height_sb = std::max(max_tile_area_sb / widest_tile_sb, static_cast<int>(1));

        start_sb = 0;
        for (i = 0; start_sb < sb_rows; i++) {
            p_frame_header->tile_info.mi_row_starts[i] = start_sb << sb_shift;
            max_height = std::min(sb_rows - start_sb, max_tile_height_sb);
            p_frame_header->tile_info.height_in_sbs_minus_1[i] = ReadUnsignedNonSymmetic(p_stream, offset, max_height);
            size_sb = p_frame_header->tile_info.height_in_sbs_minus_1[i] + 1;
            start_sb += size_sb;
        }
        p_frame_header->tile_info.mi_row_starts[ i ] = p_frame_header->frame_size.mi_rows;
        p_frame_header->tile_info.tile_rows = i;
        CHECK_ALLOWED_MAX("TileRows", p_frame_header->tile_info.tile_rows, MAX_TILE_ROWS);
        p_frame_header->tile_info.tile_rows_log2 = TileLog2(1, p_frame_header->tile_info.tile_rows);
    }

    if (p_frame_header->tile_info.tile_cols_log2 > 0 || p_frame_header->tile_info.tile_rows_log2 > 0) {
        p_frame_header->tile_info.context_update_tile_id = Parser::ReadBits(p_stream, offset, p_frame_header->tile_info.tile_rows_log2 + p_frame_header->tile_info.tile_cols_log2);
        CHECK_ALLOWED_MAX("context_update_tile_id", p_frame_header->tile_info.context_update_tile_id, p_frame_header->tile_info.tile_cols * p_frame_header->tile_info.tile_rows);
        p_frame_header->tile_info.tile_size_bytes_minus_1 = Parser::ReadBits(p_stream, offset, 2);
    } else {
        p_frame_header->tile_info.context_update_tile_id = 0;
    }
    return PARSER_OK;
}

uint32_t Av1VideoParser::TileLog2(uint32_t blk_size, uint32_t target) {
    uint32_t k;
    for (k = 0; (blk_size << k) < target; k++ ) {}
    return k;
}

void Av1VideoParser::QuantizationParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    p_frame_header->quantization_params.base_q_idx = Parser::ReadBits(p_stream, offset, 8);
    p_frame_header->quantization_params.delta_q_y_dc = ReadDeltaQ(p_stream, offset, p_frame_header);

    if (p_seq_header->color_config.num_planes > 1) {
        if (p_seq_header->color_config.separate_uv_delta_q) {
            p_frame_header->quantization_params.diff_uv_delta = Parser::GetBit(p_stream, offset);
        } else {
            p_frame_header->quantization_params.diff_uv_delta = 0;
        }
        p_frame_header->quantization_params.delta_q_u_dc = ReadDeltaQ(p_stream, offset, p_frame_header);
        p_frame_header->quantization_params.delta_q_u_ac = ReadDeltaQ(p_stream, offset, p_frame_header);

        if (p_frame_header->quantization_params.diff_uv_delta) {
            p_frame_header->quantization_params.delta_q_v_dc = ReadDeltaQ(p_stream, offset, p_frame_header);
            p_frame_header->quantization_params.delta_q_v_ac = ReadDeltaQ(p_stream, offset, p_frame_header);
        } else {
            p_frame_header->quantization_params.delta_q_v_dc = p_frame_header->quantization_params.delta_q_u_dc;
            p_frame_header->quantization_params.delta_q_v_ac = p_frame_header->quantization_params.delta_q_u_ac;
        }
    } else {
        p_frame_header->quantization_params.delta_q_u_dc = 0;
        p_frame_header->quantization_params.delta_q_u_ac = 0;
        p_frame_header->quantization_params.delta_q_v_dc = 0;
        p_frame_header->quantization_params.delta_q_v_ac = 0;
    }

    p_frame_header->quantization_params.using_qmatrix = Parser::GetBit(p_stream, offset);
    if (p_frame_header->quantization_params.using_qmatrix) {
        p_frame_header->quantization_params.qm_y = Parser::ReadBits(p_stream, offset, 4);
        p_frame_header->quantization_params.qm_u = Parser::ReadBits(p_stream, offset, 4);
        if (!p_seq_header->color_config.separate_uv_delta_q) {
            p_frame_header->quantization_params.qm_v = p_frame_header->quantization_params.qm_u;
        } else {
            p_frame_header->quantization_params.qm_v = Parser::ReadBits(p_stream, offset, 4);
        }
    }
}

int32_t Av1VideoParser::ReadDeltaQ(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header) {
    p_frame_header->quantization_params.delta_coded = Parser::GetBit(p_stream, offset);
    if (p_frame_header->quantization_params.delta_coded) {
        p_frame_header->quantization_params.delta_q = ReadSigned(p_stream, offset, 1 + 6);
    } else {
        p_frame_header->quantization_params.delta_q = 0;
    }
    return p_frame_header->quantization_params.delta_q;
}

void Av1VideoParser::SegmentationParams(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header) {
    int i, j;
    int clipped_value;
    uint32_t bits_to_read;
    uint32_t segmentation_feature_bits[SEG_LVL_MAX] = {8,6,6,6,6,3,0,0};
    uint32_t segmentation_feature_signed[SEG_LVL_MAX] = { 1, 1, 1, 1, 1, 0, 0, 0 };
    uint32_t segmentation_feature_max[SEG_LVL_MAX] = {255, MAX_LOOP_FILTER, MAX_LOOP_FILTER, MAX_LOOP_FILTER, MAX_LOOP_FILTER, 7, 0, 0 };

    p_frame_header->segmentation_params.segmentation_enabled = Parser::GetBit(p_stream, offset);
    if (p_frame_header->segmentation_params.segmentation_enabled == 1) {
        if (p_frame_header->primary_ref_frame == PRIMARY_REF_NONE) {
            p_frame_header->segmentation_params.segmentation_update_map = 1;
            p_frame_header->segmentation_params.segmentation_temporal_update = 0;
            p_frame_header->segmentation_params.segmentation_update_data = 1;
        } else {
            p_frame_header->segmentation_params.segmentation_update_map = Parser::GetBit(p_stream, offset);
            if (p_frame_header->segmentation_params.segmentation_update_map == 1) {
                p_frame_header->segmentation_params.segmentation_temporal_update = Parser::GetBit(p_stream, offset);
            }
            p_frame_header->segmentation_params.segmentation_update_data = Parser::GetBit(p_stream, offset);
        }

        if (p_frame_header->segmentation_params.segmentation_update_data == 1) {
            for (i = 0; i < MAX_SEGMENTS; i++) {
                for (j = 0; j < SEG_LVL_MAX; j++) {
                    p_frame_header->segmentation_params.feature_value = 0;
                    p_frame_header->segmentation_params.feature_enabled = Parser::GetBit(p_stream, offset);
                    p_frame_header->segmentation_params.feature_enabled_flags[i][j] = p_frame_header->segmentation_params.feature_enabled;
                    clipped_value = 0;
                    if (p_frame_header->segmentation_params.feature_enabled == 1) {
                        bits_to_read = segmentation_feature_bits[j];
                        int limit = segmentation_feature_max[j];
                        if (segmentation_feature_signed[j] == 1) {
                            p_frame_header->segmentation_params.feature_value = ReadSigned(p_stream, offset, 1 + bits_to_read);
                            clipped_value = std::clamp(static_cast<int>(p_frame_header->segmentation_params.feature_value), -limit, limit);
                        } else {
                            p_frame_header->segmentation_params.feature_value = Parser::ReadBits(p_stream, offset, bits_to_read);
                            clipped_value = std::clamp(static_cast<int>(p_frame_header->segmentation_params.feature_value), 0, limit);
                        }
                    }
                    p_frame_header->segmentation_params.feature_data[i][j] = clipped_value;
                }
            }
        }
    } else {
        for (i = 0; i < MAX_SEGMENTS; i++) {
            for (j = 0; j < SEG_LVL_MAX; j++) {
                p_frame_header->segmentation_params.feature_enabled_flags[i][j] = 0;
                p_frame_header->segmentation_params.feature_data[i][j] = 0;
            }
        }
    }

    p_frame_header->segmentation_params.seg_id_pre_skip = 0;
    p_frame_header->segmentation_params.last_active_seg_id = 0;

    for (i = 0; i < MAX_SEGMENTS; i++) {
        for (j = 0; j < SEG_LVL_MAX; j++) {
            if (p_frame_header->segmentation_params.feature_enabled_flags[i][j]) {
                p_frame_header->segmentation_params.last_active_seg_id = i;
                if (j >= SEG_LVL_REF_FRAME) {
                    p_frame_header->segmentation_params.seg_id_pre_skip = 1;
                }
            }
        }
    }
}

void Av1VideoParser::DeltaQParams(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header) {
    p_frame_header->delta_q_params.delta_q_res = 0;
    p_frame_header->delta_q_params.delta_q_present = 0;
    if (p_frame_header->quantization_params.base_q_idx > 0) {
        p_frame_header->delta_q_params.delta_q_present = Parser::GetBit(p_stream, offset);
    }
    if (p_frame_header->delta_q_params.delta_q_present) {
        p_frame_header->delta_q_params.delta_q_res = Parser::ReadBits(p_stream, offset, 2);
    }
}

void Av1VideoParser::DeltaLFParams(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header) {
    p_frame_header->delta_lf_params.delta_lf_present = 0;
    p_frame_header->delta_lf_params.delta_lf_res = 0;
    p_frame_header->delta_lf_params.delta_lf_multi = 0;
    if (p_frame_header->delta_q_params.delta_q_present) {
        if (!p_frame_header->allow_intrabc) {
            p_frame_header->delta_lf_params.delta_lf_present = Parser::GetBit(p_stream, offset);
        }
        if (p_frame_header->delta_lf_params.delta_lf_present) {
            p_frame_header->delta_lf_params.delta_lf_res = Parser::ReadBits(p_stream, offset, 2);
            p_frame_header->delta_lf_params.delta_lf_multi = Parser::GetBit(p_stream, offset);
        }
    }
}

int Av1VideoParser::GetQIndex(Av1FrameHeader *p_frame_header, int ignore_delta_q, int segment_id) {
    // seg_feature_active_idx(segment_id, SEG_LVL_ALT_Q)
    int seg_feature_active_idx = p_frame_header->segmentation_params.segmentation_enabled && p_frame_header->segmentation_params.feature_enabled_flags[segment_id][SEG_LVL_ALT_Q];
    if (seg_feature_active_idx == 1) {
        int data = p_frame_header->segmentation_params.feature_data[segment_id][SEG_LVL_ALT_Q];
        int q_index = p_frame_header->quantization_params.base_q_idx + data;
        // CurrentQIndex is base_q_idx at tile level: If ignoreDeltaQ is equal to 0 and delta_q_present is equal to 1, set qindex equal to CurrentQIndex + data.
        return std::clamp(q_index, 0, 255);
    } else if (ignore_delta_q == 0 && p_frame_header->delta_q_params.delta_q_present == 1) {
        return p_frame_header->quantization_params.base_q_idx; // CurrentQIndex is base_q_idx at tile level
    } else {
        return p_frame_header->quantization_params.base_q_idx;
    }
}

void Av1VideoParser::LoopFilterParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    int i;

    if (p_frame_header->coded_lossless || p_frame_header->allow_intrabc) {
        p_frame_header->loop_filter_params.loop_filter_level[0] = 0;
        p_frame_header->loop_filter_params.loop_filter_level[1] = 0;
        p_frame_header->loop_filter_params.loop_filter_ref_deltas[kIntraFrame] = 1;
        p_frame_header->loop_filter_params.loop_filter_ref_deltas[kLastFrame] = 0;
        p_frame_header->loop_filter_params.loop_filter_ref_deltas[kLast2Frame] = 0;
        p_frame_header->loop_filter_params.loop_filter_ref_deltas[kLast3Frame] = 0;
        p_frame_header->loop_filter_params.loop_filter_ref_deltas[kBwdRefFrame] = 0;
        p_frame_header->loop_filter_params.loop_filter_ref_deltas[kGoldenFrame] = -1;
        p_frame_header->loop_filter_params.loop_filter_ref_deltas[kAltRefFrame] = -1;
        p_frame_header->loop_filter_params.loop_filter_ref_deltas[kAltRef2Frame] = -1;
        for (i = 0; i < 2; i++) {
            p_frame_header->loop_filter_params.loop_filter_mode_deltas[i] = 0;
        }
        return;
    }

    p_frame_header->loop_filter_params.loop_filter_level[0] = Parser::ReadBits(p_stream, offset, 6);
    p_frame_header->loop_filter_params.loop_filter_level[1] = Parser::ReadBits(p_stream, offset, 6);
    if (p_seq_header->color_config.num_planes > 1) {
        if (p_frame_header->loop_filter_params.loop_filter_level[0] || p_frame_header->loop_filter_params.loop_filter_level[1]) {
            p_frame_header->loop_filter_params.loop_filter_level[2] = Parser::ReadBits(p_stream, offset, 6);
            p_frame_header->loop_filter_params.loop_filter_level[3] = Parser::ReadBits(p_stream, offset, 6);
        }
    }

    p_frame_header->loop_filter_params.loop_filter_sharpness = Parser::ReadBits(p_stream, offset, 3);
    p_frame_header->loop_filter_params.loop_filter_delta_enabled = Parser::GetBit(p_stream, offset);
    if (p_frame_header->loop_filter_params.loop_filter_delta_enabled == 1) {
        p_frame_header->loop_filter_params.loop_filter_delta_update = Parser::GetBit(p_stream, offset);
        if (p_frame_header->loop_filter_params.loop_filter_delta_update == 1) {
            for (i = 0; i < TOTAL_REFS_PER_FRAME; i++) {
                p_frame_header->loop_filter_params.update_ref_delta = Parser::GetBit(p_stream, offset);
                if (p_frame_header->loop_filter_params.update_ref_delta == 1) {
                    p_frame_header->loop_filter_params.loop_filter_ref_deltas[i] = ReadSigned(p_stream, offset, 1 + 6);
                }
            }
            for (i = 0; i < 2; i++) {
                p_frame_header->loop_filter_params.update_mode_delta = Parser::GetBit(p_stream, offset);
                if ( p_frame_header->loop_filter_params.update_mode_delta == 1 )
                {
                    p_frame_header->loop_filter_params.loop_filter_mode_deltas[i] = ReadSigned(p_stream, offset, 1 + 6);
                }
            }
        }
    }
}

void Av1VideoParser::CdefParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    if (p_frame_header->coded_lossless || p_frame_header->allow_intrabc ||!p_seq_header->enable_cdef) {
        p_frame_header->cdef_params.cdef_bits = 0;
        p_frame_header->cdef_params.cdef_y_pri_strength[0] = 0;
        p_frame_header->cdef_params.cdef_y_sec_strength[0] = 0;
        p_frame_header->cdef_params.cdef_uv_pri_strength[0] = 0;
        p_frame_header->cdef_params.cdef_uv_sec_strength[0] = 0;
        p_frame_header->cdef_params.cdef_damping = 3;
        return;
    }

    p_frame_header->cdef_params.cdef_damping_minus_3 = Parser::ReadBits(p_stream, offset, 2);
    p_frame_header->cdef_params.cdef_damping = p_frame_header->cdef_params.cdef_damping_minus_3 + 3;
    p_frame_header->cdef_params.cdef_bits = Parser::ReadBits(p_stream, offset, 2);
    for (int i = 0; i < (1 << p_frame_header->cdef_params.cdef_bits); i++) {
        p_frame_header->cdef_params.cdef_y_pri_strength[i] = Parser::ReadBits(p_stream, offset, 4);
        p_frame_header->cdef_params.cdef_y_sec_strength[i] = Parser::ReadBits(p_stream, offset, 2);
        /* Note: cdef_y_sec_strength is to be packed into the lower 2 bits of cdef_y_strengths, same way as in coded stream.
                 VA-VPI driver or below is expected to do the conditional increment, which we skip here.
        if (p_frame_header->cdef_params.cdef_y_sec_strength[i] == 3) {
            p_frame_header->cdef_params.cdef_y_sec_strength[i] += 1;
        }*/

        if (p_seq_header->color_config.num_planes > 1) {
            p_frame_header->cdef_params.cdef_uv_pri_strength[i] = Parser::ReadBits(p_stream, offset, 4);
            p_frame_header->cdef_params.cdef_uv_sec_strength[i] = Parser::ReadBits(p_stream, offset, 2);
            /* Note: cdef_uv_sec_strength is to be packed into the lower 2 bits of cdef_uv_strengths, same way as in coded stream.
                     VA-VPI driver or below is expected to do the conditional increment, which we skip here.
            if (p_frame_header->cdef_params.cdef_uv_sec_strength[i] == 3) {
                p_frame_header->cdef_params.cdef_uv_sec_strength[i] += 1;
            }*/
        }
    }
}

void Av1VideoParser::LrParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    uint32_t remap_lr_type[4] = {kRestoreNone, kRestoreSwitchable, kRestoreWiener, kRestoreSgrproj};

    if (p_frame_header->all_lossless || p_frame_header->allow_intrabc || !p_seq_header->enable_restoration) {
        p_frame_header->lr_params.frame_restoration_type[0] = kRestoreNone;
        p_frame_header->lr_params.frame_restoration_type[1] = kRestoreNone;
        p_frame_header->lr_params.frame_restoration_type[2] = kRestoreNone;
        p_frame_header->lr_params.uses_lr = 0;
        return;
    }

    p_frame_header->lr_params.uses_lr = 0;
    uint32_t uses_chroma_lr = 0;
    for (int i = 0; i < p_seq_header->color_config.num_planes; i++) {
        p_frame_header->lr_params.lr_type[i] = Parser::ReadBits(p_stream, offset, 2);
        p_frame_header->lr_params.frame_restoration_type[i] = remap_lr_type[p_frame_header->lr_params.lr_type[i]];
        if (p_frame_header->lr_params.frame_restoration_type[i] != kRestoreNone) {
            p_frame_header->lr_params.uses_lr = 1;
            if (i > 0) {
                uses_chroma_lr = 1;
            }
        }
    }

    if (p_frame_header->lr_params.uses_lr) {
        if (p_seq_header->use_128x128_superblock) {
            p_frame_header->lr_params.lr_unit_shift = Parser::GetBit(p_stream, offset);
            p_frame_header->lr_params.lr_unit_shift++;
        } else {
            p_frame_header->lr_params.lr_unit_shift = Parser::GetBit(p_stream, offset);
            if (p_frame_header->lr_params.lr_unit_shift) {
                p_frame_header->lr_params.lr_unit_extra_shift = Parser::GetBit(p_stream, offset);
                p_frame_header->lr_params.lr_unit_shift += p_frame_header->lr_params.lr_unit_extra_shift;
            }
        }

        p_frame_header->lr_params.loop_restoration_size[0] = RESTORATION_TILESIZE_MAX >> (2 - p_frame_header->lr_params.lr_unit_shift);
        if (p_seq_header->color_config.subsampling_x && p_seq_header->color_config.subsampling_y && uses_chroma_lr) {
            p_frame_header->lr_params.lr_uv_shift = Parser::GetBit(p_stream, offset);
        } else {
            p_frame_header->lr_params.lr_uv_shift = 0;
        }
        p_frame_header->lr_params.loop_restoration_size[1] = p_frame_header->lr_params.loop_restoration_size[0] >> p_frame_header->lr_params.lr_uv_shift;
        p_frame_header->lr_params.loop_restoration_size[2] = p_frame_header->lr_params.loop_restoration_size[0] >> p_frame_header->lr_params.lr_uv_shift;
    }
}

void Av1VideoParser::ReadTxMode(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header) {
    if (p_frame_header->coded_lossless == 1) {
        p_frame_header->tx_mode.tx_mode = kOnly4x4;
    } else {
        p_frame_header->tx_mode.tx_mode_select = Parser::GetBit(p_stream, offset);
        if (p_frame_header->tx_mode.tx_mode_select) {
            p_frame_header->tx_mode.tx_mode = kTxModeSelect;
        } else {
            p_frame_header->tx_mode.tx_mode = kTxModeLargest;
        }
    }
}

void Av1VideoParser::SkipModeParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    uint32_t skip_mode_allowed;
    int forward_idx, backward_idx;
    int forward_hint, backward_hint;
    int second_forward_idx, second_forward_hint;
    uint32_t ref_hint;
    int i;

    if (p_frame_header->frame_is_intra || !p_frame_header->frame_reference_mode.reference_select || !p_seq_header->enable_order_hint) {
        skip_mode_allowed = 0;
    } else {
        forward_idx = INVALID_INDEX;
        backward_idx = INVALID_INDEX;
        forward_hint = dpb_buffer_.ref_order_hint[0];  // init value. No effect.
        backward_hint = dpb_buffer_.ref_order_hint[1];  // init value. No effect.

        for (i = 0; i < REFS_PER_FRAME; i++) {
            ref_hint = dpb_buffer_.ref_order_hint[p_frame_header->ref_frame_idx[i]];
            if (GetRelativeDist(p_seq_header, ref_hint, p_frame_header->order_hint ) < 0) {
                if (forward_idx < 0 || GetRelativeDist(p_seq_header, ref_hint, forward_hint) > 0) {
                    forward_idx = i;
                    forward_hint = ref_hint;
                }
            } else if (GetRelativeDist(p_seq_header, ref_hint, p_frame_header->order_hint) > 0) {
                if (backward_idx < 0 || GetRelativeDist(p_seq_header, ref_hint, backward_hint) < 0 ) {
                    backward_idx = i;
                    backward_hint = ref_hint;
                }
            }
        }

        if (forward_idx < 0) {
            skip_mode_allowed = 0;
        } else if (backward_idx >= 0) {
            skip_mode_allowed = 1;
            p_frame_header->skip_mode_params.skip_mode_frame[0] = kLastFrame + std::min(forward_idx, backward_idx);
            p_frame_header->skip_mode_params.skip_mode_frame[1] = kLastFrame + std::max(forward_idx, backward_idx);
        } else {
            second_forward_idx = INVALID_INDEX;
            second_forward_hint = dpb_buffer_.ref_order_hint[0];  // init value. No effect.
            for (i = 0; i < REFS_PER_FRAME; i++) {
                ref_hint = dpb_buffer_.ref_order_hint[p_frame_header->ref_frame_idx[i]];
                if (GetRelativeDist(p_seq_header, ref_hint, forward_hint ) < 0) {
                    if (second_forward_idx < 0 || GetRelativeDist(p_seq_header, ref_hint, second_forward_hint ) > 0) {
                        second_forward_idx = i;
                        second_forward_hint = ref_hint;
                    }
                }
            }
            if (second_forward_idx < 0) {
                skip_mode_allowed = 0;
            } else {
                skip_mode_allowed = 1;
                p_frame_header->skip_mode_params.skip_mode_frame[0] = kLastFrame + std::min(forward_idx, second_forward_idx);
                p_frame_header->skip_mode_params.skip_mode_frame[1] = kLastFrame + std::max(forward_idx, second_forward_idx);
            }
        }
    }

    if (skip_mode_allowed ) {
        p_frame_header->skip_mode_params.skip_mode_present = Parser::GetBit(p_stream, offset);
    } else {
        p_frame_header->skip_mode_params.skip_mode_present = 0;
    }
}

void Av1VideoParser::GlobalMotionParams(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header) {
    int ref;
    int type;

    for (ref = kLastFrame; ref <= kAltRefFrame; ref++) {
        p_frame_header->global_motion_params.gm_type[ref] = kIdentity;
        for (int i = 0; i < 6; i++) {
            p_frame_header->global_motion_params.gm_params[ref][i] = ( ( i % 3 == 2 ) ? 1 << WARPEDMODEL_PREC_BITS : 0 );
        }
    }
    if (p_frame_header->frame_is_intra) {
        return;
    }

    for (ref = kLastFrame; ref <= kAltRefFrame; ref++) {
        p_frame_header->global_motion_params.is_global = Parser::GetBit(p_stream, offset);
        if (p_frame_header->global_motion_params.is_global) {
            p_frame_header->global_motion_params.is_rot_zoom = Parser::GetBit(p_stream, offset);
            if (p_frame_header->global_motion_params.is_rot_zoom) {
                type = kRotZoom;
            } else {
                p_frame_header->global_motion_params.is_translation = Parser::GetBit(p_stream, offset);
                type = p_frame_header->global_motion_params.is_translation ? kTranslation : kAffine;
            }
        } else {
            type = kIdentity;
        }
        p_frame_header->global_motion_params.gm_type[ref] = type;

        if (type >= kRotZoom) {
            ReadGlobalParam(p_stream, offset, p_frame_header, type, ref, 2);
            ReadGlobalParam(p_stream, offset, p_frame_header, type, ref, 3);
            if (type == kAffine) {
                ReadGlobalParam(p_stream, offset, p_frame_header, type, ref, 4);
                ReadGlobalParam(p_stream, offset, p_frame_header, type, ref, 5);
            } else {
                p_frame_header->global_motion_params.gm_params[ref][4] = -p_frame_header->global_motion_params.gm_params[ref][3];
                p_frame_header->global_motion_params.gm_params[ref][5] = p_frame_header->global_motion_params.gm_params[ref][2];
            }
        }
        if (type >= kTranslation) {
            ReadGlobalParam(p_stream, offset, p_frame_header, type, ref, 0);
            ReadGlobalParam(p_stream, offset, p_frame_header, type, ref, 1);
        }
        if (type <= kAffine) {
            p_frame_header->global_motion_params.gm_invalid[ref] = !ShearParamsValidation(&p_frame_header->global_motion_params.gm_params[ref][0]);
        }
    }
}

void Av1VideoParser::ReadGlobalParam(const uint8_t *p_stream, size_t &offset, Av1FrameHeader *p_frame_header, int type, int ref, int idx) {
    int abs_bits = GM_ABS_ALPHA_BITS;
    int prec_bits = GM_ALPHA_PREC_BITS;

    if (idx < 2) {
        if (type == kTranslation) {
            abs_bits = GM_ABS_TRANS_ONLY_BITS - !p_frame_header->allow_high_precision_mv;
            prec_bits = GM_TRANS_ONLY_PREC_BITS - !p_frame_header->allow_high_precision_mv;
        } else {
            abs_bits = GM_ABS_TRANS_BITS;
            prec_bits = GM_TRANS_PREC_BITS;
        }
    }
    int prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    int round = (idx % 3) == 2 ? (1 << WARPEDMODEL_PREC_BITS) : 0;
    int sub = (idx % 3) == 2 ? (1 << prec_bits) : 0;
    int mx = (1 << abs_bits);
    int r = (prev_gm_params_[ref][idx] >> prec_diff) - sub;
    p_frame_header->global_motion_params.gm_params[ref][idx] = (DecodeSignedSubexpWithRef(p_stream, offset, -mx, mx + 1, r) << prec_diff) + round;
}

int Av1VideoParser::DecodeSignedSubexpWithRef(const uint8_t *p_stream, size_t &offset, int low, int high, int r) {
    int x = DecodeUnsignedSubexpWithRef(p_stream, offset, high - low, r - low);
    return x + low;
}

int Av1VideoParser::DecodeUnsignedSubexpWithRef(const uint8_t *p_stream, size_t &offset, int mx, int r) {
    int v = DecodeSubexp(p_stream, offset, mx);
    if ((r << 1) <= mx) {
        return InverseRecenter(r, v);
    } else {
        return mx - 1 - InverseRecenter(mx - 1 - r, v);
    }
}

int Av1VideoParser::DecodeSubexp(const uint8_t *p_stream, size_t &offset, int num_syms) {
    int i = 0;
    int mk = 0;
    int k = 3;
    while (1) {
        int b2 = i ? k + i - 1 : k;
        int a = 1 << b2;
        if (num_syms <= mk + 3 * a) {
            int subexp_final_bits = ReadUnsignedNonSymmetic(p_stream, offset, num_syms - mk);
            return subexp_final_bits + mk;
        } else {
            int subexp_more_bits = Parser::GetBit(p_stream, offset);
            if (subexp_more_bits) {
                i++;
                mk += a;
            } else {
                int subexp_bits = Parser::ReadBits(p_stream, offset, b2);
                return subexp_bits + mk;
            }
        }
    }
}

int Av1VideoParser::InverseRecenter(int r, int v) {
    if (v > 2 * r) {
        return v;
    } else if (v & 1) {
        return r - ((v + 1) >> 1);
    } else {
        return r + (v >> 1);
    }
}

int Av1VideoParser::ShearParamsValidation(int32_t *p_warp_params) {
    int alpha, beta, gamma, delta, div_factor, div_shift;
    int64_t v, w;

    alpha = std::clamp(static_cast<int>(p_warp_params[2] - (1 << WARPEDMODEL_PREC_BITS)), -32768, 32767);
    beta = std::clamp(static_cast<int>(p_warp_params[3]), -32768, 32767);
    ResolveDivisor(p_warp_params[2], &div_shift, &div_factor);
    v = (int64_t)p_warp_params[4] * (1 << WARPEDMODEL_PREC_BITS);
    gamma = std::clamp(static_cast<int>(Round2Signed((v * div_factor), div_shift)), -32768, 32767);
    w = (int64_t)p_warp_params[3] * p_warp_params[4];
    delta = std::clamp((p_warp_params[5] - (int)Round2Signed((w * div_factor), div_shift) - (1 << WARPEDMODEL_PREC_BITS)), -32768, 32767);
    alpha = Round2Signed(alpha, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS;
    beta  = Round2Signed(beta,  WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS;
    gamma = Round2Signed(gamma, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS;
    delta = Round2Signed(delta, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS;
    if ((4 * std::abs(alpha) + 7 * std::abs(beta)) >= (1 << WARPEDMODEL_PREC_BITS) || (4 * std::abs(gamma) + 4 * std::abs(delta)) >= (1 << WARPEDMODEL_PREC_BITS)) {
        return 0;
    } else {
        return 1;
    }
}

static const uint16_t div_lut[DIV_LUT_NUM] = {
  16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768,
  15709, 15650, 15592, 15534, 15477, 15420, 15364, 15308, 15252, 15197, 15142,
  15087, 15033, 14980, 14926, 14873, 14821, 14769, 14717, 14665, 14614, 14564,
  14513, 14463, 14413, 14364, 14315, 14266, 14218, 14170, 14122, 14075, 14028,
  13981, 13935, 13888, 13843, 13797, 13752, 13707, 13662, 13618, 13574, 13530,
  13487, 13443, 13400, 13358, 13315, 13273, 13231, 13190, 13148, 13107, 13066,
  13026, 12985, 12945, 12906, 12866, 12827, 12788, 12749, 12710, 12672, 12633,
  12596, 12558, 12520, 12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228,
  12193, 12157, 12122, 12087, 12053, 12018, 11984, 11950, 11916, 11882, 11848,
  11815, 11782, 11749, 11716, 11683, 11651, 11619, 11586, 11555, 11523, 11491,
  11460, 11429, 11398, 11367, 11336, 11305, 11275, 11245, 11215, 11185, 11155,
  11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894, 10866, 10838,
  10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538,
  10512, 10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255,
  10230, 10205, 10180, 10156, 10131, 10107, 10082, 10058, 10034, 10010, 9986,
  9963,  9939,  9916,  9892,  9869,  9846,  9823,  9800,  9777,  9754,  9732,
  9709,  9687,  9664,  9642,  9620,  9598,  9576,  9554,  9533,  9511,  9489,
  9468,  9447,  9425,  9404,  9383,  9362,  9341,  9321,  9300,  9279,  9259,
  9239,  9218,  9198,  9178,  9158,  9138,  9118,  9098,  9079,  9059,  9039,
  9020,  9001,  8981,  8962,  8943,  8924,  8905,  8886,  8867,  8849,  8830,
  8812,  8793,  8775,  8756,  8738,  8720,  8702,  8684,  8666,  8648,  8630,
  8613,  8595,  8577,  8560,  8542,  8525,  8508,  8490,  8473,  8456,  8439,
  8422,  8405,  8389,  8372,  8355,  8339,  8322,  8306,  8289,  8273,  8257,
  8240,  8224,  8208,  8192
};
void Av1VideoParser::ResolveDivisor(int d, int *div_shift, int *div_factor) {
    uint32_t e, f;
    int n;

    n = FloorLog2(std::abs(d));
    e = std::abs(d) - (1 << n);
    if (n > DIV_LUT_BITS) {
        f = Round2(e, n - DIV_LUT_BITS);
    } else {
        f = e << (DIV_LUT_BITS - n);
    }
    *div_shift = n + DIV_LUT_PREC_BITS;
    if ( d < 0) {
        *div_factor = -div_lut[f];
    } else {
        *div_factor = div_lut[f];
    }
}

ParserResult Av1VideoParser::FilmGrainParams(const uint8_t *p_stream, size_t &offset, Av1SequenceHeader *p_seq_header, Av1FrameHeader *p_frame_header) {
    int i;

    if (!p_seq_header->film_grain_params_present || (!p_frame_header->show_frame && !p_frame_header->showable_frame)) {
        // reset_grain_params()
        memset(&p_frame_header->film_grain_params, 0, sizeof(Av1FilmGrainParams));
        return PARSER_OK;
    }
    p_frame_header->film_grain_params.apply_grain = Parser::GetBit(p_stream, offset);
    if (!p_frame_header->film_grain_params.apply_grain) {
        // reset_grain_params()
        memset(&p_frame_header->film_grain_params, 0, sizeof(Av1FilmGrainParams));
        return PARSER_OK;
    }

    p_frame_header->film_grain_params.grain_seed = Parser::ReadBits(p_stream, offset, 16);
    if (p_frame_header->frame_type == kInterFrame) {
        p_frame_header->film_grain_params.update_grain = Parser::GetBit(p_stream, offset);
    } else {
        p_frame_header->film_grain_params.update_grain = 1;
    }

    if (!p_frame_header->film_grain_params.update_grain) {
        p_frame_header->film_grain_params.film_grain_params_ref_idx = Parser::ReadBits(p_stream, offset, 3);
        int temp_grain_seed = p_frame_header->film_grain_params.grain_seed;
        p_frame_header->film_grain_params = dpb_buffer_.saved_film_grain_params[p_frame_header->film_grain_params.film_grain_params_ref_idx]; // load_grain_params()
        p_frame_header->film_grain_params.grain_seed = temp_grain_seed;
        return PARSER_OK;
    }

    p_frame_header->film_grain_params.num_y_points = Parser::ReadBits(p_stream, offset, 4);
    CHECK_ALLOWED_MAX("num_y_points", p_frame_header->film_grain_params.num_y_points, 14);
    for (i = 0; i < p_frame_header->film_grain_params.num_y_points; i++) {
        p_frame_header->film_grain_params.point_y_value[i] = Parser::ReadBits(p_stream, offset, 8);
        if (i > 0 && p_frame_header->film_grain_params.point_y_value[i] <= p_frame_header->film_grain_params.point_y_value[i - 1]) {
            logger_.ErrorLog(MakeMsg("point_y_value["+ TOSTR(i) + "] (" + TOSTR(p_frame_header->film_grain_params.point_y_value[i]) + ") should be greater than point_y_value[" + TOSTR(i - 1) + "] (" + TOSTR(p_frame_header->film_grain_params.point_y_value[i - 1]) + ")"));
            return PARSER_INVALID_ARG;
        }
        p_frame_header->film_grain_params.point_y_scaling[i] = Parser::ReadBits(p_stream, offset, 8);
    }

    if (p_seq_header->color_config.mono_chrome) {
        p_frame_header->film_grain_params.chroma_scaling_from_luma = 0;
    } else {
        p_frame_header->film_grain_params.chroma_scaling_from_luma = Parser::GetBit(p_stream, offset);
    }

    if (p_seq_header->color_config.mono_chrome || p_frame_header->film_grain_params.chroma_scaling_from_luma || (p_seq_header->color_config.subsampling_x == 1 && p_seq_header->color_config.subsampling_y == 1 && p_frame_header->film_grain_params.num_y_points == 0)) {
        p_frame_header->film_grain_params.num_cb_points = 0;
        p_frame_header->film_grain_params.num_cr_points = 0;
    } else {
        p_frame_header->film_grain_params.num_cb_points = Parser::ReadBits(p_stream, offset, 4);
        CHECK_ALLOWED_MAX("num_cb_points", p_frame_header->film_grain_params.num_cb_points, 10);
        for (i = 0; i < p_frame_header->film_grain_params.num_cb_points; i++) {
            p_frame_header->film_grain_params.point_cb_value[i] = Parser::ReadBits(p_stream, offset, 8);
            if (i > 0 && p_frame_header->film_grain_params.point_cb_value[i] <= p_frame_header->film_grain_params.point_cb_value[i - 1]) {
                logger_.ErrorLog(MakeMsg("point_cb_value["+ TOSTR(i) + "] (" + TOSTR(p_frame_header->film_grain_params.point_cb_value[i]) + ") should be greater than point_cb_value[" + TOSTR(i - 1) + "] (" + TOSTR(p_frame_header->film_grain_params.point_cb_value[i - 1]) + ")"));
                return PARSER_INVALID_ARG;
            }
            p_frame_header->film_grain_params.point_cb_scaling[i] = Parser::ReadBits(p_stream, offset, 8);
        }
        p_frame_header->film_grain_params.num_cr_points = Parser::ReadBits(p_stream, offset, 4);
        CHECK_ALLOWED_MAX("num_cr_points", p_frame_header->film_grain_params.num_cr_points, 10);
        if (p_seq_header->color_config.subsampling_x == 1 && p_seq_header->color_config.subsampling_y == 1) {
            if (p_frame_header->film_grain_params.num_cb_points == 0 && p_frame_header->film_grain_params.num_cr_points != 0) {
                logger_.ErrorLog(MakeMsg("If subsampling_x is equal to 1 and subsampling_y is equal to 1 and num_cb_points is equal to 0, it is a requirement of bitstream conformance that num_cr_points is equal to 0."));
                return PARSER_WRONG_STATE;
            }
            if (p_frame_header->film_grain_params.num_cb_points != 0 && p_frame_header->film_grain_params.num_cr_points == 0) {
                logger_.ErrorLog(MakeMsg("If subsampling_x is equal to 1 and subsampling_y is equal to 1 and num_cb_points is not equal to 0, it is a requirement of bitstream conformance that num_cr_points is not equal to 0."));
                return PARSER_WRONG_STATE;
            }
        }
        for ( i = 0; i < p_frame_header->film_grain_params.num_cr_points; i++ )
        {
            p_frame_header->film_grain_params.point_cr_value[i] = Parser::ReadBits(p_stream, offset, 8);
            if (i > 0 && p_frame_header->film_grain_params.point_cr_value[i] <= p_frame_header->film_grain_params.point_cr_value[i - 1]) {
                logger_.ErrorLog(MakeMsg("point_cr_value["+ TOSTR(i) + "] (" + TOSTR(p_frame_header->film_grain_params.point_cr_value[i]) + ") should be greater than point_cr_value[" + TOSTR(i - 1) + "] (" + TOSTR(p_frame_header->film_grain_params.point_cr_value[i - 1]) + ")"));
                return PARSER_INVALID_ARG;
            }
            p_frame_header->film_grain_params.point_cr_scaling[i] = Parser::ReadBits(p_stream, offset, 8);
        }
    }

    p_frame_header->film_grain_params.grain_scaling_minus_8 = Parser::ReadBits(p_stream, offset, 2);
    p_frame_header->film_grain_params.ar_coeff_lag = Parser::ReadBits(p_stream, offset, 2);
    uint32_t num_pos_luma = 2 * p_frame_header->film_grain_params.ar_coeff_lag * (p_frame_header->film_grain_params.ar_coeff_lag + 1);
    uint32_t num_pos_chroma;
    if (p_frame_header->film_grain_params.num_y_points) {
        num_pos_chroma = num_pos_luma + 1;
        for (i = 0; i < num_pos_luma; i++) {
            p_frame_header->film_grain_params.ar_coeffs_y_plus_128[i] = Parser::ReadBits(p_stream, offset, 8);
        }
    } else {
        num_pos_chroma = num_pos_luma;
    }

    if (p_frame_header->film_grain_params.chroma_scaling_from_luma || p_frame_header->film_grain_params.num_cb_points) {
        for (i = 0; i < num_pos_chroma; i++) {
            p_frame_header->film_grain_params.ar_coeffs_cb_plus_128[i] = Parser::ReadBits(p_stream, offset, 8);
        }
    }

    if (p_frame_header->film_grain_params.chroma_scaling_from_luma || p_frame_header->film_grain_params.num_cr_points) {
        for (i = 0; i < num_pos_chroma; i++) {
            p_frame_header->film_grain_params.ar_coeffs_cr_plus_128[i] = Parser::ReadBits(p_stream, offset, 8);
        }
    }

    p_frame_header->film_grain_params.ar_coeff_shift_minus_6 = Parser::ReadBits(p_stream, offset, 2);
    p_frame_header->film_grain_params.grain_scale_shift = Parser::ReadBits(p_stream, offset, 2);

    if (p_frame_header->film_grain_params.num_cb_points) {
        p_frame_header->film_grain_params.cb_mult = Parser::ReadBits(p_stream, offset, 8);
        p_frame_header->film_grain_params.cb_luma_mult = Parser::ReadBits(p_stream, offset, 8);
        p_frame_header->film_grain_params.cb_offset = Parser::ReadBits(p_stream, offset, 9);
    }

    if (p_frame_header->film_grain_params.num_cr_points) {
        p_frame_header->film_grain_params.cr_mult = Parser::ReadBits(p_stream, offset, 8);
        p_frame_header->film_grain_params.cr_luma_mult = Parser::ReadBits(p_stream, offset, 8);
        p_frame_header->film_grain_params.cr_offset = Parser::ReadBits(p_stream, offset, 9);
    }

    p_frame_header->film_grain_params.overlap_flag = Parser::GetBit(p_stream, offset);
    p_frame_header->film_grain_params.clip_to_restricted_range = Parser::GetBit(p_stream, offset);
    return PARSER_OK;
}

#if DBGINFO
void Av1VideoParser::PrintVaapiParams() {
    int i, j;
    MSG("=======================");
    MSG("VAAPI parameter Info: ");
    MSG("=======================");
    MSG("pic_width = " << dec_pic_params_.pic_width);
    MSG("pic_height = " << dec_pic_params_.pic_height);
    MSG("curr_pic_idx = " << dec_pic_params_.curr_pic_idx);
    MSG("field_pic_flag = " << dec_pic_params_.field_pic_flag);
    MSG("bottom_field_flag = " << dec_pic_params_.bottom_field_flag);
    MSG("second_field = " << dec_pic_params_.second_field);
    MSG("bitstream_data_len = " << dec_pic_params_.bitstream_data_len);
    MSG("num_slices = " << dec_pic_params_.num_slices);
    MSG("ref_pic_flag = " << dec_pic_params_.ref_pic_flag);
    MSG("intra_pic_flag = " << dec_pic_params_.intra_pic_flag);

    MSG("=======================");
    MSG("Picture parameter Info:");
    MSG("=======================");
    RocdecAv1PicParams *p_pic_param = &dec_pic_params_.pic_params.av1;
    MSG("profile = " << static_cast<uint32_t>(p_pic_param->profile));
    MSG("order_hint_bits_minus_1 = " << static_cast<uint32_t>(p_pic_param->order_hint_bits_minus_1));
    MSG("bit_depth_idx = " << static_cast<uint32_t>(p_pic_param->bit_depth_idx));
    MSG("matrix_coefficients = " << static_cast<uint32_t>(p_pic_param->matrix_coefficients));
    MSG("still_picture = " << p_pic_param->seq_info_fields.fields.still_picture);
    MSG("use_128x128_superblock = " << p_pic_param->seq_info_fields.fields.use_128x128_superblock);
    MSG("enable_filter_intra = " << p_pic_param->seq_info_fields.fields.enable_filter_intra);
    MSG("enable_intra_edge_filter = " << p_pic_param->seq_info_fields.fields.enable_intra_edge_filter);
    MSG("enable_interintra_compound = " << p_pic_param->seq_info_fields.fields.enable_interintra_compound);
    MSG("enable_masked_compound = " << p_pic_param->seq_info_fields.fields.enable_masked_compound);
    MSG("enable_dual_filter = " << p_pic_param->seq_info_fields.fields.enable_dual_filter);
    MSG("enable_order_hint = " << p_pic_param->seq_info_fields.fields.enable_order_hint);
    MSG("enable_jnt_comp = " << p_pic_param->seq_info_fields.fields.enable_jnt_comp);
    MSG("enable_cdef = " << p_pic_param->seq_info_fields.fields.enable_cdef);
    MSG("mono_chrome = " << p_pic_param->seq_info_fields.fields.mono_chrome);
    MSG("color_range = " << p_pic_param->seq_info_fields.fields.color_range);
    MSG("subsampling_x = " << p_pic_param->seq_info_fields.fields.subsampling_x);
    MSG("subsampling_y = " << p_pic_param->seq_info_fields.fields.subsampling_y);
    MSG("chroma_sample_position = " << p_pic_param->seq_info_fields.fields.chroma_sample_position);
    MSG("film_grain_params_present = " << p_pic_param->seq_info_fields.fields.film_grain_params_present);
    MSG("current_frame = " << p_pic_param->current_frame);
    MSG("current_display_picture = " << p_pic_param->current_display_picture);
    MSG("anchor_frames_num = " << static_cast<uint32_t>(p_pic_param->anchor_frames_num));
    MSG("anchor_frames_list = " << p_pic_param->anchor_frames_list);
    MSG("frame_width_minus1 = " << p_pic_param->frame_width_minus1);
    MSG("frame_height_minus1 = " << p_pic_param->frame_height_minus1);
    MSG("output_frame_width_in_tiles_minus_1 = " << p_pic_param->output_frame_width_in_tiles_minus_1);
    MSG("output_frame_height_in_tiles_minus_1 = " << p_pic_param->output_frame_height_in_tiles_minus_1);
    MSG_NO_NEWLINE("ref_frame_map[]:");
    for (i = 0; i < NUM_REF_FRAMES; i++) {
        MSG_NO_NEWLINE(" " << p_pic_param->ref_frame_map[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("ref_frame_idx[]:");
    for (i = 0; i < REFS_PER_FRAME; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->ref_frame_idx[i]));
    }
    MSG("");
    MSG("primary_ref_frame = " << static_cast<uint32_t>(p_pic_param->primary_ref_frame));
    MSG("order_hint = " << static_cast<uint32_t>(p_pic_param->order_hint));
    MSG("segmentation_enabled = " << p_pic_param->seg_info.segment_info_fields.bits.enabled);
    MSG("segmentation_update_map = " << p_pic_param->seg_info.segment_info_fields.bits.update_map);
    MSG("segmentation_temporal_update = " << p_pic_param->seg_info.segment_info_fields.bits.temporal_update);
    MSG("segmentation_update_data = " << p_pic_param->seg_info.segment_info_fields.bits.update_data);
    for (i = 0; i < MAX_SEGMENTS; i++) {
        MSG("Segment " << i << ":");
        MSG_NO_NEWLINE("feature_data[]:");
        for (j = 0; j < SEG_LVL_MAX; j++) {
            MSG_NO_NEWLINE(" " << p_pic_param->seg_info.feature_data[i][j]);
        }
        MSG("");
        MSG("feature_mask = 0x" << std::hex << static_cast<uint32_t>(p_pic_param->seg_info.feature_mask[i]));
        MSG_NO_NEWLINE(std::dec);
    }
    MSG("apply_grain = " << p_pic_param->film_grain_info.film_grain_info_fields.bits.apply_grain);
    MSG("chroma_scaling_from_luma = " << p_pic_param->film_grain_info.film_grain_info_fields.bits.chroma_scaling_from_luma);
    MSG("grain_scaling_minus_8 = " << p_pic_param->film_grain_info.film_grain_info_fields.bits.grain_scaling_minus_8);
    MSG("ar_coeff_lag = " << p_pic_param->film_grain_info.film_grain_info_fields.bits.ar_coeff_lag);
    MSG("ar_coeff_shift_minus_6 = " << p_pic_param->film_grain_info.film_grain_info_fields.bits.ar_coeff_shift_minus_6);
    MSG("grain_scale_shift = " << p_pic_param->film_grain_info.film_grain_info_fields.bits.grain_scale_shift);
    MSG("overlap_flag = " << p_pic_param->film_grain_info.film_grain_info_fields.bits.overlap_flag);
    MSG("clip_to_restricted_range = " << p_pic_param->film_grain_info.film_grain_info_fields.bits.clip_to_restricted_range);
    MSG("grain_seed = " << p_pic_param->film_grain_info.grain_seed);
    MSG("num_y_points = " << static_cast<uint32_t>(p_pic_param->film_grain_info.num_y_points));
    MSG_NO_NEWLINE("point_y_value[]:");
    for (i = 0; i < p_pic_param->film_grain_info.num_y_points; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->film_grain_info.point_y_value[i]));
    }
    MSG("");
    MSG_NO_NEWLINE("point_y_scaling[]:");
    for (i = 0; i < p_pic_param->film_grain_info.num_y_points; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->film_grain_info.point_y_scaling[i]));
    }
    MSG("");
    MSG("num_cb_points = " << static_cast<uint32_t>(p_pic_param->film_grain_info.num_cb_points));
    MSG_NO_NEWLINE("point_cb_value[]:");
    for (i = 0; i < p_pic_param->film_grain_info.num_cb_points; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->film_grain_info.point_cb_value[i]));
    }
    MSG("");
    MSG_NO_NEWLINE("point_cb_scaling[]:");
    for (i = 0; i < p_pic_param->film_grain_info.num_cb_points; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->film_grain_info.point_cb_scaling[i]));
    }
    MSG("");
    MSG("num_cr_points = " << static_cast<uint32_t>(p_pic_param->film_grain_info.num_cr_points));
    MSG_NO_NEWLINE("point_cr_value[]:");
    for (i = 0; i < p_pic_param->film_grain_info.num_cr_points; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->film_grain_info.point_cr_value[i]));
    }
    MSG("");
    MSG_NO_NEWLINE("point_cr_scaling[]:");
    for (i = 0; i < p_pic_param->film_grain_info.num_cr_points; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->film_grain_info.point_cr_scaling[i]));
    }
    MSG("");
    MSG_NO_NEWLINE("ar_coeffs_y[]:");
    for (i = 0; i < 24; i++) {
        MSG_NO_NEWLINE(" " << static_cast<int>(p_pic_param->film_grain_info.ar_coeffs_y[i]));
    }
    MSG("");
    MSG_NO_NEWLINE("ar_coeffs_cb[]:");
    for (i = 0; i < 25; i++) {
        MSG_NO_NEWLINE(" " << static_cast<int>(p_pic_param->film_grain_info.ar_coeffs_cb[i]));
    }
    MSG("");
    MSG_NO_NEWLINE("ar_coeffs_cr[]:");
    for (i = 0; i < 25; i++) {
        MSG_NO_NEWLINE(" " << static_cast<int>(p_pic_param->film_grain_info.ar_coeffs_cr[i]));
    }
    MSG("");
    MSG("cb_mult = " << static_cast<uint32_t>(p_pic_param->film_grain_info.cb_mult));
    MSG("cb_luma_mult = " << static_cast<uint32_t>(p_pic_param->film_grain_info.cb_luma_mult));
    MSG("cb_offset = " << p_pic_param->film_grain_info.cb_offset);
    MSG("cr_mult = " << static_cast<uint32_t>(p_pic_param->film_grain_info.cr_mult));
    MSG("cr_luma_mult = " << static_cast<uint32_t>(p_pic_param->film_grain_info.cr_luma_mult));
    MSG("cr_offset = " << p_pic_param->film_grain_info.cr_offset);
    MSG("tile_cols = " << static_cast<uint32_t>(p_pic_param->tile_cols));
    MSG("tile_rows = " << static_cast<uint32_t>(p_pic_param->tile_rows));
    MSG_NO_NEWLINE("width_in_sbs_minus_1[]:");
    for (i = 0; i < p_pic_param->tile_cols; i++) {
        MSG_NO_NEWLINE(" " << p_pic_param->width_in_sbs_minus_1[i]);
    }
    MSG("");
    MSG_NO_NEWLINE("height_in_sbs_minus_1[]:");
    for (i = 0; i < p_pic_param->tile_rows; i++) {
        MSG_NO_NEWLINE(" " << p_pic_param->height_in_sbs_minus_1[i]);
    }
    MSG("");
    MSG("tile_count_minus_1 = " << p_pic_param->tile_count_minus_1);
    MSG("context_update_tile_id = " << p_pic_param->context_update_tile_id);
    MSG("frame_type = " << p_pic_param->pic_info_fields.bits.frame_type);
    MSG("show_frame = " << p_pic_param->pic_info_fields.bits.show_frame);
    MSG("showable_frame = " << p_pic_param->pic_info_fields.bits.showable_frame);
    MSG("error_resilient_mode = " << p_pic_param->pic_info_fields.bits.error_resilient_mode);
    MSG("disable_cdf_update = " << p_pic_param->pic_info_fields.bits.disable_cdf_update);
    MSG("allow_screen_content_tools = " << p_pic_param->pic_info_fields.bits.allow_screen_content_tools);
    MSG("force_integer_mv = " << p_pic_param->pic_info_fields.bits.force_integer_mv);
    MSG("allow_intrabc = " << p_pic_param->pic_info_fields.bits.allow_intrabc);
    MSG("use_superres = " << p_pic_param->pic_info_fields.bits.use_superres);
    MSG("allow_high_precision_mv = " << p_pic_param->pic_info_fields.bits.allow_high_precision_mv);
    MSG("is_motion_mode_switchable = " << p_pic_param->pic_info_fields.bits.is_motion_mode_switchable);
    MSG("use_ref_frame_mvs = " << p_pic_param->pic_info_fields.bits.use_ref_frame_mvs);
    MSG("disable_frame_end_update_cdf = " << p_pic_param->pic_info_fields.bits.disable_frame_end_update_cdf);
    MSG("uniform_tile_spacing_flag = " << p_pic_param->pic_info_fields.bits.uniform_tile_spacing_flag);
    MSG("allow_warped_motion = " << p_pic_param->pic_info_fields.bits.allow_warped_motion);
    MSG("large_scale_tile = " << p_pic_param->pic_info_fields.bits.large_scale_tile);
    MSG("superres_scale_denominator = " << static_cast<uint32_t>(p_pic_param->superres_scale_denominator));
    MSG("interp_filter = " << static_cast<uint32_t>(p_pic_param->interp_filter));
    MSG("filter_level[] = " << static_cast<uint32_t>(p_pic_param->filter_level[0]) <<", " << static_cast<uint32_t>(p_pic_param->filter_level[1]));
    MSG("filter_level_u = " << static_cast<uint32_t>(p_pic_param->filter_level_u));
    MSG("filter_level_v = " << static_cast<uint32_t>(p_pic_param->filter_level_v));
    MSG("sharpness_level = " << static_cast<uint32_t>(p_pic_param->loop_filter_info_fields.bits.sharpness_level));
    MSG("mode_ref_delta_enabled = " << static_cast<uint32_t>(p_pic_param->loop_filter_info_fields.bits.mode_ref_delta_enabled));
    MSG("mode_ref_delta_update = " << static_cast<uint32_t>(p_pic_param->loop_filter_info_fields.bits.mode_ref_delta_update));
    MSG_NO_NEWLINE("ref_deltas[]:");
    for (i = 0; i < TOTAL_REFS_PER_FRAME; i++) {
        MSG_NO_NEWLINE(" " << static_cast<int>(p_pic_param->ref_deltas[i]));
    }
    MSG("");
    MSG_NO_NEWLINE("mode_deltas[]:");
    for (i = 0; i < 2; i++) {
        MSG_NO_NEWLINE(" " << static_cast<int>(p_pic_param->mode_deltas[i]));
    }
    MSG("");
    MSG("base_qindex = " << static_cast<uint32_t>(p_pic_param->base_qindex));
    MSG("y_dc_delta_q = " << static_cast<int>(p_pic_param->y_dc_delta_q));
    MSG("u_dc_delta_q = " << static_cast<int>(p_pic_param->u_dc_delta_q));
    MSG("u_ac_delta_q = " << static_cast<int>(p_pic_param->u_ac_delta_q));
    MSG("v_dc_delta_q = " << static_cast<int>(p_pic_param->v_dc_delta_q));
    MSG("v_ac_delta_q = " << static_cast<int>(p_pic_param->v_ac_delta_q));
    MSG("using_qmatrix = " << p_pic_param->qmatrix_fields.bits.using_qmatrix);
    MSG("qm_y = " << p_pic_param->qmatrix_fields.bits.qm_y);
    MSG("qm_u = " << p_pic_param->qmatrix_fields.bits.qm_u);
    MSG("qm_v = " << p_pic_param->qmatrix_fields.bits.qm_v);
    MSG("delta_q_present_flag = " << p_pic_param->mode_control_fields.bits.delta_q_present_flag);
    MSG("log2_delta_q_res = " << p_pic_param->mode_control_fields.bits.log2_delta_q_res);
    MSG("delta_lf_present_flag = " << p_pic_param->mode_control_fields.bits.delta_lf_present_flag);
    MSG("log2_delta_lf_res = " << p_pic_param->mode_control_fields.bits.log2_delta_lf_res);
    MSG("delta_lf_multi = " << p_pic_param->mode_control_fields.bits.delta_lf_multi);
    MSG("tx_mode = " << p_pic_param->mode_control_fields.bits.tx_mode);
    MSG("reference_select = " << p_pic_param->mode_control_fields.bits.reference_select);
    MSG("reduced_tx_set_used = " << p_pic_param->mode_control_fields.bits.reduced_tx_set_used);
    MSG("skip_mode_present = " << p_pic_param->mode_control_fields.bits.skip_mode_present);
    MSG("cdef_damping_minus_3 = " << static_cast<uint32_t>(p_pic_param->cdef_damping_minus_3));
    MSG("cdef_bits = " << static_cast<uint32_t>(p_pic_param->cdef_bits));
    MSG_NO_NEWLINE("cdef_y_strengths[]:");
    for (int i = 0; i < 8; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->cdef_y_strengths[i]));
    }
    MSG("");
    MSG_NO_NEWLINE("cdef_uv_strengths[]:");
    for (int i = 0; i < 8; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->cdef_uv_strengths[i]));
    }
    MSG("");
    MSG("yframe_restoration_type = " << p_pic_param->loop_restoration_fields.bits.yframe_restoration_type);
    MSG("cbframe_restoration_type = " << p_pic_param->loop_restoration_fields.bits.cbframe_restoration_type);
    MSG("crframe_restoration_type = " << p_pic_param->loop_restoration_fields.bits.crframe_restoration_type);
    MSG("lr_unit_shift = " << p_pic_param->loop_restoration_fields.bits.lr_unit_shift);
    MSG("lr_uv_shift = " << p_pic_param->loop_restoration_fields.bits.lr_uv_shift);
    for (i = kLastFrame; i <= kAltRefFrame; i++) {
        MSG("wm[" << i - 1 << "]:");
        MSG("invalid = " << static_cast<uint32_t>(p_pic_param->wm[i - 1].invalid) << ", wmtype = " <<  p_pic_param->wm[i - 1].wmtype);
        MSG_NO_NEWLINE("wmmat[]:");
        for (int j = 0; j < 6; j++) {
            MSG_NO_NEWLINE(" " << p_pic_param->wm[i - 1].wmmat[j]);
        }
        MSG("");
    }

    MSG("=======================");
    MSG("Tile group parameter Info: ");
    MSG("=======================");
    for (int i = 0; i < tile_group_data_.num_tiles; i++) {
        MSG("Tile number " << i << ":");
        RocdecAv1SliceParams *p_tile_param = &dec_pic_params_.slice_params.av1[i];
        MSG("slice_data_size = " << p_tile_param->slice_data_size);
        MSG("slice_data_offset = " << p_tile_param->slice_data_offset);
        MSG("slice_data_flag = " << p_tile_param->slice_data_flag);
        MSG("tile_row = " << p_tile_param->tile_row);
        MSG("tile_column = " << p_tile_param->tile_column);
        MSG("tg_start = " << p_tile_param->tg_start);
        MSG("tg_end = " << p_tile_param->tg_end);
        MSG("anchor_frame_idx = " << static_cast<int>(p_tile_param->anchor_frame_idx));
        MSG("tile_idx_in_tile_list = " << p_tile_param->tile_idx_in_tile_list);
    }
}

void Av1VideoParser::PrintDpb() {
    uint32_t i;

    MSG("=======================");
    MSG("DPB buffer content: ");
    MSG("=======================");
    MSG("Current frame: pic_idx = " << curr_pic_.pic_idx << ", dec_buf_idx = " << curr_pic_.dec_buf_idx << ", fg_buf_idx = " << curr_pic_.fg_buf_idx << ", order_hint = " << curr_pic_.order_hint << ", frame_type = " << curr_pic_.frame_type);
    for (i = 0; i < BUFFER_POOL_MAX_SIZE; i++) {
        MSG("Frame store " << i << ": " << "dec_ref_count = " << dpb_buffer_.dec_ref_count[i] << ", pic_idx = " << dpb_buffer_.frame_store[i].pic_idx << ", dec_buf_idx = " << dpb_buffer_.frame_store[i].dec_buf_idx << ", fg_buf_idx = " << dpb_buffer_.frame_store[i].fg_buf_idx << ", current_frame_id = " << dpb_buffer_.frame_store[i].current_frame_id << ", order_hint = " << dpb_buffer_.frame_store[i].order_hint << ", frame_type = " << dpb_buffer_.frame_store[i].frame_type << ", use_status = " << dpb_buffer_.frame_store[i].use_status << ", show_frame = " << dpb_buffer_.frame_store[i].show_frame);
    }
    MSG_NO_NEWLINE("virtual_buffer_index[] =");
    for (i = 0; i < NUM_REF_FRAMES; i++) {
        MSG_NO_NEWLINE(" " << dpb_buffer_.virtual_buffer_index[i]);
    }
    MSG("");

    MSG("Decode buffer pool:");
    for(i = 0; i < dec_buf_pool_size_; i++) {
        DecodeFrameBuffer *p_dec_buf = &decode_buffer_pool_[i];
        MSG("Decode buffer " << i << ": use_status = " << p_dec_buf->use_status << ", pic_order_cnt = " << p_dec_buf->pic_order_cnt << ", pts = " << p_dec_buf->pts);
    }
    MSG("num_output_pics_ = " << num_output_pics_);
    if (num_output_pics_) {
        MSG_NO_NEWLINE("output_pic_list:");
        for (i = 0; i < num_output_pics_; i++) {
            MSG_NO_NEWLINE(" " << output_pic_list_[i]);
        }
        MSG("");
    }
}
#endif // DBGINFO