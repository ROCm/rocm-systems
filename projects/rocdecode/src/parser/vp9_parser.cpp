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
#include "vp9_parser.h"

Vp9VideoParser::Vp9VideoParser() {
    memset(&curr_pic_, 0, sizeof(Vp9Picture));
    memset(&dpb_buffer_, 0, sizeof(DecodedPictureBuffer));
    memset(&uncompressed_header_, 0, sizeof(Vp9UncompressedHeader));
    uncomp_header_size_ = 0;
    memset(&tile_params_, 0, sizeof(RocdecVp9SliceParams));
    memset(y_dequant_, 0, sizeof(y_dequant_));
    memset(uv_dequant_, 0, sizeof(uv_dequant_));
    memset(lvl_lookup_, 0, sizeof(lvl_lookup_));
    InitDpb();
    num_frames_in_chunck_ = 1;
    frame_sizes_.assign(1, 0);
    curr_surface_width_ = 0;
    curr_surface_height_ = 0;
    reconfig_option_ = ROCDEC_RECONFIG_NEW_SURFACES;
}

Vp9VideoParser::~Vp9VideoParser() {
}

rocDecStatus Vp9VideoParser::Initialize(RocdecParserParams *p_params) {
    rocDecStatus ret;
    if ((ret = RocVideoParser::Initialize(p_params)) != ROCDEC_SUCCESS) {
        return ret;
    }
    // Set display delay to at least DECODE_BUF_POOL_EXTENSION (2) to prevent synchronous submission
    if (parser_params_.max_display_delay < DECODE_BUF_POOL_EXTENSION) {
        parser_params_.max_display_delay = DECODE_BUF_POOL_EXTENSION;
    }
    CheckAndAdjustDecBufPoolSize(VP9_BUFFER_POOL_MAX_SIZE);
    return ROCDEC_SUCCESS;
}

rocDecStatus Vp9VideoParser::UnInitialize() {
    return ROCDEC_SUCCESS;
}

rocDecStatus Vp9VideoParser::ParseVideoData(RocdecSourceDataPacket *p_data) { 
    if (p_data->payload && p_data->payload_size) {
        curr_pts_ = p_data->pts;
        if (ParsePictureData(p_data->payload, p_data->payload_size) != PARSER_OK) {
            logger_.ErrorLog(MakeMsg("Error occurred in ParsePictureData()."));
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

ParserResult Vp9VideoParser::ParsePictureData(const uint8_t *p_stream, uint32_t pic_data_size) {
    ParserResult ret = PARSER_OK;

    CheckSuperframe(p_stream, pic_data_size);

    uint8_t *pic_data_ptr = const_cast<uint8_t*>(p_stream);
    for (int frame_index = 0; frame_index < num_frames_in_chunck_; frame_index++) {
        if ((ret = ParseUncompressedHeader(pic_data_ptr, frame_sizes_[frame_index])) != PARSER_OK) {
            logger_.ErrorLog(MakeMsg("Error occurred in ParseUncompressedHeader(). Skip this picture."));
        } else {
            // Init Roc decoder for the first time or reconfigure the existing decoder
            if (new_seq_activated_) {
                new_seq_activated_ = false;
                if ((ret = FlushDpb()) != PARSER_OK) {
                    return ret;
                }
                if ((ret = NotifyNewSequence(&uncompressed_header_)) != PARSER_OK) {
                    return ret;
                }
            }

            if (uncompressed_header_.show_existing_frame) {
                int disp_idx = dpb_buffer_.virtual_buffer_index[uncompressed_header_.frame_to_show_map_idx];
                if (disp_idx == INVALID_INDEX) {
                    logger_.ErrorLog(MakeMsg("Invalid existing frame index to show."));
                    return PARSER_INVALID_ARG;
                }
                if (pfn_display_picture_cb_) {
                    disp_idx = dpb_buffer_.frame_store[disp_idx].dec_buf_idx;
                    decode_buffer_pool_[disp_idx].use_status |= kFrameUsedForDisplay;
                    decode_buffer_pool_[disp_idx].pts = curr_pts_;
                    // Insert into output/display picture list
                    if (num_output_pics_ < dec_buf_pool_size_) {
                        output_pic_list_[num_output_pics_] = disp_idx;
                        num_output_pics_++;
                    } else {
                        logger_.ErrorLog(MakeMsg("Display list size larger than decode buffer pool size!"));
                        return PARSER_OUT_OF_RANGE;
                    }
                }
        #if DBGINFO
                PrintDpb();
        #endif // DBGINFO
            } else {
                pic_stream_data_ptr_ = pic_data_ptr;
                pic_stream_data_size_ = frame_sizes_[frame_index];
                num_slices_ = 1;

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
        #if DBGINFO
                PrintDpb();
        #endif // DBGINFO
                // Output decoded pictures from DPB if any are ready
                if (pfn_display_picture_cb_ && num_output_pics_ > 0) {
                    if ((ret = OutputDecodedPictures(false)) != PARSER_OK) {
                        return ret;
                    }
                }
                UpdateRefFrames();
                pic_count_++;
                dpb_buffer_.dec_ref_count[curr_pic_.pic_idx]--;
                CheckAndUpdateDecStatus();
            }
        }
        pic_data_ptr += frame_sizes_[frame_index];
    }

    return PARSER_OK;
}
void Vp9VideoParser::CheckSuperframe(const uint8_t *p_stream, uint32_t chunk_data_size) {
    const uint32_t superframe_marker = 6; // 0b110
    bool is_superframe = false;
    int num_frames = 1;
    int size_index = 0;
    int size_bytes = 0;
    uint8_t last_byte = p_stream[chunk_data_size - 1];
    if ((last_byte >> 5) == superframe_marker) {
        size_bytes = ((last_byte >> 3) & 0x3) + 1;
        num_frames = (last_byte & 0x7) + 1;
        size_index = 2 + num_frames * size_bytes;
        // Check if the first byte and the last byte match
        if (chunk_data_size >= size_index && p_stream[chunk_data_size - size_index] == last_byte) {
            is_superframe = true;
        }
    }

    if (is_superframe) {
        if (num_frames > frame_sizes_.size()) {
            frame_sizes_.resize(num_frames);
        }
        int offset = chunk_data_size - size_index + 1;
        for (int i = 0; i < num_frames; i++) {
            int count = 0;
            frame_sizes_[i] = 0;
            do {
                frame_sizes_[i] = (p_stream[offset++] << (8 * count)) + frame_sizes_[i];
                count++;
            } while (count < size_bytes);
        }
        num_frames_in_chunck_ = num_frames;
    } else {
        num_frames_in_chunck_ = 1;
        frame_sizes_[0] = chunk_data_size;
    }
}

ParserResult Vp9VideoParser::NotifyNewSequence(Vp9UncompressedHeader *p_uncomp_header) {
    video_format_params_.codec = rocDecVideoCodec_VP9;
    video_format_params_.frame_rate.numerator = frame_rate_.numerator;
    video_format_params_.frame_rate.denominator = frame_rate_.denominator;
    video_format_params_.bit_depth_luma_minus8 = p_uncomp_header->color_config.bit_depth - 8;
    video_format_params_.bit_depth_chroma_minus8 = p_uncomp_header->color_config.bit_depth - 8;
    video_format_params_.progressive_sequence = 1;
    video_format_params_.min_num_decode_surfaces = dec_buf_pool_size_;
    video_format_params_.coded_width = pic_width_;
    video_format_params_.coded_height = pic_height_;

    // 7.2.2. Color config semantics
    if (p_uncomp_header->color_config.subsampling_x == 1 && p_uncomp_header->color_config.subsampling_y == 1) {
            video_format_params_.chroma_format = rocDecVideoChromaFormat_420;
    } else if (p_uncomp_header->color_config.subsampling_x == 1 && p_uncomp_header->color_config.subsampling_y == 0) {
        video_format_params_.chroma_format = rocDecVideoChromaFormat_422;
    } else if (p_uncomp_header->color_config.subsampling_x == 0 && p_uncomp_header->color_config.subsampling_y == 0) {
        video_format_params_.chroma_format = rocDecVideoChromaFormat_444;
    } else {
        logger_.ErrorLog(MakeMsg("Unsupported chroma format."));
        return PARSER_INVALID_FORMAT;
    }

    video_format_params_.display_area.left = 0;
    video_format_params_.display_area.top = 0;
    video_format_params_.display_area.right = p_uncomp_header->frame_size.frame_width;
    video_format_params_.display_area.bottom = p_uncomp_header->frame_size.frame_height;
    video_format_params_.bitrate = 0;

    // Dispaly aspect ratio
    int disp_width = (video_format_params_.display_area.right - video_format_params_.display_area.left);
    int disp_height = (video_format_params_.display_area.bottom - video_format_params_.display_area.top);
    int gcd = std::__gcd(disp_width, disp_height); // greatest common divisor
    if (gcd) {
        video_format_params_.display_aspect_ratio.x = disp_width / gcd;
        video_format_params_.display_aspect_ratio.y = disp_height / gcd;
    }

    video_format_params_.reconfig_options = reconfig_option_;
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

ParserResult Vp9VideoParser::SendPicForDecode() {
    Vp9UncompressedHeader *p_uncomp_header = &uncompressed_header_;
    dec_pic_params_ = {0};

    dec_pic_params_.pic_width = pic_width_;
    dec_pic_params_.pic_height = pic_height_;
    dec_pic_params_.curr_pic_idx = curr_pic_.dec_buf_idx;
    dec_pic_params_.field_pic_flag = 0;
    dec_pic_params_.bottom_field_flag = 0;
    dec_pic_params_.second_field = 0;

    dec_pic_params_.bitstream_data_len = pic_stream_data_size_;
    dec_pic_params_.bitstream_data = pic_stream_data_ptr_;
    dec_pic_params_.num_slices = num_slices_;

    dec_pic_params_.ref_pic_flag = 1;
    dec_pic_params_.intra_pic_flag = frame_is_intra_;

    // Set up the picture parameter buffer
    RocdecVp9PicParams *p_pic_param = &dec_pic_params_.pic_params.vp9;
    p_pic_param->frame_width = pic_width_;
    p_pic_param->frame_height = pic_height_;
    p_pic_param->pic_fields.bits.subsampling_x = p_uncomp_header->color_config.subsampling_x;
    p_pic_param->pic_fields.bits.subsampling_y = p_uncomp_header->color_config.subsampling_y;
    p_pic_param->pic_fields.bits.frame_type = p_uncomp_header->frame_type;
    p_pic_param->pic_fields.bits.show_frame = p_uncomp_header->show_frame;
    p_pic_param->pic_fields.bits.error_resilient_mode = p_uncomp_header->error_resilient_mode;
    p_pic_param->pic_fields.bits.intra_only = p_uncomp_header->intra_only;
    p_pic_param->pic_fields.bits.allow_high_precision_mv = p_uncomp_header->allow_high_precision_mv;
    p_pic_param->pic_fields.bits.mcomp_filter_type = p_uncomp_header->interpolation_filter;
    p_pic_param->pic_fields.bits.frame_parallel_decoding_mode = p_uncomp_header->frame_parallel_decoding_mode;
    p_pic_param->pic_fields.bits.reset_frame_context = p_uncomp_header->reset_frame_context;
    p_pic_param->pic_fields.bits.refresh_frame_context = p_uncomp_header->refresh_frame_context;
    p_pic_param->pic_fields.bits.frame_context_idx = p_uncomp_header->frame_context_idx;
    p_pic_param->pic_fields.bits.segmentation_enabled = p_uncomp_header->segmentation_params.segmentation_enabled;
    p_pic_param->pic_fields.bits.segmentation_temporal_update = p_uncomp_header->segmentation_params.segmentation_temporal_update;
    p_pic_param->pic_fields.bits.segmentation_update_map = p_uncomp_header->segmentation_params.segmentation_update_map;
    p_pic_param->pic_fields.bits.last_ref_frame = p_uncomp_header->ref_frame_idx[kVp9LastFrame - kVp9LastFrame];
    p_pic_param->pic_fields.bits.last_ref_frame_sign_bias = p_uncomp_header->ref_frame_sign_bias[kVp9LastFrame];
    p_pic_param->pic_fields.bits.golden_ref_frame = p_uncomp_header->ref_frame_idx[kVp9GoldenFrame - kVp9LastFrame];
    p_pic_param->pic_fields.bits.golden_ref_frame_sign_bias = p_uncomp_header->ref_frame_sign_bias[kVp9GoldenFrame];
    p_pic_param->pic_fields.bits.alt_ref_frame = p_uncomp_header->ref_frame_idx[kVp9AltRefFrame - kVp9LastFrame];
    p_pic_param->pic_fields.bits.alt_ref_frame_sign_bias = p_uncomp_header->ref_frame_sign_bias[kVp9AltRefFrame];
    p_pic_param->pic_fields.bits.lossless_flag = p_uncomp_header->quantization_params.lossless;

    p_pic_param->filter_level = p_uncomp_header->loop_filter_params.loop_filter_level;
    p_pic_param->sharpness_level = p_uncomp_header->loop_filter_params.loop_filter_sharpness;
    p_pic_param->log2_tile_rows = p_uncomp_header->tile_info.tile_rows_log2;
    p_pic_param->log2_tile_columns = p_uncomp_header->tile_info.tile_cols_log2;
    p_pic_param->frame_header_length_in_bytes = uncomp_header_size_;
    p_pic_param->first_partition_size = p_uncomp_header->header_size_in_bytes;
    for (int i = 0; i < 7; i++) {
        p_pic_param->mb_segment_tree_probs[i] = p_uncomp_header->segmentation_params.segmentation_tree_probs[i];
    }
    for (int i = 0; i < 3; i++) {
        p_pic_param->segment_pred_probs[i] = p_uncomp_header->segmentation_params.segmentation_pred_prob[i];
    }
    p_pic_param->profile = p_uncomp_header->profile;
    p_pic_param->bit_depth = p_uncomp_header->color_config.bit_depth;

    for (int i = 0; i < VP9_NUM_REF_FRAMES; i++) {
        p_pic_param->reference_frames[i] = 0xFF;
        if (dpb_buffer_.virtual_buffer_index[i] != INVALID_INDEX) {
            p_pic_param->reference_frames[i] = dpb_buffer_.frame_store[dpb_buffer_.virtual_buffer_index[i]].dec_buf_idx;
        }
    }

    RocdecVp9SliceParams *p_tile_params = &tile_params_;
    p_tile_params->slice_data_offset = 0;
    p_tile_params->slice_data_size = pic_stream_data_size_;
    p_tile_params->slice_data_flag = 0; // VA_SLICE_DATA_FLAG_ALL;
    for (int i = 0; i < VP9_MAX_SEGMENTS; i++) {
        p_tile_params->seg_param[i].segment_flags.fields.segment_reference_enabled = p_uncomp_header->segmentation_params.feature_enabled[i][VP9_SEG_LVL_REF_FRAME];
        p_tile_params->seg_param[i].segment_flags.fields.segment_reference = p_uncomp_header->segmentation_params.feature_data[i][VP9_SEG_LVL_REF_FRAME];
        p_tile_params->seg_param[i].segment_flags.fields.segment_reference_skipped = p_uncomp_header->segmentation_params.feature_enabled[i][VP9_SEG_LVL_SKIP];
        p_tile_params->seg_param[i].luma_dc_quant_scale = y_dequant_[i][0];
        p_tile_params->seg_param[i].luma_ac_quant_scale = y_dequant_[i][1];
        p_tile_params->seg_param[i].chroma_dc_quant_scale = uv_dequant_[i][0];
        p_tile_params->seg_param[i].chroma_ac_quant_scale = uv_dequant_[i][1];
        memcpy(p_tile_params->seg_param[i].filter_level, lvl_lookup_[i], VP9_MAX_REF_FRAMES * MAX_MODE_LF_DELTAS * sizeof(uint8_t));
    }
    dec_pic_params_.slice_params.vp9 = p_tile_params;

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

void Vp9VideoParser::UpdateRefFrames() {
    for (int i = 0; i < VP9_NUM_REF_FRAMES; i++) {
        if ((uncompressed_header_.refresh_frame_flags >> i) & 1) {
            dpb_buffer_.ref_frame_width[i] = uncompressed_header_.frame_size.frame_width;
            dpb_buffer_.ref_frame_height[i] = uncompressed_header_.frame_size.frame_height;
            dpb_buffer_.ref_subsampling_x[i] = uncompressed_header_.color_config.subsampling_x;
            dpb_buffer_.ref_subsampling_y[i] = uncompressed_header_.color_config.subsampling_y;
            dpb_buffer_.ref_bit_depth[i] = uncompressed_header_.color_config.bit_depth;
            if (dpb_buffer_.virtual_buffer_index[i] != INVALID_INDEX) {
                dpb_buffer_.dec_ref_count[dpb_buffer_.virtual_buffer_index[i]]--;
            }
            dpb_buffer_.virtual_buffer_index[i] = curr_pic_.pic_idx;
            dpb_buffer_.dec_ref_count[curr_pic_.pic_idx]++;
        }
    }
}

void Vp9VideoParser::InitDpb() {
    int i;
    memset(&dpb_buffer_, 0, sizeof(DecodedPictureBuffer));
    for (i = 0; i < VP9_BUFFER_POOL_MAX_SIZE; i++) {
        dpb_buffer_.frame_store[i].pic_idx = i;
        dpb_buffer_.frame_store[i].use_status = kNotUsed;
        dpb_buffer_.dec_ref_count[i] = 0;
    }
    for (i = 0; i < VP9_NUM_REF_FRAMES; i++) {
        dpb_buffer_.virtual_buffer_index[i] = INVALID_INDEX;
    }
}

ParserResult Vp9VideoParser::FlushDpb() {
    if (pfn_display_picture_cb_ && num_output_pics_ > 0) {
        if (OutputDecodedPictures(true) != PARSER_OK) {
            return PARSER_FAIL;
        }
    }
    return PARSER_OK;
}

ParserResult Vp9VideoParser::FindFreeInDecBufPool() {
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
    decode_buffer_pool_[dec_buf_index].pts = curr_pts_;
    return PARSER_OK;
}

ParserResult Vp9VideoParser::FindFreeInDpbAndMark() {
    int i;
    for (i = 0; i < VP9_BUFFER_POOL_MAX_SIZE; i++ ) {
        if (dpb_buffer_.dec_ref_count[i] == 0) {
            break;
        }
    }
    if (i == VP9_BUFFER_POOL_MAX_SIZE) {
        logger_.ErrorLog(MakeMsg("DPB buffer overflow!"));
        return PARSER_NOT_FOUND;
    }
    curr_pic_.pic_idx = i;
    curr_pic_.use_status = kFrameUsedForDecode;
    dpb_buffer_.frame_store[curr_pic_.pic_idx] = curr_pic_;
    dpb_buffer_.dec_ref_count[curr_pic_.pic_idx]++;
    // Mark as used in decode/display buffer pool
    if (pfn_display_picture_cb_ && uncompressed_header_.show_frame) {
        int disp_idx = curr_pic_.dec_buf_idx;
        decode_buffer_pool_[disp_idx].use_status |= kFrameUsedForDisplay;
        decode_buffer_pool_[disp_idx].pts = curr_pts_;
        // Insert into output/display picture list
        if (num_output_pics_ < dec_buf_pool_size_) {
            output_pic_list_[num_output_pics_] = disp_idx;
            num_output_pics_++;
        } else {
            logger_.ErrorLog(MakeMsg("Display list size larger than decode buffer pool size!"));
            return PARSER_OUT_OF_RANGE;
        }
    }
    return PARSER_OK;
}

void Vp9VideoParser::CheckAndUpdateDecStatus() {
    for (int i = 0; i < VP9_BUFFER_POOL_MAX_SIZE; i++) {
        if (dpb_buffer_.frame_store[i].use_status != kNotUsed && dpb_buffer_.dec_ref_count[i] == 0) {
            dpb_buffer_.frame_store[i].use_status = kNotUsed;
            decode_buffer_pool_[dpb_buffer_.frame_store[i].dec_buf_idx].use_status &= ~kFrameUsedForDecode;
        }
    }
}

ParserResult Vp9VideoParser::ParseUncompressedHeader(uint8_t *p_stream, size_t size) {
    ParserResult ret = PARSER_OK;
    size_t offset = 0;  // current bit offset
    Vp9UncompressedHeader *p_uncomp_header = &uncompressed_header_;

    // memset(p_uncomp_header, 0, sizeof(Vp9UncompressedHeader));
    p_uncomp_header->frame_marker = Parser::ReadBits(p_stream, offset, 2);
    p_uncomp_header->profile_low_bit = Parser::GetBit(p_stream, offset);
    p_uncomp_header->profile_high_bit = Parser::GetBit(p_stream, offset);
    p_uncomp_header->profile = (p_uncomp_header->profile_high_bit << 1) + p_uncomp_header->profile_low_bit;
    if (p_uncomp_header->profile == 3) {
        p_uncomp_header->reserved_zero = Parser::GetBit(p_stream, offset);
        if (p_uncomp_header->reserved_zero) {
            logger_.ErrorLog(MakeMsg("Syntax error: reserved_zero in Uncompressed header is not 0 when Profile is 3"));
            return PARSER_INVALID_ARG;
        }
    }
    p_uncomp_header->show_existing_frame = Parser::GetBit(p_stream, offset);
    if (p_uncomp_header->show_existing_frame) {
        p_uncomp_header->frame_to_show_map_idx = Parser::ReadBits(p_stream, offset, 3);
        p_uncomp_header->header_size_in_bytes = 0;
        p_uncomp_header->refresh_frame_flags = 0;
        p_uncomp_header->loop_filter_params.loop_filter_level = 0;
        return PARSER_OK;
    }
    last_frame_type_ = p_uncomp_header->frame_type;
    p_uncomp_header->frame_type = Parser::GetBit(p_stream, offset);
    p_uncomp_header->show_frame = Parser::GetBit(p_stream, offset);
    p_uncomp_header->error_resilient_mode = Parser::GetBit(p_stream, offset);
    if (p_uncomp_header->frame_type == kVp9KeyFrame) {
        if ((ret = FrameSyncCode(p_stream, offset, p_uncomp_header)) != PARSER_OK) {
            return ret;
        }
        if ((ret = ColorConfig(p_stream, offset, p_uncomp_header)) != PARSER_OK) {
            return ret;
        }
        FrameSize(p_stream, offset, p_uncomp_header);
        RenderSize(p_stream, offset, p_uncomp_header);
        p_uncomp_header->refresh_frame_flags = 0xFF;
        frame_is_intra_ = 1;
    } else {
        if (p_uncomp_header->show_frame == 0) {
            p_uncomp_header->intra_only = Parser::GetBit(p_stream, offset);
        } else {
            p_uncomp_header->intra_only = 0;
        }
        frame_is_intra_ = p_uncomp_header->intra_only;
        if (p_uncomp_header->error_resilient_mode == 0) {
            p_uncomp_header->reset_frame_context = Parser::ReadBits(p_stream, offset, 2);
        } else {
            p_uncomp_header->reset_frame_context = 0;
        }
        if (p_uncomp_header->intra_only == 1) {
            if ((ret = FrameSyncCode(p_stream, offset, p_uncomp_header)) != PARSER_OK) {
                return ret;
            }
            if (p_uncomp_header->profile > 0) {
                if ((ret = ColorConfig(p_stream, offset, p_uncomp_header)) != PARSER_OK) {
                    return ret;
                }                
            } else {
                p_uncomp_header->color_config.color_space = CS_BT_601;
                p_uncomp_header->color_config.subsampling_x = 1;
                p_uncomp_header->color_config.subsampling_y = 1;
                p_uncomp_header->color_config.bit_depth = 8;
            }
            p_uncomp_header->refresh_frame_flags = Parser::ReadBits(p_stream, offset, 8);
            FrameSize(p_stream, offset, p_uncomp_header);
            RenderSize(p_stream, offset, p_uncomp_header);
        } else {
            p_uncomp_header->refresh_frame_flags = Parser::ReadBits(p_stream, offset, 8);
            for (int i = 0; i < VP9_REFS_PER_FRAME; i++) {
                p_uncomp_header->ref_frame_idx[i] = Parser::ReadBits(p_stream, offset, 3);
                p_uncomp_header->ref_frame_sign_bias[kVp9LastFrame + i] = Parser::GetBit(p_stream, offset);
            }
            FrameSizeWithRefs(p_stream, offset, p_uncomp_header);
            p_uncomp_header->allow_high_precision_mv = Parser::GetBit(p_stream, offset);
            // read_interpolation_filter()
            uint8_t literal_to_type[4] = {kVp9EightTapSmooth, kVp9EightTap, kVp9EightTapSharp, kVp9Bilinear};
            p_uncomp_header->is_filter_switchable = Parser::GetBit(p_stream, offset);
            if (p_uncomp_header->is_filter_switchable) {
                p_uncomp_header->interpolation_filter = kVp9Switchable;
            } else {
                p_uncomp_header->raw_interpolation_filter = Parser::ReadBits(p_stream, offset, 2);
                p_uncomp_header->interpolation_filter = literal_to_type[p_uncomp_header->raw_interpolation_filter];
            }
        }
    }
    if (p_uncomp_header->frame_size.frame_width == 0 && p_uncomp_header->frame_size.frame_height == 0) {
        logger_.ErrorLog(MakeMsg("Invalid picture size: width = " + TOSTR(p_uncomp_header->frame_size.frame_width) + ", height = " + TOSTR(p_uncomp_header->frame_size.frame_height) + "."));
        return PARSER_WRONG_STATE;
    }

    if (p_uncomp_header->error_resilient_mode == 0) {
        p_uncomp_header->refresh_frame_context = Parser::GetBit(p_stream, offset);
        p_uncomp_header->frame_parallel_decoding_mode = Parser::GetBit(p_stream, offset);
    } else {
        p_uncomp_header->refresh_frame_context = 0;
        p_uncomp_header->frame_parallel_decoding_mode = 1;
    }
    p_uncomp_header->frame_context_idx = Parser::ReadBits(p_stream, offset, 2);
    if (frame_is_intra_ || p_uncomp_header->error_resilient_mode) {
        SetupPastIndependence(p_uncomp_header);
        if (p_uncomp_header->frame_type == kVp9KeyFrame || p_uncomp_header->error_resilient_mode == 1 || p_uncomp_header->reset_frame_context == 3) {
            for (int i = 0; i < 4; i++) {
                // save_probs( i )
            }
        } else if (p_uncomp_header->reset_frame_context == 2) {
            // save_probs(p_uncomp_header->frame_context_idx)
        }
        p_uncomp_header->frame_context_idx = 0;
    }
    LoopFilterParams(p_stream, offset, p_uncomp_header);
    QuantizationParams(p_stream, offset, p_uncomp_header);
    if ((ret = SegmentationParams(p_stream, offset, p_uncomp_header)) != PARSER_OK) {
        return ret;
    }
    SetupSegDequant(p_uncomp_header);
    LoopFilterFrameInit(p_uncomp_header);
    if ((ret = TileInfo(p_stream, offset, p_uncomp_header)) != PARSER_OK) {
        return ret;
    }

    p_uncomp_header->header_size_in_bytes = Parser::ReadBits(p_stream, offset, 16);

    // Arbitrary size change is only supported on key frames. For other frame types, particularly inter-coded frames, only size down is
    // supported where the existing surface can be re-used.
    if (pic_width_ != p_uncomp_header->frame_size.frame_width || pic_height_ != p_uncomp_header->frame_size.frame_height) {
        pic_width_ = p_uncomp_header->frame_size.frame_width;
        pic_height_ = p_uncomp_header->frame_size.frame_height;
        if (p_uncomp_header->frame_type == kVp9KeyFrame) {
            curr_surface_width_ = pic_width_;
            curr_surface_height_ = pic_height_;
            reconfig_option_ = ROCDEC_RECONFIG_NEW_SURFACES; // Normal mode: free existing surfaces and allocate new surfaces.
        } else {
            if (pic_width_ <= curr_surface_width_ && pic_height_ <= curr_surface_height_) {
                reconfig_option_ = ROCDEC_RECONFIG_KEEP_SURFACES; // Keep the existing surfaces
            } else {
                logger_.ErrorLog(MakeMsg("VP9 video size (up) change on non-key frames is not supported."));
                return PARSER_WRONG_STATE;
            }
        }
        new_seq_activated_ = true;
    }
    uncomp_header_size_ = (offset + 7) >> 3;
    if (uncomp_header_size_ > size) {
        logger_.ErrorLog(MakeMsg("Uncompressed header size (" + TOSTR(uncomp_header_size_) + ") exceeds frame data size (" + TOSTR(size) + ")"));
        return PARSER_WRONG_STATE;
    }
    if (p_uncomp_header->header_size_in_bytes > (size - uncomp_header_size_)) {
        logger_.ErrorLog(MakeMsg("header_size_in_bytes (" + TOSTR(p_uncomp_header->header_size_in_bytes) + ") exceeds allowed size (" + TOSTR(size - uncomp_header_size_) + ")"));
        return PARSER_WRONG_STATE;
    }
    return PARSER_OK;
}

ParserResult Vp9VideoParser::FrameSyncCode(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header) {
    p_uncomp_header->frame_sync_code.frame_sync_byte_0 = Parser::ReadBits(p_stream, offset, 8);
    if (p_uncomp_header->frame_sync_code.frame_sync_byte_0 != 0x49) {
        logger_.ErrorLog(MakeMsg("Syntax error: frame_sync_byte_0 is " + TOSTR(p_uncomp_header->frame_sync_code.frame_sync_byte_0) + " but shall be equal to 0x49."));
        return PARSER_INVALID_ARG;
    }
    p_uncomp_header->frame_sync_code.frame_sync_byte_1 = Parser::ReadBits(p_stream, offset, 8);
    if (p_uncomp_header->frame_sync_code.frame_sync_byte_1 != 0x83) {
        logger_.ErrorLog(MakeMsg("Syntax error: frame_sync_byte_1 is " + TOSTR(p_uncomp_header->frame_sync_code.frame_sync_byte_1) + " but shall be equal to 0x83."));
        return PARSER_INVALID_ARG;
    }
    p_uncomp_header->frame_sync_code.frame_sync_byte_2 = Parser::ReadBits(p_stream, offset, 8);
    if (p_uncomp_header->frame_sync_code.frame_sync_byte_2 != 0x42) {
        logger_.ErrorLog(MakeMsg("Syntax error: frame_sync_byte_2 is " + TOSTR(p_uncomp_header->frame_sync_code.frame_sync_byte_2) + " but shall be equal to 0x42."));
        return PARSER_INVALID_ARG;
    }
    return PARSER_OK;
}

ParserResult Vp9VideoParser::ColorConfig(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header) {
    if (p_uncomp_header->profile >= 2) {
        p_uncomp_header->color_config.ten_or_twelve_bit = Parser::GetBit(p_stream, offset);
        p_uncomp_header->color_config.bit_depth = p_uncomp_header->color_config.ten_or_twelve_bit ? 12 : 10;
    } else {
        p_uncomp_header->color_config.bit_depth = 8;
    }
    p_uncomp_header->color_config.color_space = Parser::ReadBits(p_stream, offset, 3);
    if (p_uncomp_header->profile_low_bit == 0 && p_uncomp_header->color_config.color_space == CS_RGB) {
        logger_.ErrorLog(MakeMsg("It is a requirement of bitstream conformance that color_space is not equal to CS_RGB when profile_low_bit is equal to 0."));
        return PARSER_WRONG_STATE;
    }
    if (p_uncomp_header->color_config.color_space != CS_RGB) {
        p_uncomp_header->color_config.color_range = Parser::GetBit(p_stream, offset);
        if (p_uncomp_header->profile == 1 || p_uncomp_header->profile == 3) {
            p_uncomp_header->color_config.subsampling_x = Parser::GetBit(p_stream, offset);
            p_uncomp_header->color_config.subsampling_y = Parser::GetBit(p_stream, offset);
            p_uncomp_header->color_config.reserved_zero = Parser::GetBit(p_stream, offset);
            if (p_uncomp_header->color_config.reserved_zero) {
                logger_.ErrorLog(MakeMsg("Syntax error: reserved_zero in color config is not 0 when Profile is 1 or 3"));
                return PARSER_INVALID_ARG;
            }
        } else {
            p_uncomp_header->color_config.subsampling_x = 1;
            p_uncomp_header->color_config.subsampling_y = 1;
        }
    } else {
        p_uncomp_header->color_config.color_range = 1;
        if (p_uncomp_header->profile == 1 || p_uncomp_header->profile == 3) {
            p_uncomp_header->color_config.subsampling_x = 0;
            p_uncomp_header->color_config.subsampling_y = 0;
            p_uncomp_header->color_config.reserved_zero = Parser::GetBit(p_stream, offset);
            if (p_uncomp_header->color_config.reserved_zero) {
                logger_.ErrorLog(MakeMsg("Syntax error: reserved_zero in color config is not 0 when Profile is 1 or 3"));
                return PARSER_INVALID_ARG;
            }
        }
    }
    if (p_uncomp_header->profile_low_bit == 1 && p_uncomp_header->color_config.subsampling_x == 1 && p_uncomp_header->color_config.subsampling_y == 1) {
        logger_.ErrorLog(MakeMsg("It is a requirement of bitstream conformance that either subsampling_x is equal to 0 or subsampling_y is equal to 0 when profile_low_bit is equal to 1."));
        return PARSER_WRONG_STATE;
    }
    return PARSER_OK;
}

void Vp9VideoParser::FrameSize(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header) {
    p_uncomp_header->frame_size.frame_width_minus_1 = Parser::ReadBits(p_stream, offset, 16);
    p_uncomp_header->frame_size.frame_height_minus_1 = Parser::ReadBits(p_stream, offset, 16);
    p_uncomp_header->frame_size.frame_width = p_uncomp_header->frame_size.frame_width_minus_1 + 1;
    p_uncomp_header->frame_size.frame_height = p_uncomp_header->frame_size.frame_height_minus_1 + 1;
    ComputeImageSize(p_uncomp_header);
}

void Vp9VideoParser::RenderSize(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header) {
    p_uncomp_header->render_size.render_and_frame_size_different = Parser::GetBit(p_stream, offset);
    if (p_uncomp_header->render_size.render_and_frame_size_different) {
        p_uncomp_header->render_size.render_width_minus_1 = Parser::ReadBits(p_stream, offset, 16);
        p_uncomp_header->render_size.render_height_minus_1 = Parser::ReadBits(p_stream, offset, 16);
        p_uncomp_header->render_size.render_width = p_uncomp_header->render_size.render_width_minus_1 + 1;
        p_uncomp_header->render_size.render_height = p_uncomp_header->render_size.render_height_minus_1 + 1;
    } else {
        p_uncomp_header->render_size.render_width_minus_1 = p_uncomp_header->frame_size.frame_width_minus_1;
        p_uncomp_header->render_size.render_height_minus_1 = p_uncomp_header->frame_size.frame_height_minus_1;
        p_uncomp_header->render_size.render_width = p_uncomp_header->frame_size.frame_width;
        p_uncomp_header->render_size.render_height = p_uncomp_header->frame_size.frame_height;
    }
}

void Vp9VideoParser::FrameSizeWithRefs(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header) {
    uint8_t found_ref;
    for (int i = 0; i < 3; i++) {
        found_ref = Parser::GetBit(p_stream, offset);
        if (found_ref) {
            p_uncomp_header->frame_size.frame_width = dpb_buffer_.ref_frame_width[p_uncomp_header->ref_frame_idx[i]];
            p_uncomp_header->frame_size.frame_height = dpb_buffer_.ref_frame_height[p_uncomp_header->ref_frame_idx[i]];
            break;
        }
    }
    if (found_ref == 0) {
        FrameSize(p_stream, offset, p_uncomp_header);
    } else {
        ComputeImageSize(p_uncomp_header);
    }
    RenderSize(p_stream, offset, p_uncomp_header);
}

void Vp9VideoParser::ComputeImageSize(Vp9UncompressedHeader *p_uncomp_header) {
    p_uncomp_header->frame_size.mi_cols = (p_uncomp_header->frame_size.frame_width + 7) >> 3;
    p_uncomp_header->frame_size.mi_rows = (p_uncomp_header->frame_size.frame_height + 7) >> 3;
    p_uncomp_header->frame_size.sb64_cols = (p_uncomp_header->frame_size.mi_cols + 7) >> 3;
    p_uncomp_header->frame_size.sb64_rows = (p_uncomp_header->frame_size.mi_rows + 7) >> 3;
    // steps in 7.2.6
}

void Vp9VideoParser::SetupPastIndependence(Vp9UncompressedHeader *p_uncomp_header) {
    for (int i = 0; i < VP9_MAX_SEGMENTS; i++) {
        for (int j = 0; j < VP9_SEG_LVL_MAX; j++) {
            p_uncomp_header->segmentation_params.feature_data[i][j] = 0;
            p_uncomp_header->segmentation_params.feature_enabled[i][j] = 0;
        }
    }
    p_uncomp_header->segmentation_params.segmentation_abs_or_delta_update = 0;
    // At block level, PrevSegmentIds[ row ][ col ] is set equal to 0 for row = 0..MiRows-1 and col = 0..MiCols-1
    p_uncomp_header->loop_filter_params.loop_filter_delta_enabled = 1;
    p_uncomp_header->loop_filter_params.loop_filter_ref_deltas[kVp9IntraFrame] = 1;
    p_uncomp_header->loop_filter_params.loop_filter_ref_deltas[kVp9LastFrame] = 0;
    p_uncomp_header->loop_filter_params.loop_filter_ref_deltas[kVp9GoldenFrame] = -1;
    p_uncomp_header->loop_filter_params.loop_filter_ref_deltas[kVp9AltRefFrame] = -1;
    p_uncomp_header->loop_filter_params.loop_filter_mode_deltas[0] = 0;
    p_uncomp_header->loop_filter_params.loop_filter_mode_deltas[1] = 0;
    for (int i = 0; i < VP9_MAX_REF_FRAMES; i++) {
        p_uncomp_header->ref_frame_sign_bias[i] = 0;
    }
}

void Vp9VideoParser::LoopFilterParams(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header) {
    p_uncomp_header->loop_filter_params.loop_filter_level = Parser::ReadBits(p_stream, offset, 6);
    p_uncomp_header->loop_filter_params.loop_filter_sharpness = Parser::ReadBits(p_stream, offset, 3);
    p_uncomp_header->loop_filter_params.loop_filter_delta_enabled = Parser::GetBit(p_stream, offset);
    if (p_uncomp_header->loop_filter_params.loop_filter_delta_enabled) {
        p_uncomp_header->loop_filter_params.loop_filter_delta_update = Parser::GetBit(p_stream, offset);
        if (p_uncomp_header->loop_filter_params.loop_filter_delta_update) {
            for (int i = 0; i < 4; i++) {
                p_uncomp_header->loop_filter_params.update_ref_delta[i] = Parser::GetBit(p_stream, offset);
                if (p_uncomp_header->loop_filter_params.update_ref_delta[i]) {
                    p_uncomp_header->loop_filter_params.loop_filter_ref_deltas[i] = ReadSigned(p_stream, offset, 6);
                }
            }
            for (int i = 0; i < 2; i++) {
                p_uncomp_header->loop_filter_params.update_mode_delta[i] = Parser::GetBit(p_stream, offset);
                if (p_uncomp_header->loop_filter_params.update_mode_delta[i]) {
                    p_uncomp_header->loop_filter_params.loop_filter_mode_deltas[i] = ReadSigned(p_stream, offset, 6);
                }
            }
        }
    }
}

void Vp9VideoParser::QuantizationParams(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header) {
    p_uncomp_header->quantization_params.base_q_idx = Parser::ReadBits(p_stream, offset, 8);
    p_uncomp_header->quantization_params.delta_q_y_dc = ReadDeltaQ(p_stream, offset);
    p_uncomp_header->quantization_params.delta_q_uv_dc = ReadDeltaQ(p_stream, offset);
    p_uncomp_header->quantization_params.delta_q_uv_ac = ReadDeltaQ(p_stream, offset);
    p_uncomp_header->quantization_params.lossless = p_uncomp_header->quantization_params.base_q_idx == 0 && p_uncomp_header->quantization_params.delta_q_y_dc == 0 && p_uncomp_header->quantization_params.delta_q_uv_dc == 0 && p_uncomp_header->quantization_params.delta_q_uv_ac == 0;
}

int8_t Vp9VideoParser::ReadDeltaQ(const uint8_t *p_stream, size_t &offset) {
    uint8_t delta_coded;
    int8_t delta_q;
    delta_coded = Parser::GetBit(p_stream, offset);
    if (delta_coded) {
        delta_q = ReadSigned(p_stream, offset, 4);
    } else {
        delta_q = 0;
    }
    return delta_q;
}

ParserResult Vp9VideoParser::SegmentationParams(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header) {
    const uint8_t segmentation_feature_bits[VP9_SEG_LVL_MAX] = {8, 6, 2, 0};
    const uint8_t segmentation_feature_signed[VP9_SEG_LVL_MAX] = {1, 1, 0, 0};
    p_uncomp_header->segmentation_params.segmentation_enabled = Parser::GetBit(p_stream, offset);
    if (p_uncomp_header->segmentation_params.segmentation_enabled) {
        p_uncomp_header->segmentation_params.segmentation_update_map = Parser::GetBit(p_stream, offset);
        if (p_uncomp_header->segmentation_params.segmentation_update_map) {
            for (int i = 0; i < 7; i++) {
                p_uncomp_header->segmentation_params.segmentation_tree_probs[i] = ReadProb(p_stream, offset);
            }
            p_uncomp_header->segmentation_params.segmentation_temporal_update = Parser::GetBit(p_stream, offset);
            for (int i = 0; i < 3; i++) {
                p_uncomp_header->segmentation_params.segmentation_pred_prob[i] = p_uncomp_header->segmentation_params.segmentation_temporal_update ? ReadProb(p_stream, offset) : 255;
            }
        }
        p_uncomp_header->segmentation_params.segmentation_update_data = Parser::GetBit(p_stream, offset);
        if (p_uncomp_header->segmentation_params.segmentation_update_data) {
            p_uncomp_header->segmentation_params.segmentation_abs_or_delta_update = Parser::GetBit(p_stream, offset);
            for (int i = 0; i < VP9_MAX_SEGMENTS; i++) {
                for (int j = 0; j < VP9_SEG_LVL_MAX; j++) {
                    int feature_value = 0;
                    p_uncomp_header->segmentation_params.feature_enabled[i][j] = Parser::GetBit(p_stream, offset);
                    if (p_uncomp_header->segmentation_params.feature_enabled[i][j]) {
                        int bits_to_read = segmentation_feature_bits[j];
                        if (bits_to_read) {
                            feature_value = Parser::ReadBits(p_stream, offset, bits_to_read);
                        }
                        if (segmentation_feature_signed[j] == 1) {
                            uint8_t feature_sign = Parser::GetBit(p_stream, offset);
                            if (feature_sign) {
                                if (p_uncomp_header->segmentation_params.segmentation_abs_or_delta_update == 1) {
                                    logger_.ErrorLog(MakeMsg("It is a requirement of bitstream conformance that feature_sign is equal to 0 when segmentation_abs_or_delta_update is equal to 1."));
                                    return PARSER_WRONG_STATE;
                                }
                                feature_value *= -1;
                            }
                        }
                    }
                    p_uncomp_header->segmentation_params.feature_data[i][j] = feature_value;
                }
            }
        }
    } else {
        p_uncomp_header->segmentation_params.segmentation_update_map = 0;
        p_uncomp_header->segmentation_params.segmentation_temporal_update = 0;
    }
    return PARSER_OK;
}

uint8_t Vp9VideoParser::ReadProb(const uint8_t *p_stream, size_t &offset) {
    uint8_t prob_coded;
    uint8_t prob = 255;
    prob_coded = Parser::GetBit(p_stream, offset);
    if (prob_coded) {
        prob = Parser::ReadBits(p_stream, offset, 8);
    }
    return prob;
}

ParserResult Vp9VideoParser::TileInfo(const uint8_t *p_stream, size_t &offset, Vp9UncompressedHeader *p_uncomp_header) {
    // calc_min_log2_tile_cols()
    int min_log2 = 0;
    while ((MAX_TILE_WIDTH_B64 << min_log2) < p_uncomp_header->frame_size.sb64_cols) {
        min_log2++;
    }
    p_uncomp_header->tile_info.min_log2_tile_cols = min_log2;
    // calc_max_log2_tile_cols()
    int max_log2 = 1;
    while ((p_uncomp_header->frame_size.sb64_cols >> max_log2) >= MIN_TILE_WIDTH_B64) {
        max_log2++;
    }
    p_uncomp_header->tile_info.max_log2_tile_cols = max_log2 - 1;
    p_uncomp_header->tile_info.tile_cols_log2 = p_uncomp_header->tile_info.min_log2_tile_cols;
    while (p_uncomp_header->tile_info.tile_cols_log2 < p_uncomp_header->tile_info.max_log2_tile_cols) {
        if (Parser::GetBit(p_stream, offset)) { // increment_tile_cols_log2
            p_uncomp_header->tile_info.tile_cols_log2++;
        } else {
            break;
        }
    }
    CHECK_ALLOWED_MAX("tile_cols_log2", p_uncomp_header->tile_info.tile_cols_log2, 6);
    p_uncomp_header->tile_info.tile_rows_log2 = Parser::GetBit(p_stream, offset);
    if (p_uncomp_header->tile_info.tile_rows_log2) {
        uint8_t increment_tile_rows_log2 = Parser::GetBit(p_stream, offset);
        p_uncomp_header->tile_info.tile_rows_log2 += increment_tile_rows_log2;
    }
    return PARSER_OK;
}

static const int16_t dc_qlookup[3][256] = {
 {4,    8,    8,    9,    10,  11,  12,  12,  13,  14,  15,   16,   17,   18,
  19,   19,   20,   21,   22,  23,  24,  25,  26,  26,  27,   28,   29,   30,
  31,   32,   32,   33,   34,  35,  36,  37,  38,  38,  39,   40,   41,   42,
  43,   43,   44,   45,   46,  47,  48,  48,  49,  50,  51,   52,   53,   53,
  54,   55,   56,   57,   57,  58,  59,  60,  61,  62,  62,   63,   64,   65,
  66,   66,   67,   68,   69,  70,  70,  71,  72,  73,  74,   74,   75,   76,
  77,   78,   78,   79,   80,  81,  81,  82,  83,  84,  85,   85,   87,   88,
  90,   92,   93,   95,   96,  98,  99,  101, 102, 104, 105,  107,  108,  110,
  111,  113,  114,  116,  117, 118, 120, 121, 123, 125, 127,  129,  131,  134,
  136,  138,  140,  142,  144, 146, 148, 150, 152, 154, 156,  158,  161,  164,
  166,  169,  172,  174,  177, 180, 182, 185, 187, 190, 192,  195,  199,  202,
  205,  208,  211,  214,  217, 220, 223, 226, 230, 233, 237,  240,  243,  247,
  250,  253,  257,  261,  265, 269, 272, 276, 280, 284, 288,  292,  296,  300,
  304,  309,  313,  317,  322, 326, 330, 335, 340, 344, 349,  354,  359,  364,
  369,  374,  379,  384,  389, 395, 400, 406, 411, 417, 423,  429,  435,  441,
  447,  454,  461,  467,  475, 482, 489, 497, 505, 513, 522,  530,  539,  549,
  559,  569,  579,  590,  602, 614, 626, 640, 654, 668, 684,  700,  717,  736,
  755,  775,  796,  819,  843, 869, 896, 925, 955, 988, 1022, 1058, 1098, 1139,
  1184, 1232, 1282, 1336,},
 {4,    9,    10,   13,   15,   17,   20,   22,   25,   28,   31,   34,   37,
  40,   43,   47,   50,   53,   57,   60,   64,   68,   71,   75,   78,   82,
  86,   90,   93,   97,   101,  105,  109,  113,  116,  120,  124,  128,  132,
  136,  140,  143,  147,  151,  155,  159,  163,  166,  170,  174,  178,  182,
  185,  189,  193,  197,  200,  204,  208,  212,  215,  219,  223,  226,  230,
  233,  237,  241,  244,  248,  251,  255,  259,  262,  266,  269,  273,  276,
  280,  283,  287,  290,  293,  297,  300,  304,  307,  310,  314,  317,  321,
  324,  327,  331,  334,  337,  343,  350,  356,  362,  369,  375,  381,  387,
  394,  400,  406,  412,  418,  424,  430,  436,  442,  448,  454,  460,  466,
  472,  478,  484,  490,  499,  507,  516,  525,  533,  542,  550,  559,  567,
  576,  584,  592,  601,  609,  617,  625,  634,  644,  655,  666,  676,  687,
  698,  708,  718,  729,  739,  749,  759,  770,  782,  795,  807,  819,  831,
  844,  856,  868,  880,  891,  906,  920,  933,  947,  961,  975,  988,  1001,
  1015, 1030, 1045, 1061, 1076, 1090, 1105, 1120, 1137, 1153, 1170, 1186, 1202,
  1218, 1236, 1253, 1271, 1288, 1306, 1323, 1342, 1361, 1379, 1398, 1416, 1436,
  1456, 1476, 1496, 1516, 1537, 1559, 1580, 1601, 1624, 1647, 1670, 1692, 1717,
  1741, 1766, 1791, 1817, 1844, 1871, 1900, 1929, 1958, 1990, 2021, 2054, 2088,
  2123, 2159, 2197, 2236, 2276, 2319, 2363, 2410, 2458, 2508, 2561, 2616, 2675,
  2737, 2802, 2871, 2944, 3020, 3102, 3188, 3280, 3375, 3478, 3586, 3702, 3823,
  3953, 4089, 4236, 4394, 4559, 4737, 4929, 5130, 5347,},
 {4,     12,    18,    25,    33,    41,    50,    60,    70,    80,    91,
  103,   115,   127,   140,   153,   166,   180,   194,   208,   222,   237,
  251,   266,   281,   296,   312,   327,   343,   358,   374,   390,   405,
  421,   437,   453,   469,   484,   500,   516,   532,   548,   564,   580,
  596,   611,   627,   643,   659,   674,   690,   706,   721,   737,   752,
  768,   783,   798,   814,   829,   844,   859,   874,   889,   904,   919,
  934,   949,   964,   978,   993,   1008,  1022,  1037,  1051,  1065,  1080,
  1094,  1108,  1122,  1136,  1151,  1165,  1179,  1192,  1206,  1220,  1234,
  1248,  1261,  1275,  1288,  1302,  1315,  1329,  1342,  1368,  1393,  1419,
  1444,  1469,  1494,  1519,  1544,  1569,  1594,  1618,  1643,  1668,  1692,
  1717,  1741,  1765,  1789,  1814,  1838,  1862,  1885,  1909,  1933,  1957,
  1992,  2027,  2061,  2096,  2130,  2165,  2199,  2233,  2267,  2300,  2334,
  2367,  2400,  2434,  2467,  2499,  2532,  2575,  2618,  2661,  2704,  2746,
  2788,  2830,  2872,  2913,  2954,  2995,  3036,  3076,  3127,  3177,  3226,
  3275,  3324,  3373,  3421,  3469,  3517,  3565,  3621,  3677,  3733,  3788,
  3843,  3897,  3951,  4005,  4058,  4119,  4181,  4241,  4301,  4361,  4420,
  4479,  4546,  4612,  4677,  4742,  4807,  4871,  4942,  5013,  5083,  5153,
  5222,  5291,  5367,  5442,  5517,  5591,  5665,  5745,  5825,  5905,  5984,
  6063,  6149,  6234,  6319,  6404,  6495,  6587,  6678,  6769,  6867,  6966,
  7064,  7163,  7269,  7376,  7483,  7599,  7715,  7832,  7958,  8085,  8214,
  8352,  8492,  8635,  8788,  8945,  9104,  9275,  9450,  9639,  9832,  10031,
  10245, 10465, 10702, 10946, 11210, 11482, 11776, 12081, 12409, 12750, 13118,
  13501, 13913, 14343, 14807, 15290, 15812, 16356, 16943, 17575, 18237, 18949,
  19718, 20521, 21387,}
};
static const int16_t ac_qlookup[3][256] = {
 {4,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17,   18,   19,
  20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,
  33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,
  46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,
  59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,
  72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,
  85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,
  98,   99,   100,  101,  102,  104,  106,  108,  110,  112,  114,  116,  118,
  120,  122,  124,  126,  128,  130,  132,  134,  136,  138,  140,  142,  144,
  146,  148,  150,  152,  155,  158,  161,  164,  167,  170,  173,  176,  179,
  182,  185,  188,  191,  194,  197,  200,  203,  207,  211,  215,  219,  223,
  227,  231,  235,  239,  243,  247,  251,  255,  260,  265,  270,  275,  280,
  285,  290,  295,  300,  305,  311,  317,  323,  329,  335,  341,  347,  353,
  359,  366,  373,  380,  387,  394,  401,  408,  416,  424,  432,  440,  448,
  456,  465,  474,  483,  492,  501,  510,  520,  530,  540,  550,  560,  571,
  582,  593,  604,  615,  627,  639,  651,  663,  676,  689,  702,  715,  729,
  743,  757,  771,  786,  801,  816,  832,  848,  864,  881,  898,  915,  933,
  951,  969,  988,  1007, 1026, 1046, 1066, 1087, 1108, 1129, 1151, 1173, 1196,
  1219, 1243, 1267, 1292, 1317, 1343, 1369, 1396, 1423, 1451, 1479, 1508, 1537,
  1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,},
 {4,    9,    11,   13,   16,   18,   21,   24,   27,   30,   33,   37,   40,
  44,   48,   51,   55,   59,   63,   67,   71,   75,   79,   83,   88,   92,
  96,   100,  105,  109,  114,  118,  122,  127,  131,  136,  140,  145,  149,
  154,  158,  163,  168,  172,  177,  181,  186,  190,  195,  199,  204,  208,
  213,  217,  222,  226,  231,  235,  240,  244,  249,  253,  258,  262,  267,
  271,  275,  280,  284,  289,  293,  297,  302,  306,  311,  315,  319,  324,
  328,  332,  337,  341,  345,  349,  354,  358,  362,  367,  371,  375,  379,
  384,  388,  392,  396,  401,  409,  417,  425,  433,  441,  449,  458,  466,
  474,  482,  490,  498,  506,  514,  523,  531,  539,  547,  555,  563,  571,
  579,  588,  596,  604,  616,  628,  640,  652,  664,  676,  688,  700,  713,
  725,  737,  749,  761,  773,  785,  797,  809,  825,  841,  857,  873,  889,
  905,  922,  938,  954,  970,  986,  1002, 1018, 1038, 1058, 1078, 1098, 1118,
  1138, 1158, 1178, 1198, 1218, 1242, 1266, 1290, 1314, 1338, 1362, 1386, 1411,
  1435, 1463, 1491, 1519, 1547, 1575, 1603, 1631, 1663, 1695, 1727, 1759, 1791,
  1823, 1859, 1895, 1931, 1967, 2003, 2039, 2079, 2119, 2159, 2199, 2239, 2283,
  2327, 2371, 2415, 2459, 2507, 2555, 2603, 2651, 2703, 2755, 2807, 2859, 2915,
  2971, 3027, 3083, 3143, 3203, 3263, 3327, 3391, 3455, 3523, 3591, 3659, 3731,
  3803, 3876, 3952, 4028, 4104, 4184, 4264, 4348, 4432, 4516, 4604, 4692, 4784,
  4876, 4972, 5068, 5168, 5268, 5372, 5476, 5584, 5692, 5804, 5916, 6032, 6148,
  6268, 6388, 6512, 6640, 6768, 6900, 7036, 7172, 7312,},
 {4,     13,    19,    27,    35,    44,    54,    64,    75,    87,    99,
  112,   126,   139,   154,   168,   183,   199,   214,   230,   247,   263,
  280,   297,   314,   331,   349,   366,   384,   402,   420,   438,   456,
  475,   493,   511,   530,   548,   567,   586,   604,   623,   642,   660,
  679,   698,   716,   735,   753,   772,   791,   809,   828,   846,   865,
  884,   902,   920,   939,   957,   976,   994,   1012,  1030,  1049,  1067,
  1085,  1103,  1121,  1139,  1157,  1175,  1193,  1211,  1229,  1246,  1264,
  1282,  1299,  1317,  1335,  1352,  1370,  1387,  1405,  1422,  1440,  1457,
  1474,  1491,  1509,  1526,  1543,  1560,  1577,  1595,  1627,  1660,  1693,
  1725,  1758,  1791,  1824,  1856,  1889,  1922,  1954,  1987,  2020,  2052,
  2085,  2118,  2150,  2183,  2216,  2248,  2281,  2313,  2346,  2378,  2411,
  2459,  2508,  2556,  2605,  2653,  2701,  2750,  2798,  2847,  2895,  2943,
  2992,  3040,  3088,  3137,  3185,  3234,  3298,  3362,  3426,  3491,  3555,
  3619,  3684,  3748,  3812,  3876,  3941,  4005,  4069,  4149,  4230,  4310,
  4390,  4470,  4550,  4631,  4711,  4791,  4871,  4967,  5064,  5160,  5256,
  5352,  5448,  5544,  5641,  5737,  5849,  5961,  6073,  6185,  6297,  6410,
  6522,  6650,  6778,  6906,  7034,  7162,  7290,  7435,  7579,  7723,  7867,
  8011,  8155,  8315,  8475,  8635,  8795,  8956,  9132,  9308,  9484,  9660,
  9836,  10028, 10220, 10412, 10604, 10812, 11020, 11228, 11437, 11661, 11885,
  12109, 12333, 12573, 12813, 13053, 13309, 13565, 13821, 14093, 14365, 14637,
  14925, 15213, 15502, 15806, 16110, 16414, 16734, 17054, 17390, 17726, 18062,
  18414, 18766, 19134, 19502, 19886, 20270, 20670, 21070, 21486, 21902, 22334,
  22766, 23214, 23662, 24126, 24590, 25070, 25551, 26047, 26559, 27071, 27599,
  28143, 28687, 29247,}
};

int Vp9VideoParser::DcQ(int bit_depth, int index) {
    return dc_qlookup[(bit_depth - 8) >> 1][std::clamp(index, 0, 255)];
}

int Vp9VideoParser::AcQ(int bit_depth, int index) {
    return ac_qlookup[(bit_depth - 8) >> 1][std::clamp(index, 0, 255)];
}

int Vp9VideoParser::GetQIndex(Vp9UncompressedHeader *p_uncomp_header, int seg_id) {
    int value = 0;
    if (p_uncomp_header->segmentation_params.segmentation_enabled && p_uncomp_header->segmentation_params.feature_enabled[seg_id][VP9_SEG_LVL_ALT_Q]) {
        value = p_uncomp_header->segmentation_params.feature_data[seg_id][VP9_SEG_LVL_ALT_Q];
        if (p_uncomp_header->segmentation_params.segmentation_abs_or_delta_update == 0) {
            value += p_uncomp_header->quantization_params.base_q_idx;
            value = std::clamp(value, 0, 255);
        }
    } else {
        value = p_uncomp_header->quantization_params.base_q_idx;
    }
    return value;
}

void Vp9VideoParser::SetupSegDequant(Vp9UncompressedHeader *p_uncomp_header) {
    int q_index;
    if (p_uncomp_header->segmentation_params.segmentation_enabled) {
        for (int i = 0; i < VP9_MAX_SEGMENTS; i++) {
            q_index = GetQIndex(p_uncomp_header, i);
            y_dequant_[i][0] = DcQ(p_uncomp_header->color_config.bit_depth, q_index + p_uncomp_header->quantization_params.delta_q_y_dc);
            y_dequant_[i][1] = AcQ(p_uncomp_header->color_config.bit_depth, q_index);
            uv_dequant_[i][0] = DcQ(p_uncomp_header->color_config.bit_depth, q_index + p_uncomp_header->quantization_params.delta_q_uv_dc);
            uv_dequant_[i][1] = AcQ(p_uncomp_header->color_config.bit_depth, q_index + p_uncomp_header->quantization_params.delta_q_uv_ac);
        }
    } else {
        // When segmentation is disabled, only the first value is used.
        q_index = p_uncomp_header->quantization_params.base_q_idx;
        y_dequant_[0][0] = DcQ(p_uncomp_header->color_config.bit_depth, q_index + p_uncomp_header->quantization_params.delta_q_y_dc);
        y_dequant_[0][1] = AcQ(p_uncomp_header->color_config.bit_depth, q_index);
        uv_dequant_[0][0] = DcQ(p_uncomp_header->color_config.bit_depth, q_index + p_uncomp_header->quantization_params.delta_q_uv_dc);
        uv_dequant_[0][1] = AcQ(p_uncomp_header->color_config.bit_depth, q_index + p_uncomp_header->quantization_params.delta_q_uv_ac);
    }
}

void Vp9VideoParser::LoopFilterFrameInit(Vp9UncompressedHeader *p_uncomp_header) {
    int n_shift = p_uncomp_header->loop_filter_params.loop_filter_level >> 5;
    for (int seg_id = 0; seg_id < VP9_MAX_SEGMENTS; seg_id++) {
        uint8_t lvl_seg = p_uncomp_header->loop_filter_params.loop_filter_level;
        if (p_uncomp_header->segmentation_params.feature_enabled[seg_id][VP9_SEG_LVL_ALT_L]) {
            if (p_uncomp_header->segmentation_params.segmentation_abs_or_delta_update) {
                lvl_seg = p_uncomp_header->segmentation_params.feature_data[seg_id][VP9_SEG_LVL_ALT_L];
            } else {
                lvl_seg += p_uncomp_header->segmentation_params.feature_data[seg_id][VP9_SEG_LVL_ALT_L];
            }
            lvl_seg = std::clamp(static_cast<int>(lvl_seg), 0, VP9_MAX_LOOP_FILTER);
        }
        if (p_uncomp_header->loop_filter_params.loop_filter_delta_update == 0) {
            memset(lvl_lookup_[seg_id], lvl_seg, VP9_MAX_REF_FRAMES * MAX_MODE_LF_DELTAS * sizeof(uint8_t));
        }
        if (p_uncomp_header->loop_filter_params.loop_filter_delta_enabled) {
            uint8_t intra_lvl = lvl_seg + (p_uncomp_header->loop_filter_params.loop_filter_ref_deltas[kVp9IntraFrame] << n_shift);
            lvl_lookup_[seg_id][kVp9IntraFrame][0] = std::clamp(static_cast<int>(intra_lvl), 0, VP9_MAX_LOOP_FILTER);
            for (int ref = kVp9LastFrame; ref < VP9_MAX_REF_FRAMES; ref++) {
                for (int mode = 0; mode < MAX_MODE_LF_DELTAS; mode++) {
                    uint8_t inter_lvl = lvl_seg + (p_uncomp_header->loop_filter_params.loop_filter_ref_deltas[ref] << n_shift)
                    + (p_uncomp_header->loop_filter_params.loop_filter_mode_deltas[mode] << n_shift);
                    lvl_lookup_[seg_id][ref][mode] = std::clamp(static_cast<int>(inter_lvl), 0, VP9_MAX_LOOP_FILTER);
                }
            }
        }
    }
}

#if DBGINFO
void Vp9VideoParser::PrintVaapiParams() {
    int i;
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
    RocdecVp9PicParams *p_pic_param = &dec_pic_params_.pic_params.vp9;
    MSG("frame_width = " << p_pic_param->frame_width);
    MSG("frame_height = " << p_pic_param->frame_height);
    MSG_NO_NEWLINE("reference_frames[]:");
    for (i = 0; i < VP9_NUM_REF_FRAMES; i++) {
        MSG_NO_NEWLINE(" " << p_pic_param->reference_frames[i]);
    }
    MSG("");
    MSG("subsampling_x = " << p_pic_param->pic_fields.bits.subsampling_x);
    MSG("subsampling_y = " << p_pic_param->pic_fields.bits.subsampling_y);
    MSG("frame_type = " << p_pic_param->pic_fields.bits.frame_type);
    MSG("show_frame = " << p_pic_param->pic_fields.bits.show_frame);
    MSG("error_resilient_mode = " << p_pic_param->pic_fields.bits.error_resilient_mode);
    MSG("intra_only = " << p_pic_param->pic_fields.bits.intra_only);
    MSG("allow_high_precision_mv = " << p_pic_param->pic_fields.bits.allow_high_precision_mv);
    MSG("mcomp_filter_type = " << p_pic_param->pic_fields.bits.mcomp_filter_type);
    MSG("frame_parallel_decoding_mode = " << p_pic_param->pic_fields.bits.frame_parallel_decoding_mode);
    MSG("reset_frame_context = " << p_pic_param->pic_fields.bits.reset_frame_context);
    MSG("refresh_frame_context = " << p_pic_param->pic_fields.bits.refresh_frame_context);
    MSG("frame_context_idx = " << p_pic_param->pic_fields.bits.frame_context_idx);
    MSG("segmentation_enabled = " << p_pic_param->pic_fields.bits.segmentation_enabled);
    MSG("segmentation_temporal_update = " << p_pic_param->pic_fields.bits.segmentation_temporal_update);
    MSG("segmentation_update_map = " << p_pic_param->pic_fields.bits.segmentation_update_map);
    MSG("last_ref_frame = " << p_pic_param->pic_fields.bits.last_ref_frame);
    MSG("last_ref_frame_sign_bias = " << p_pic_param->pic_fields.bits.last_ref_frame_sign_bias);
    MSG("golden_ref_frame = " << p_pic_param->pic_fields.bits.golden_ref_frame);
    MSG("golden_ref_frame_sign_bias = " << p_pic_param->pic_fields.bits.golden_ref_frame_sign_bias);
    MSG("alt_ref_frame = " << p_pic_param->pic_fields.bits.alt_ref_frame);
    MSG("alt_ref_frame_sign_bias = " << p_pic_param->pic_fields.bits.alt_ref_frame_sign_bias);
    MSG("lossless_flag = " << p_pic_param->pic_fields.bits.lossless_flag);
    MSG("filter_level = " << static_cast<uint32_t>(p_pic_param->filter_level));
    MSG("sharpness_level = " << static_cast<uint32_t>(p_pic_param->sharpness_level));
    MSG("log2_tile_rows = " << static_cast<uint32_t>(p_pic_param->log2_tile_rows));
    MSG("log2_tile_columns = " << static_cast<uint32_t>(p_pic_param->log2_tile_columns));
    MSG("frame_header_length_in_bytes = " << static_cast<uint32_t>(p_pic_param->frame_header_length_in_bytes));
    MSG("first_partition_size = " << p_pic_param->first_partition_size);
    MSG_NO_NEWLINE("mb_segment_tree_probs[]:");
    for (i = 0; i < 7; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->mb_segment_tree_probs[i]));
    }
    MSG("");
    MSG_NO_NEWLINE("segment_pred_probs[]:");
    for (i = 0; i < 3; i++) {
        MSG_NO_NEWLINE(" " << static_cast<uint32_t>(p_pic_param->segment_pred_probs[i]));
    }
    MSG("");
    MSG("profile = " << static_cast<uint32_t>(p_pic_param->profile));
    MSG("bit_depth = " << static_cast<uint32_t>(p_pic_param->bit_depth));

    MSG("=======================");
    MSG("Tile parameter Info: ");
    MSG("=======================");
    RocdecVp9SliceParams *p_tile_param = dec_pic_params_.slice_params.vp9;
    MSG("slice_data_size = " << p_tile_param->slice_data_size);
    MSG("slice_data_offset = " << p_tile_param->slice_data_offset);
    MSG("slice_data_flag = " << p_tile_param->slice_data_flag);

    for (i = 0; i < 8; i++) {
        RocdecVp9SegmentParameter *p_seg_param = &p_tile_param->seg_param[i];
        MSG("Segment " << i << ":");
        MSG("segment_reference_enabled = " << p_seg_param->segment_flags.fields.segment_reference_enabled);
        MSG("segment_reference = " << p_seg_param->segment_flags.fields.segment_reference);
        MSG("segment_reference_skipped = " << p_seg_param->segment_flags.fields.segment_reference_skipped);
        MSG_NO_NEWLINE("filter_level[4][2]: ");
        for (int j = 0; j < 4; j++) {
            MSG_NO_NEWLINE("(" << static_cast<uint32_t>(p_seg_param->filter_level[j][0]) << " " << static_cast<uint32_t>(p_seg_param->filter_level[j][1]) << ") ");
        }
        MSG("");
        MSG("luma_ac_quant_scale = " << p_seg_param->luma_ac_quant_scale);
        MSG("luma_dc_quant_scale = " << p_seg_param->luma_dc_quant_scale);
        MSG("chroma_ac_quant_scale = " << p_seg_param->chroma_ac_quant_scale);
        MSG("chroma_dc_quant_scale = " << p_seg_param->chroma_dc_quant_scale);
    }
}

void Vp9VideoParser::PrintDpb() {
    uint32_t i;

    MSG("=======================");
    MSG("DPB buffer content: ");
    MSG("=======================");
    MSG("Current frame: pic_idx = " << curr_pic_.pic_idx << ", dec_buf_idx = " << curr_pic_.dec_buf_idx);
    for (i = 0; i < VP9_BUFFER_POOL_MAX_SIZE; i++) {
        MSG("Frame store " << i << ": " << "dec_ref_count = " << dpb_buffer_.dec_ref_count[i] << ", pic_idx = " << dpb_buffer_.frame_store[i].pic_idx << ", dec_buf_idx = " << dpb_buffer_.frame_store[i].dec_buf_idx << ", use_status = " << dpb_buffer_.frame_store[i].use_status);
    }
    MSG_NO_NEWLINE("virtual_buffer_index[] =");
    for (i = 0; i < VP9_NUM_REF_FRAMES; i++) {
        MSG_NO_NEWLINE(" " << dpb_buffer_.virtual_buffer_index[i]);
    }
    MSG("");

    MSG("Decode buffer pool:");
    for(i = 0; i < dec_buf_pool_size_; i++) {
        DecodeFrameBuffer *p_dec_buf = &decode_buffer_pool_[i];
        MSG("Decode buffer " << i << ": use_status = " << p_dec_buf->use_status << ", pts = " << p_dec_buf->pts);
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