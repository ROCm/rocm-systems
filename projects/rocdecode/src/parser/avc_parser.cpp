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

#include <algorithm>
#include "avc_parser.h"

AvcVideoParser::AvcVideoParser() {
    active_sps_id_ = -1;
    active_pps_id_ = -1;

    prev_pic_order_cnt_msb_ = 0;
    prev_pic_order_cnt_lsb_ = 0;
    prev_top_field_order_cnt_ = 0;
    prev_frame_num_offset_ = 0;
    prev_frame_num_ = 0;
    prev_ref_frame_num_ = 0;
    prev_has_mmco_5_ = 0;
    curr_has_mmco_5_ = 0;
    prev_ref_pic_bottom_field_ = 0;
    curr_ref_pic_bottom_field_ = 0;
    max_long_term_frame_idx_ = NO_LONG_TERM_FRAME_INDICES;

    slice_info_list_.assign(INIT_SLICE_LIST_NUM, {{0}});
    slice_param_list_.assign(INIT_SLICE_LIST_NUM, {0});
    memset(&curr_pic_, 0, sizeof(AvcPicture));
    field_pic_count_ = 0;
    second_field_ = 0;
    first_field_pic_idx_ = 0;
    first_field_dec_buf_idx_ = 0;

    InitDpb();
}

AvcVideoParser::~AvcVideoParser() {
}

rocDecStatus AvcVideoParser::Initialize(RocdecParserParams *p_params) {
    return RocVideoParser::Initialize(p_params);
}

rocDecStatus AvcVideoParser::UnInitialize() {
    return ROCDEC_SUCCESS;
}

rocDecStatus AvcVideoParser::ParseVideoData(RocdecSourceDataPacket *p_data) {
    if (p_data->payload && p_data->payload_size) {
        curr_pts_ = p_data->pts;
        if (ParsePictureData(p_data->payload, p_data->payload_size) != PARSER_OK) {
            logger_.ErrorLog(MakeMsg(STR("Parser failed!")));
            return ROCDEC_RUNTIME_ERROR;
        }

        // Init Roc decoder for the first time or reconfigure the existing decoder
        if (new_seq_activated_) {
            if (FlushDpb() != PARSER_OK) {
                return ROCDEC_RUNTIME_ERROR;
            }
            if (NotifyNewSps(&sps_list_[active_sps_id_]) != PARSER_OK) {
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

        // Output decoded pictures from DPB if any are ready in case of frame_num gaps.
        if (pfn_display_picture_cb_ && num_output_pics_ > 0) {
            if (OutputDecodedPictures(false) != PARSER_OK) {
                return ROCDEC_RUNTIME_ERROR;
            }
        }

        // Decode the picture
        if (SendPicForDecode() != PARSER_OK) {
            logger_.ErrorLog(MakeMsg(STR("Failed to decode!")));
            return ROCDEC_RUNTIME_ERROR;
        }

        // Decoded reference picture marking (8.2.5) for later pictures
        if (MarkDecodedRefPics() != PARSER_OK) {
            return ROCDEC_RUNTIME_ERROR;
        }

        if (InsertCurrPicIntoDpb() != PARSER_OK) {
            return ROCDEC_RUNTIME_ERROR;
        }
        if (CheckDpbAndOutput() != PARSER_OK) {
            return ROCDEC_RUNTIME_ERROR;
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

ParserResult AvcVideoParser::ParsePictureData(const uint8_t *p_stream, uint32_t pic_data_size) {
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
    curr_pic_ = {0};

    do {
        ret = GetNalUnit();
        if (ret == PARSER_NOT_FOUND) {
            logger_.ErrorLog(MakeMsg(STR("Error: no start code found in the frame data.")));
            return ret;
        }

        // Parse the NAL unit
        if (nal_unit_size_ >= 4) {
            // start code + NAL unit header = 4 bytes
            int ebsp_size = nal_unit_size_ - 4 > RBSP_BUF_SIZE ? RBSP_BUF_SIZE : nal_unit_size_ - 4; // only copy enough bytes for header parsing

            nal_unit_header_ = ParseNalUnitHeader(pic_data_buffer_ptr_[curr_start_code_offset_ + 3]);
            switch (nal_unit_header_.nal_unit_type) {
                case kAvcNalTypeSeq_Parameter_Set: {
                    memcpy(rbsp_buf_, (pic_data_buffer_ptr_ + curr_start_code_offset_ + 4), ebsp_size);
                    rbsp_size_ = EbspToRbsp(rbsp_buf_, 0, ebsp_size);
                    if ((ret2 = ParseSps(rbsp_buf_, rbsp_size_)) != PARSER_OK) {
                        logger_.ErrorLog(MakeMsg("Error occurred in SPS parsing. This SPS NAL unit is skipped."));
                    }
                    break;
                }

                case kAvcNalTypePic_Parameter_Set: {
                    memcpy(rbsp_buf_, (pic_data_buffer_ptr_ + curr_start_code_offset_ + 4), ebsp_size);
                    rbsp_size_ = EbspToRbsp(rbsp_buf_, 0, ebsp_size);
                    if ((ret2 = ParsePps(rbsp_buf_, rbsp_size_)) != PARSER_OK) {
                        logger_.ErrorLog(MakeMsg("Error occurred in PPS parsing. This PPS NAL unit is skipped."));
                    }
                    break;
                }
                
                case kAvcNalTypeSlice_IDR:
                case kAvcNalTypeSlice_Non_IDR:
                case kAvcNalTypeSlice_Data_Partition_A:
                case kAvcNalTypeSlice_Data_Partition_B:
                case kAvcNalTypeSlice_Data_Partition_C: {
                    // Save slice NAL unit header
                    slice_nal_unit_header_ = nal_unit_header_;

                    // Resize slice info list if needed
                    if ((num_slices_ + 1) > slice_info_list_.size()) {
                        slice_info_list_.resize(num_slices_ + 1, {{0}});
                    }

                    memcpy(rbsp_buf_, (pic_data_buffer_ptr_ + curr_start_code_offset_ + 4), ebsp_size);
                    rbsp_size_ = EbspToRbsp(rbsp_buf_, 0, ebsp_size);
                    AvcSliceHeader *p_slice_header = &slice_info_list_[num_slices_].slice_header;
                    if ((ret2 = ParseSliceHeader(rbsp_buf_, rbsp_size_, p_slice_header)) != PARSER_OK) {
                        logger_.ErrorLog(MakeMsg("Error occurred in slice header parsing. This slice NAL unit is skipped."));
                        break;      // ignore and continue to next nal_unit
                    }
                    slice_info_list_[num_slices_].slice_data_offset = curr_start_code_offset_;
                    slice_info_list_[num_slices_].slice_data_size = nal_unit_size_;

                    // Start decode process
                    if (num_slices_ == 0) {
                        if (p_slice_header->field_pic_flag) {
                            second_field_ = field_pic_count_ & 1;
                            field_pic_count_++;
                        } else {
                            second_field_ = 0;
                        }

                        // Use the data directly from demuxer without copying
                        pic_stream_data_ptr_ = pic_data_buffer_ptr_ + curr_start_code_offset_;
                        // Picture stream data size is calculated as the diff between the frame end and the first slice offset.
                        // This is to consider the possibility of non-slice NAL units between slices.
                        pic_stream_data_size_ = pic_data_size - curr_start_code_offset_;

                        // Decode gaps in frame_num if needed (8.2.5.2)
                        if (DecodeFrameNumGaps() == PARSER_NOT_SUPPORTED) {
                            FlushDpb();
                            break;
                        }

                        // Set current picture properties
                        CalculateCurrPoc(); // 8.2.1
                        prev_has_mmco_5_ = curr_has_mmco_5_;
                        prev_ref_pic_bottom_field_ = curr_ref_pic_bottom_field_;
                        if (p_slice_header->field_pic_flag) {
                            if (p_slice_header->bottom_field_flag) {
                                curr_pic_.pic_structure = kBottomField;
                            } else {
                                curr_pic_.pic_structure = kTopField;
                            }
                        } else {
                            curr_pic_.pic_structure = kFrame;
                        }
                        curr_pic_.frame_num = p_slice_header->frame_num;
                        if (p_slice_header->field_pic_flag == 0 || second_field_) {
                            curr_pic_.pic_output_flag = 1; // Annex C. OutputFlag is set to 1 for Annex A streams
                        }
                    }

                    // Reference picture lists construction (8.2.4)
                    if (SetupReflist(&slice_info_list_[num_slices_]) != PARSER_OK) {
                        logger_.ErrorLog(MakeMsg("Error occurred in SetupReflist(). Ignore and continue."));
                    }

                    if (num_slices_ == 0) {
                        if ((ret2 = FindFreeInDecBufPool()) != PARSER_OK) {
                            return ret2;
                        }
                        if ((ret2 = FindFreeBufInDpb()) != PARSER_OK) {
                            return ret2;
                        }
                    }
                    num_slices_++;
                    break;
                }

                case kAvcNalTypeSEI_Info: {
                    if (pfn_get_sei_message_cb_) {
                        int sei_ebsp_size = nal_unit_size_ - 4; // copy the entire NAL unit
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
                        memcpy(sei_rbsp_buf_, (pic_data_buffer_ptr_ + curr_start_code_offset_ + 4), sei_ebsp_size);
                        rbsp_size_ = EbspToRbsp(sei_rbsp_buf_, 0, sei_ebsp_size);
                        ParseSeiMessage(sei_rbsp_buf_, rbsp_size_);
                    }
                    break;
                }

                case kAvcNalTypeEnd_Of_Seq: {
                    break;
                }

                case kAvcNalTypeEnd_Of_Stream: {
                    pic_count_ = 0;
                    field_pic_count_ = 0;
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

ParserResult AvcVideoParser::NotifyNewSps(AvcSeqParameterSet *p_sps) {
    video_format_params_.codec = rocDecVideoCodec_AVC;
    video_format_params_.frame_rate.numerator = frame_rate_.numerator;
    video_format_params_.frame_rate.denominator = frame_rate_.denominator;
    video_format_params_.bit_depth_luma_minus8 = p_sps->bit_depth_luma_minus8;
    video_format_params_.bit_depth_chroma_minus8 = p_sps->bit_depth_chroma_minus8;
    video_format_params_.progressive_sequence = p_sps->frame_mbs_only_flag ? 1 : 0;
    video_format_params_.min_num_decode_surfaces = dec_buf_pool_size_;
    video_format_params_.coded_width = pic_width_;
    video_format_params_.coded_height = pic_height_;
    video_format_params_.chroma_format = static_cast<rocDecVideoChromaFormat>(p_sps->chroma_format_idc);

    // Table 6-1
    int sub_width_c, sub_height_c;
    switch (video_format_params_.chroma_format) {
        case rocDecVideoChromaFormat_Monochrome: {
            sub_width_c = 0; // not defined
            sub_height_c = 0; // not defined
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
            if (p_sps->separate_colour_plane_flag) {
                sub_width_c = 0; // not defined
                sub_height_c = 0; // not defined
            } else {
                sub_width_c = 1;
                sub_height_c = 1;
            }
            break;
        }
        default:
            logger_.ErrorLog(MakeMsg(STR("Error: Sequence Callback function - Chroma Format is not supported")));
            return PARSER_FAIL;
    }
    int chroma_array_type = p_sps->separate_colour_plane_flag ? 0 : p_sps->chroma_format_idc;
    int crop_unit_x, crop_unit_y;
    if (chroma_array_type == 0) {
        crop_unit_x = 1; // (7-19)
        crop_unit_y = 2 - p_sps->frame_mbs_only_flag; // (7-20)
    } else {
        crop_unit_x = sub_width_c; // (7-21)
        crop_unit_y = sub_height_c * (2 - p_sps->frame_mbs_only_flag); // (7-22)
    }
    if (p_sps->frame_cropping_flag) {
        video_format_params_.display_area.left = crop_unit_x * p_sps->frame_crop_left_offset;
        video_format_params_.display_area.top = crop_unit_y * p_sps->frame_crop_top_offset;
        video_format_params_.display_area.right = pic_width_ - (crop_unit_x * p_sps->frame_crop_right_offset);
        video_format_params_.display_area.bottom = pic_height_ - (crop_unit_y * p_sps->frame_crop_bottom_offset);
    }  else { // default values
        video_format_params_.display_area.left = 0;
        video_format_params_.display_area.top = 0;
        video_format_params_.display_area.right = pic_width_;
        video_format_params_.display_area.bottom = pic_height_;
    }

    video_format_params_.bitrate = 0;

    // Dispaly aspect ratio
    // Table E-1.
    static const Rational avc_sar[] = {
        {0, 0}, // unspecified
        {1, 1}, {12, 11}, {10, 11}, {16, 11}, {40, 33}, {24, 11}, {20, 11}, {32, 11},
        {80, 33}, {18, 11}, {15, 11}, {64, 33}, {160, 99}, {4, 3}, {3, 2}, {2, 1},
    };
    Rational sar;
    sar.numerator = 1; // set to square pixel if not present or unspecified
    sar.denominator = 1; // set to square pixel if not present or unspecified
    if (p_sps->vui_parameters_present_flag) {
        if (p_sps->vui_seq_parameters.aspect_ratio_info_present_flag) {
            if (p_sps->vui_seq_parameters.aspect_ratio_idc == 255 /*Extended_SAR*/) {
                sar.numerator = p_sps->vui_seq_parameters.sar_width;
                sar.denominator = p_sps->vui_seq_parameters.sar_height;
            } else if (p_sps->vui_seq_parameters.aspect_ratio_idc > 0 && p_sps->vui_seq_parameters.aspect_ratio_idc < 17) {
                sar = avc_sar[p_sps->vui_seq_parameters.aspect_ratio_idc];
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
    if (p_sps->vui_parameters_present_flag) {
        video_format_params_.video_signal_description.video_format = p_sps->vui_seq_parameters.video_format;
        video_format_params_.video_signal_description.video_full_range_flag = p_sps->vui_seq_parameters.video_full_range_flag;
        video_format_params_.video_signal_description.color_primaries = p_sps->vui_seq_parameters.colour_primaries;
        video_format_params_.video_signal_description.transfer_characteristics = p_sps->vui_seq_parameters.transfer_characteristics;
        video_format_params_.video_signal_description.matrix_coefficients = p_sps->vui_seq_parameters.matrix_coefficients;
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

void AvcVideoParser::SendSeiMsgPayload() {
    sei_message_info_params_.sei_message_count = sei_message_count_;
    sei_message_info_params_.sei_message = sei_message_list_.data();
    sei_message_info_params_.sei_data = (void*)sei_payload_buf_;
    sei_message_info_params_.picIdx = curr_pic_.dec_buf_idx;

    // callback function with RocdecSeiMessageInfo params filled out
    if (pfn_get_sei_message_cb_) pfn_get_sei_message_cb_(parser_params_.user_data, &sei_message_info_params_);
}

static const int diag_scan_4x4[16] = {
    0, 1, 4, 8,
    5, 2, 3, 6,
    9, 12, 13, 10,
    7, 11, 14, 15
};

static const int diag_scan_8x8[64] = {
    0,  1,  8,  16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

ParserResult AvcVideoParser::SendPicForDecode() {
    int i, j;
    AvcSeqParameterSet *p_sps = &sps_list_[active_sps_id_];
    AvcPicParameterSet *p_pps = &pps_list_[active_pps_id_];
    AvcSliceHeader *p_slice_header = &slice_info_list_[0].slice_header;
    dec_pic_params_ = {0};

    dec_pic_params_.pic_width = pic_width_;
    dec_pic_params_.pic_height = pic_height_;
    dec_pic_params_.curr_pic_idx = curr_pic_.dec_buf_idx;
    dec_pic_params_.field_pic_flag = p_slice_header->field_pic_flag;
    dec_pic_params_.bottom_field_flag = p_slice_header->bottom_field_flag;
    dec_pic_params_.second_field = second_field_;

    dec_pic_params_.bitstream_data_len = pic_stream_data_size_;
    dec_pic_params_.bitstream_data = pic_stream_data_ptr_;
    dec_pic_params_.num_slices = num_slices_;

    dec_pic_params_.ref_pic_flag = slice_nal_unit_header_.nal_ref_idc;
    dec_pic_params_.intra_pic_flag = p_slice_header->slice_type == kAvcSliceTypeI || p_slice_header->slice_type == kAvcSliceTypeI_7 || p_slice_header->slice_type == kAvcSliceTypeSI || p_slice_header->slice_type == kAvcSliceTypeSI_9;

    // Set up the picture parameter buffer
    RocdecAvcPicParams *p_pic_param = &dec_pic_params_.pic_params.avc;

    // Current picture
    p_pic_param->curr_pic.pic_idx = curr_pic_.dec_buf_idx;
    if (curr_pic_.is_reference == kUsedForLongTerm) {
        p_pic_param->curr_pic.frame_idx = curr_pic_.long_term_pic_num;
    } else {
        p_pic_param->curr_pic.frame_idx = curr_pic_.frame_num;
    }
    p_pic_param->curr_pic.flags = 0;
    if (curr_pic_.pic_structure != kFrame) {
        p_pic_param->curr_pic.flags |= curr_pic_.pic_structure == kBottomField ? RocdecAvcPicture_FLAGS_BOTTOM_FIELD : RocdecAvcPicture_FLAGS_TOP_FIELD;
    }
    if (curr_pic_.is_reference != kUnusedForReference) {
        p_pic_param->curr_pic.flags |= curr_pic_.is_reference == kUsedForShortTerm ? RocdecAvcPicture_FLAGS_SHORT_TERM_REFERENCE : RocdecAvcPicture_FLAGS_LONG_TERM_REFERENCE;
    }
    p_pic_param->curr_pic.top_field_order_cnt = curr_pic_.top_field_order_cnt;
    p_pic_param->curr_pic.bottom_field_order_cnt = curr_pic_.bottom_field_order_cnt;

    // Reference pictures
    int buf_index = 0;
    for (i = 0; i < AVC_MAX_DPB_FRAMES; i++) {
        if (p_slice_header->field_pic_flag == 0) { // frame picture
            AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
            if (p_ref_pic->is_reference != kUnusedForReference) {
                p_pic_param->ref_frames[buf_index].pic_idx = p_ref_pic->dec_buf_idx;
                if ( p_ref_pic->is_reference == kUsedForLongTerm) {
                    p_pic_param->ref_frames[buf_index].frame_idx = p_ref_pic->long_term_pic_num;
                } else {
                    p_pic_param->ref_frames[buf_index].frame_idx = p_ref_pic->frame_num;
                }
                p_pic_param->ref_frames[buf_index].flags = 0;
                if (p_ref_pic->pic_structure != kFrame) {
                    p_pic_param->ref_frames[buf_index].flags |= p_ref_pic->pic_structure == kBottomField ? RocdecAvcPicture_FLAGS_BOTTOM_FIELD : RocdecAvcPicture_FLAGS_TOP_FIELD;
                }
                p_pic_param->ref_frames[buf_index].flags |= p_ref_pic->is_reference == kUsedForShortTerm ? RocdecAvcPicture_FLAGS_SHORT_TERM_REFERENCE : RocdecAvcPicture_FLAGS_LONG_TERM_REFERENCE;
                p_pic_param->ref_frames[buf_index].top_field_order_cnt = p_ref_pic->top_field_order_cnt;
                p_pic_param->ref_frames[buf_index].bottom_field_order_cnt = p_ref_pic->bottom_field_order_cnt;
                buf_index++;
            }
        } else { // field picture
            bool found = false;
            RocdecAvcPicture *p_ref_frame = &p_pic_param->ref_frames[buf_index];
            for (int j = 0; j < 2; j++) {
                AvcPicture *p_field_pic = &dpb_buffer_.field_pic_list[i * 2 + j];
                if (p_field_pic->is_reference != kUnusedForReference) {
                    p_ref_frame->pic_idx = p_field_pic->dec_buf_idx;
                    if (p_field_pic->is_reference == kUsedForLongTerm) {
                        p_ref_frame->frame_idx = p_field_pic->long_term_frame_idx;
                    } else {
                        p_ref_frame->frame_idx = p_field_pic->frame_num;
                    }
                    p_ref_frame->flags |= p_field_pic->pic_structure == kBottomField ? RocdecAvcPicture_FLAGS_BOTTOM_FIELD : RocdecAvcPicture_FLAGS_TOP_FIELD;
                    p_ref_frame->flags |= p_field_pic->is_reference == kUsedForShortTerm ? RocdecAvcPicture_FLAGS_SHORT_TERM_REFERENCE : RocdecAvcPicture_FLAGS_LONG_TERM_REFERENCE;
                    if (p_field_pic->pic_structure == kTopField) {
                        p_ref_frame->top_field_order_cnt = p_field_pic->top_field_order_cnt;
                    } else {
                        p_ref_frame->bottom_field_order_cnt = p_field_pic->bottom_field_order_cnt;
                    }
                    found = true;
                }
            }
            if (found) {
                buf_index++;
            }
        }
    }

    for (i = buf_index; i < 16; i++) {
        p_pic_param->ref_frames[i].pic_idx = 0xFF;
        p_pic_param->ref_frames[i].frame_idx = 0;
        p_pic_param->ref_frames[i].flags = RocdecAvcPicture_FLAGS_INVALID;
        p_pic_param->ref_frames[i].top_field_order_cnt = 0;
        p_pic_param->ref_frames[i].bottom_field_order_cnt = 0;
    }

    p_pic_param->picture_width_in_mbs_minus1 = p_sps->pic_width_in_mbs_minus1;
    p_pic_param->picture_height_in_mbs_minus1 = (2 - p_sps->frame_mbs_only_flag) * (p_sps->pic_height_in_map_units_minus1 + 1) - 1;
    p_pic_param->bit_depth_luma_minus8 = p_sps->bit_depth_luma_minus8;
    p_pic_param->bit_depth_chroma_minus8 = p_sps->bit_depth_chroma_minus8;
    p_pic_param->num_ref_frames = p_sps->max_num_ref_frames;

    p_pic_param->seq_fields.bits.chroma_format_idc = p_sps->chroma_format_idc;
    p_pic_param->seq_fields.bits.residual_colour_transform_flag = p_sps->separate_colour_plane_flag;
    p_pic_param->seq_fields.bits.gaps_in_frame_num_value_allowed_flag = p_sps->gaps_in_frame_num_value_allowed_flag;
    p_pic_param->seq_fields.bits.frame_mbs_only_flag = p_sps->frame_mbs_only_flag;
    p_pic_param->seq_fields.bits.mb_adaptive_frame_field_flag = p_sps->mb_adaptive_frame_field_flag;
    p_pic_param->seq_fields.bits.direct_8x8_inference_flag = p_sps->direct_8x8_inference_flag;
    p_pic_param->seq_fields.bits.MinLumaBiPredSize8x8 = p_sps->level_idc >= 31;  // A.3.3.2
    p_pic_param->seq_fields.bits.log2_max_frame_num_minus4 = p_sps->log2_max_frame_num_minus4;
    p_pic_param->seq_fields.bits.pic_order_cnt_type = p_sps->pic_order_cnt_type;
    p_pic_param->seq_fields.bits.log2_max_pic_order_cnt_lsb_minus4 = p_sps->log2_max_pic_order_cnt_lsb_minus4;
    p_pic_param->seq_fields.bits.delta_pic_order_always_zero_flag = p_sps->delta_pic_order_always_zero_flag;

    p_pic_param->pic_init_qp_minus26 = p_pps->pic_init_qp_minus26;
    p_pic_param->pic_init_qs_minus26 = p_pps->pic_init_qs_minus26;
    p_pic_param->chroma_qp_index_offset = p_pps->chroma_qp_index_offset;
    p_pic_param->second_chroma_qp_index_offset = p_pps->second_chroma_qp_index_offset;

    p_pic_param->pic_fields.bits.entropy_coding_mode_flag = p_pps->entropy_coding_mode_flag;
    p_pic_param->pic_fields.bits.weighted_pred_flag = p_pps->weighted_pred_flag;
    p_pic_param->pic_fields.bits.weighted_bipred_idc = p_pps->weighted_bipred_idc;
    p_pic_param->pic_fields.bits.transform_8x8_mode_flag = p_pps->transform_8x8_mode_flag;
    p_pic_param->pic_fields.bits.field_pic_flag = p_slice_header->field_pic_flag;
    p_pic_param->pic_fields.bits.constrained_intra_pred_flag = p_pps->constrained_intra_pred_flag;
    p_pic_param->pic_fields.bits.pic_order_present_flag = p_pps->bottom_field_pic_order_in_frame_present_flag;
    p_pic_param->pic_fields.bits.deblocking_filter_control_present_flag = p_pps->deblocking_filter_control_present_flag;
    p_pic_param->pic_fields.bits.redundant_pic_cnt_present_flag = p_pps->redundant_pic_cnt_present_flag;
    p_pic_param->pic_fields.bits.reference_pic_flag = slice_nal_unit_header_.nal_ref_idc != 0;

    p_pic_param->frame_num = p_slice_header->frame_num;

    // Set up slice parameters
    // Resize if needed
    if (num_slices_ > slice_param_list_.size()) {
        slice_param_list_.resize(num_slices_, {0});
    }
    for (int slice_index = 0; slice_index < num_slices_; slice_index++) {
        RocdecAvcSliceParams *p_slice_param = &slice_param_list_[slice_index];
        AvcSliceInfo *p_slice_info = &slice_info_list_[slice_index];
        AvcSliceHeader *p_slice_header = &p_slice_info->slice_header;

        p_slice_param->slice_data_size = p_slice_info->slice_data_size;
        p_slice_param->slice_data_offset = p_slice_info->slice_data_offset;
        p_slice_param->slice_data_flag = 0; // VA_SLICE_DATA_FLAG_ALL;
        p_slice_param->slice_data_bit_offset = 0;
        p_slice_param->first_mb_in_slice = p_slice_header->first_mb_in_slice;
        p_slice_param->slice_type = p_slice_header->slice_type;
        p_slice_param->direct_spatial_mv_pred_flag = p_slice_header->direct_spatial_mv_pred_flag;
        p_slice_param->num_ref_idx_l0_active_minus1 = p_slice_header->num_ref_idx_l0_active_minus1;
        p_slice_param->num_ref_idx_l1_active_minus1 = p_slice_header->num_ref_idx_l1_active_minus1;
        p_slice_param->cabac_init_idc = p_slice_header->cabac_init_idc;
        p_slice_param->slice_qp_delta = p_slice_header->slice_qp_delta;
        p_slice_param->disable_deblocking_filter_idc = p_slice_header->disable_deblocking_filter_idc;
        p_slice_param->slice_alpha_c0_offset_div2 = p_slice_header->slice_alpha_c0_offset_div2;
        p_slice_param->slice_beta_offset_div2 = p_slice_header->slice_beta_offset_div2;
        p_slice_param->luma_log2_weight_denom = p_slice_header->pred_weight_table.luma_log2_weight_denom;
        p_slice_param->chroma_log2_weight_denom = p_slice_header->pred_weight_table.chroma_log2_weight_denom;

        // Ref lists
        for (j = 0; j < 32; j++) {
            p_slice_param->ref_pic_list_0[j].pic_idx = 0xFF;
            p_slice_param->ref_pic_list_1[j].pic_idx = 0xFF;
            p_slice_param->ref_pic_list_0[j].frame_idx = 0;
            p_slice_param->ref_pic_list_1[j].frame_idx = 0;
            p_slice_param->ref_pic_list_0[j].flags = RocdecAvcPicture_FLAGS_INVALID;
            p_slice_param->ref_pic_list_1[j].flags = RocdecAvcPicture_FLAGS_INVALID;
            p_slice_param->ref_pic_list_0[j].top_field_order_cnt = 0;
            p_slice_param->ref_pic_list_1[j].top_field_order_cnt = 0;
            p_slice_param->ref_pic_list_0[j].bottom_field_order_cnt = 0;
            p_slice_param->ref_pic_list_1[j].bottom_field_order_cnt = 0;
        }

        if (p_slice_header->slice_type == kAvcSliceTypeP || p_slice_header->slice_type == kAvcSliceTypeP_5 || p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6) {
            for (i = 0; i <= p_slice_info->slice_header.num_ref_idx_l0_active_minus1; i++) {
                AvcPicture *p_ref_pic = &p_slice_info->ref_list_0_[i];
                if (p_ref_pic->is_reference != kUnusedForReference) {
                    p_slice_param->ref_pic_list_0[i].pic_idx = p_ref_pic->dec_buf_idx;
                    if ( p_ref_pic->is_reference == kUsedForLongTerm) {
                        p_slice_param->ref_pic_list_0[i].frame_idx = p_ref_pic->long_term_pic_num;
                    } else {
                        p_slice_param->ref_pic_list_0[i].frame_idx = p_ref_pic->frame_num;
                    }
                    p_slice_param->ref_pic_list_0[i].top_field_order_cnt = p_ref_pic->top_field_order_cnt;
                    p_slice_param->ref_pic_list_0[i].bottom_field_order_cnt = p_ref_pic->bottom_field_order_cnt;
                    p_slice_param->ref_pic_list_0[i].flags = 0;
                    if (p_ref_pic->pic_structure != kFrame) {
                        p_slice_param->ref_pic_list_0[i].flags |= p_ref_pic->pic_structure == kBottomField ? RocdecAvcPicture_FLAGS_BOTTOM_FIELD : RocdecAvcPicture_FLAGS_TOP_FIELD;
                    }
                    p_slice_param->ref_pic_list_0[i].flags |= p_ref_pic->is_reference == kUsedForShortTerm ? RocdecAvcPicture_FLAGS_SHORT_TERM_REFERENCE : RocdecAvcPicture_FLAGS_LONG_TERM_REFERENCE;
                }
            }
        }

        if (p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6 ) {
            for (i = 0; i <= p_slice_info->slice_header.num_ref_idx_l1_active_minus1; i++) {
                AvcPicture *p_ref_pic = &p_slice_info->ref_list_1_[i];
                if (p_ref_pic->is_reference != kUnusedForReference) {
                    p_slice_param->ref_pic_list_1[i].pic_idx = p_ref_pic->dec_buf_idx;
                    if ( p_ref_pic->is_reference == kUsedForLongTerm) {
                        p_slice_param->ref_pic_list_1[i].frame_idx = p_ref_pic->long_term_pic_num;
                    } else {
                        p_slice_param->ref_pic_list_1[i].frame_idx = p_ref_pic->frame_num;
                    }
                    p_slice_param->ref_pic_list_1[i].top_field_order_cnt = p_ref_pic->top_field_order_cnt;
                    p_slice_param->ref_pic_list_1[i].bottom_field_order_cnt = p_ref_pic->bottom_field_order_cnt;
                    p_slice_param->ref_pic_list_1[i].flags = 0;
                    if (p_ref_pic->pic_structure != kFrame) {
                        p_slice_param->ref_pic_list_1[i].flags |= p_ref_pic->pic_structure == kBottomField ? RocdecAvcPicture_FLAGS_BOTTOM_FIELD : RocdecAvcPicture_FLAGS_TOP_FIELD;
                    }
                    p_slice_param->ref_pic_list_1[i].flags |= p_ref_pic->is_reference == kUsedForShortTerm ? RocdecAvcPicture_FLAGS_SHORT_TERM_REFERENCE : RocdecAvcPicture_FLAGS_LONG_TERM_REFERENCE;
                }
            }
        }

        // Prediction weight table
        // Note luma_weight_l0_flag should be an array. Set it using the first one in the table.
        p_slice_param->luma_weight_l0_flag = p_slice_header->pred_weight_table.weight_factor[0].luma_weight_l0_flag;
        for (i = 0; i <= p_slice_header->num_ref_idx_l0_active_minus1; i++) {
            p_slice_param->luma_weight_l0[i] = p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l0;
            p_slice_param->luma_offset_l0[i] = p_slice_header->pred_weight_table.weight_factor[i].luma_offset_l0;
        }

        // Note chroma_weight_l0_flag should be an array. Set it using the first one in the table.
        p_slice_param->chroma_weight_l0_flag = p_slice_header->pred_weight_table.weight_factor[0].chroma_weight_l0_flag;
        for (i = 0; i <= p_slice_header->num_ref_idx_l0_active_minus1; i++) {
            for (j = 0; j < 2; j++) {
                p_slice_param->chroma_weight_l0[i][j] = p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l0[j];
                p_slice_param->chroma_offset_l0[i][j] = p_slice_header->pred_weight_table.weight_factor[i].chroma_offset_l0[j];
            }
        }
        if (p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6 ) {
            // Note luma_weight_l1_flag should be an array. Set it using the first one in the table.
            p_slice_param->luma_weight_l1_flag = p_slice_header->pred_weight_table.weight_factor[0].luma_weight_l1_flag;
            for (i = 0; i <= p_slice_header->num_ref_idx_l1_active_minus1; i++) {
                p_slice_param->luma_weight_l1[i] = p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l1;
                p_slice_param->luma_offset_l1[i] = p_slice_header->pred_weight_table.weight_factor[i].luma_offset_l1;
            }
            // Note chroma_weight_l0_flag should be an array. Set it using the first one in the table.
            p_slice_param->chroma_weight_l1_flag = p_slice_header->pred_weight_table.weight_factor[0].chroma_weight_l1_flag;
            for (i = 0; i <= p_slice_header->num_ref_idx_l1_active_minus1; i++) {
                for (j = 0; j < 2; j++) {
                    p_slice_param->chroma_weight_l1[i][j] = p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l1[j];
                    p_slice_param->chroma_offset_l1[i][j] = p_slice_header->pred_weight_table.weight_factor[i].chroma_offset_l1[j];
                }
            }
        }
    }
    dec_pic_params_.slice_params.avc = slice_param_list_.data();

    // Set up scaling lists
    RocdecAvcIQMatrix *p_iq_matrix = &dec_pic_params_.iq_matrix.avc;
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 16; j++) {
            p_iq_matrix->scaling_list_4x4[i][diag_scan_4x4[j]] = p_pps->scaling_list_4x4[i][j];
        }
    }
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 64; j++) {
            p_iq_matrix->scaling_list_8x8[i][diag_scan_8x8[j]] = p_pps->scaling_list_8x8[i][j];
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

AvcNalUnitHeader AvcVideoParser::ParseNalUnitHeader(uint8_t header_byte) {
    size_t bit_offset = 0;
    AvcNalUnitHeader nal_header;

    nal_header.forbidden_zero_bit = Parser::GetBit(&header_byte, bit_offset);
    nal_header.nal_ref_idc = Parser::ReadBits(&header_byte, bit_offset, 2);
    nal_header.nal_unit_type = Parser::ReadBits(&header_byte, bit_offset, 5);
    return nal_header;
}

const int Flat_4x4_16[16] = {
    16,16,16,16,
    16,16,16,16,
    16,16,16,16,
    16,16,16,16
};

const int Flat_8x8_16[64] = {
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16
};

const int Default_4x4_Intra[16] = {
    6, 13, 13, 20,
    20, 20, 28, 28,
    28, 28, 32, 32,
    32, 37, 37, 42
};

const int Default_4x4_Inter[16] = {
    10, 14, 14, 20,
    20, 20, 24, 24,
    24, 24, 27, 27,
    27, 30, 30, 34
};

const int Default_8x8_Intra[64] = {
    6, 10, 10, 13, 11, 13, 16, 16,
    16, 16, 18, 18, 18, 18, 18, 23,
    23, 23, 23, 23, 23, 25, 25, 25,
    25, 25, 25, 25, 27, 27, 27, 27,
    27, 27, 27, 27, 29, 29, 29, 29,
    29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36,
    36, 36, 38, 38, 38, 40, 40, 42
};

const int Default_8x8_Inter[64] = {
    9, 13, 13, 15, 13, 15, 17, 17,
    17, 17, 19, 19, 19, 19, 19, 21,
    21, 21, 21, 21, 21, 22, 22, 22,
    22, 22, 22, 22, 24, 24, 24, 24,
    24, 24, 24, 24, 25, 25, 25, 25,
    25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30,
    30, 30, 32, 32, 32, 33, 33, 35
};

ParserResult AvcVideoParser::ParseSps(uint8_t *p_stream, size_t size) {
    size_t offset = 0;  // current bit offset
    AvcSeqParameterSet *p_sps = nullptr;

    // Parse and temporarily store till set id
    uint32_t profile_idc = Parser::ReadBits(p_stream, offset, 8);
    uint32_t constraint_set0_flag = Parser::GetBit(p_stream, offset);
    uint32_t constraint_set1_flag = Parser::GetBit(p_stream, offset);
    uint32_t constraint_set2_flag = Parser::GetBit(p_stream, offset);
    uint32_t constraint_set3_flag = Parser::GetBit(p_stream, offset);
    uint32_t constraint_set4_flag = Parser::GetBit(p_stream, offset);
    uint32_t constraint_set5_flag = Parser::GetBit(p_stream, offset);
    uint32_t reserved_zero_2bits = Parser::ReadBits(p_stream, offset, 2);
    uint32_t level_idc = Parser::ReadBits(p_stream, offset, 8);
    uint32_t seq_parameter_set_id = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("seq_parameter_set_id", seq_parameter_set_id, 0, 31);

    p_sps = &sps_list_[seq_parameter_set_id];
    memset(p_sps, 0, sizeof(AvcSeqParameterSet));

    p_sps->profile_idc = profile_idc;
    p_sps->constraint_set0_flag = constraint_set0_flag;
    p_sps->constraint_set1_flag = constraint_set1_flag;
    p_sps->constraint_set2_flag = constraint_set2_flag;
    p_sps->constraint_set3_flag = constraint_set3_flag;
    p_sps->constraint_set4_flag = constraint_set4_flag;
    p_sps->constraint_set5_flag = constraint_set5_flag;
    p_sps->reserved_zero_2bits = reserved_zero_2bits;
    p_sps->level_idc = level_idc;
    p_sps->seq_parameter_set_id = seq_parameter_set_id;

    if (p_sps->profile_idc == 100 || 
        p_sps->profile_idc == 110 ||
        p_sps->profile_idc == 122 || 
        p_sps->profile_idc == 244 ||
        p_sps->profile_idc == 44  ||
        p_sps->profile_idc == 83  ||
        p_sps->profile_idc == 86  ||
        p_sps->profile_idc == 118 ||
        p_sps->profile_idc == 128 ||
        p_sps->profile_idc == 138 ||
        p_sps->profile_idc == 139 ||
        p_sps->profile_idc == 134 ||
        p_sps->profile_idc == 135) {
        p_sps->chroma_format_idc = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("chroma_format_idc", p_sps->chroma_format_idc, 0, 3);
        if (p_sps->chroma_format_idc == 3) {
            p_sps->separate_colour_plane_flag = Parser::GetBit(p_stream, offset);
        }
        
        p_sps->bit_depth_luma_minus8 = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("bit_depth_luma_minus8", p_sps->bit_depth_luma_minus8, 0, 0);
        p_sps->bit_depth_chroma_minus8 = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("bit_depth_chroma_minus8", p_sps->bit_depth_chroma_minus8, 0, 0);
        p_sps->qpprime_y_zero_transform_bypass_flag = Parser::GetBit(p_stream, offset);
        p_sps->seq_scaling_matrix_present_flag = Parser::GetBit(p_stream, offset);
        if (p_sps->seq_scaling_matrix_present_flag == 1) {
            for (int i = 0; i < ((p_sps->chroma_format_idc != 3) ? 8 : 12); i++) {
                p_sps->seq_scaling_list_present_flag[i] = Parser::GetBit(p_stream, offset);
                if (p_sps->seq_scaling_list_present_flag[i] == 1) {
                    ParserResult ret;
                    if ( i < 6 ) {
                        if ((ret = GetScalingList(p_stream, offset, p_sps->scaling_list_4x4[i], 16, &p_sps->use_default_scaling_matrix_4x4_flag[i])) != PARSER_OK) {
                            return ret;
                        }
                    } else {
                        if ((ret = GetScalingList(p_stream, offset, p_sps->scaling_list_8x8[i - 6], 64, &p_sps->use_default_scaling_matrix_8x8_flag[i - 6])) != PARSER_OK) {
                            return ret;
                        }
                    }
                }
            }
        }
    } else {
        p_sps->chroma_format_idc = 1;
    }

    // Setup default scaling list if needed
    if (p_sps->seq_scaling_matrix_present_flag == 0) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 16; j++) {
                p_sps->scaling_list_4x4[i][j] = Flat_4x4_16[j];
            }
        }
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 64; j++) {
                p_sps->scaling_list_8x8[i][j] = Flat_8x8_16[j];
            }
        }
    } else {
        // 4 x 4
        for (int i = 0; i < 6; i++) {
            if (!p_sps->seq_scaling_list_present_flag[i]) { // fall back rule set A
                if (i == 0) {
                    for (int j = 0; j < 16; j++) {
                        p_sps->scaling_list_4x4[i][j] = Default_4x4_Intra[j];
                    }
                } else if (i == 3) {
                    for (int j = 0; j < 16; j++) {
                        p_sps->scaling_list_4x4[i][j] = Default_4x4_Inter[j];
                    }
                } else {
                    for (int j = 0; j < 16; j++) {
                        p_sps->scaling_list_4x4[i][j] = p_sps->scaling_list_4x4[i - 1][j];
                    }
                }
            } else {
                if (p_sps->use_default_scaling_matrix_4x4_flag[i]) {
                    if (i < 3) {
                        for (int j = 0; j < 16; j++) {
                            p_sps->scaling_list_4x4[i][j] = Default_4x4_Intra[j];
                        }
                    } else {
                        for (int j = 0; j < 16; j++) {
                            p_sps->scaling_list_4x4[i][j] = Default_4x4_Inter[j];
                        }
                    }
                }
            }
        }
        
        // 8 x 8
        for (int i = 0; i < 6; i++) {
            if(!p_sps->seq_scaling_list_present_flag[i + 6]) { // fall back rule set A
                if (i == 0) {
                    for (int j = 0; j < 64; j++) {
                        p_sps->scaling_list_8x8[i][j] = Default_8x8_Intra[j];
                    }
                } else if (i == 1) {
                    for (int j = 0; j < 64; j++) {
                        p_sps->scaling_list_8x8[i][j] = Default_8x8_Inter[j];
                    }
                } else {
                    for (int j = 0; j < 64; j++) {
                        p_sps->scaling_list_8x8[i][j] = p_sps->scaling_list_8x8[i - 2][j];
                    }
                }
            } else {
                if (p_sps->use_default_scaling_matrix_8x8_flag[i]) {
                    if (i % 2 == 0) {
                        for (int j = 0; j < 64; j++) {
                            p_sps->scaling_list_8x8[i][j] = Default_8x8_Intra[j];
                        }
                    } else {
                        for (int j = 0; j < 64; j++) {
                            p_sps->scaling_list_8x8[i][j] = Default_8x8_Inter[j];
                        }
                    }
                }
            }
        }
    }

    p_sps->log2_max_frame_num_minus4 = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("log2_max_frame_num_minus4", p_sps->log2_max_frame_num_minus4, 0, 12);
    p_sps->pic_order_cnt_type = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("pic_order_cnt_type", p_sps->pic_order_cnt_type, 0, 2);
    if (p_sps->pic_order_cnt_type == 0 ) {
        p_sps->log2_max_pic_order_cnt_lsb_minus4 = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("log2_max_pic_order_cnt_lsb_minus4", p_sps->log2_max_pic_order_cnt_lsb_minus4, 0, 12);
    } else if (p_sps->pic_order_cnt_type == 1) {
        p_sps->delta_pic_order_always_zero_flag = Parser::GetBit(p_stream, offset);
        p_sps->offset_for_non_ref_pic = Parser::ExpGolomb::ReadSe(p_stream, offset);
        p_sps->offset_for_top_to_bottom_field = Parser::ExpGolomb::ReadSe(p_stream, offset);
        p_sps->num_ref_frames_in_pic_order_cnt_cycle = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("num_ref_frames_in_pic_order_cnt_cycle", p_sps->num_ref_frames_in_pic_order_cnt_cycle, 0, 255);
        for (int i = 0; i < p_sps->num_ref_frames_in_pic_order_cnt_cycle; i++) {
            p_sps->offset_for_ref_frame[i] = Parser::ExpGolomb::ReadSe(p_stream, offset);
        }
    }

    p_sps->max_num_ref_frames = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("max_num_ref_frames", p_sps->max_num_ref_frames, 0, AVC_MAX_DPB_FRAMES - 1);
    p_sps->gaps_in_frame_num_value_allowed_flag = Parser::GetBit(p_stream, offset);
    p_sps->pic_width_in_mbs_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
    p_sps->pic_height_in_map_units_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
    p_sps->frame_mbs_only_flag = Parser::GetBit(p_stream, offset);
    if (!p_sps->frame_mbs_only_flag) {
        p_sps->mb_adaptive_frame_field_flag = Parser::GetBit(p_stream, offset);
    }

    p_sps->direct_8x8_inference_flag = Parser::GetBit(p_stream, offset);
    p_sps->frame_cropping_flag = Parser::GetBit(p_stream, offset);
    if (p_sps->frame_cropping_flag) {
        int chroma_array_type = p_sps->separate_colour_plane_flag == 0 ? p_sps->chroma_format_idc : 0;
        int sub_width_c = 0, sub_height_c = 0;
        switch (p_sps->chroma_format_idc) {
            case 1:
                sub_width_c = 2;
                sub_height_c = 2;
                break;
            case 2:
                sub_width_c = 2;
                sub_height_c = 1;
                break;
            case 3:
                sub_width_c = 1;
                sub_height_c = 1;
                break;
            default:
                sub_width_c = 0;
                sub_height_c = 0;
                break;
        }
        int crop_unit_x, crop_unit_y;
        if (chroma_array_type == 0) {
            crop_unit_x = 1;
            crop_unit_y = 2 - p_sps->frame_mbs_only_flag;
        } else {
            crop_unit_x = sub_width_c;
            crop_unit_y = sub_height_c * (2 - p_sps->frame_mbs_only_flag);
        }
        uint32_t pic_width_in_samples_l = (p_sps->pic_width_in_mbs_minus1 + 1) * AVC_MACRO_BLOCK_SIZE;
        uint32_t frame_height_in_samples_l = (2 - p_sps->frame_mbs_only_flag) * (p_sps->pic_height_in_map_units_minus1 + 1) * AVC_MACRO_BLOCK_SIZE;
        p_sps->frame_crop_left_offset = Parser::ExpGolomb::ReadUe(p_stream, offset);
        p_sps->frame_crop_right_offset = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_MAX("frame_crop_left_offset + frame_crop_right_offset", p_sps->frame_crop_left_offset + p_sps->frame_crop_right_offset + 1, pic_width_in_samples_l / crop_unit_x);
        p_sps->frame_crop_top_offset = Parser::ExpGolomb::ReadUe(p_stream, offset);
        p_sps->frame_crop_bottom_offset = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_MAX("frame_crop_top_offset + frame_crop_bottom_offset", p_sps->frame_crop_top_offset + p_sps->frame_crop_bottom_offset + 1, frame_height_in_samples_l / crop_unit_y);
    }

    p_sps->vui_parameters_present_flag = Parser::GetBit(p_stream, offset);
    if (p_sps->vui_parameters_present_flag == 1) {
        ParserResult ret;
        if ((ret = GetVuiParameters(p_stream, offset, &p_sps->vui_seq_parameters)) != PARSER_OK) {
            return ret;
        }
    }

    p_sps->is_received = 1;  // confirm SPS with seq_parameter_set_id received (but not activated)

#if DBGINFO
    PrintSps(p_sps);
#endif // DBGINFO
    return PARSER_OK;
}

ParserResult AvcVideoParser::ParsePps(uint8_t *p_stream, size_t stream_size_in_byte) {
    AvcSeqParameterSet *p_sps = nullptr;
    AvcPicParameterSet *p_pps = nullptr;
    size_t offset = 0; // current bit offset

    // Parse and temporarily store
    uint32_t pic_parameter_set_id = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("pic_parameter_set_id", pic_parameter_set_id, 0, 255);
    uint32_t seq_parameter_set_id = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("seq_parameter_set_id", seq_parameter_set_id, 0, 31);
    if (sps_list_[seq_parameter_set_id].is_received == 0) {
        logger_.ErrorLog(MakeMsg("Empty SPS is referred."));
        return PARSER_WRONG_STATE;
    }
    p_sps = &sps_list_[seq_parameter_set_id];
    p_pps = &pps_list_[pic_parameter_set_id];
    memset(p_pps, 0, sizeof(AvcPicParameterSet));	

    p_pps->pic_parameter_set_id = pic_parameter_set_id;
    p_pps->seq_parameter_set_id = seq_parameter_set_id;
    p_pps->entropy_coding_mode_flag = Parser::GetBit(p_stream, offset);
    p_pps->bottom_field_pic_order_in_frame_present_flag = Parser::GetBit(p_stream, offset);

    p_pps->num_slice_groups_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
    if (p_pps->num_slice_groups_minus1 > 0) {
        // Note: VCN supports High Profile only (num_slice_groups_minus1 = 0)
        logger_.ErrorLog(MakeMsg("Multiple slice groups are not supported"));
        return PARSER_NOT_SUPPORTED;
    }

    p_pps->num_ref_idx_l0_default_active_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("num_ref_idx_l0_default_active_minus1", p_pps->num_ref_idx_l0_default_active_minus1, 0, 31);
    p_pps->num_ref_idx_l1_default_active_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("num_ref_idx_l1_default_active_minus1", p_pps->num_ref_idx_l1_default_active_minus1, 0, 31);
    p_pps->weighted_pred_flag = Parser::GetBit(p_stream, offset);
    p_pps->weighted_bipred_idc = Parser::ReadBits(p_stream, offset, 2);
    p_pps->pic_init_qp_minus26 = Parser::ExpGolomb::ReadSe(p_stream, offset);
    CHECK_ALLOWED_RANGE("pic_init_qp_minus26", p_pps->pic_init_qp_minus26, static_cast<int>(-(26 + 6 * p_sps->bit_depth_luma_minus8)), 25);
    p_pps->pic_init_qs_minus26 = Parser::ExpGolomb::ReadSe(p_stream, offset);
    CHECK_ALLOWED_RANGE("pic_init_qs_minus26", p_pps->pic_init_qs_minus26, -26, 25);
    p_pps->chroma_qp_index_offset = Parser::ExpGolomb::ReadSe(p_stream, offset);
    CHECK_ALLOWED_RANGE("chroma_qp_index_offset", p_pps->chroma_qp_index_offset, -12, 12);
    p_pps->deblocking_filter_control_present_flag = Parser::GetBit(p_stream, offset);
    p_pps->constrained_intra_pred_flag = Parser::GetBit(p_stream, offset);
    p_pps->redundant_pic_cnt_present_flag = Parser::GetBit(p_stream, offset);

    if (MoreRbspData(p_stream, stream_size_in_byte, offset)) {
        p_pps->transform_8x8_mode_flag = Parser::GetBit(p_stream, offset);
        p_pps->pic_scaling_matrix_present_flag = Parser::GetBit(p_stream, offset);
        if (p_pps->pic_scaling_matrix_present_flag == 1) {
            int count = p_sps->chroma_format_idc != 3 ? 2 : 6;
            for (int i = 0; i < 6 + count * p_pps->transform_8x8_mode_flag; i++) {
                p_pps->pic_scaling_list_present_flag [i] = Parser::GetBit(p_stream, offset);
                if (p_pps->pic_scaling_list_present_flag[i] == 1) {
                    ParserResult ret;
                    if ( i < 6 ) {
                        if ((ret = GetScalingList(p_stream, offset, p_pps->scaling_list_4x4[i], 16, &p_pps->use_default_scaling_matrix_4x4_flag[i])) != PARSER_OK) {
                            return ret;
                        }
                    } else {
                        if ((ret = GetScalingList(p_stream, offset, p_pps->scaling_list_8x8[i - 6], 64, &p_pps->use_default_scaling_matrix_8x8_flag[i - 6])) != PARSER_OK) {
                            return ret;
                        }
                    }
                }
            }
        }
        p_pps->second_chroma_qp_index_offset = Parser::ExpGolomb::ReadSe(p_stream, offset);
        CHECK_ALLOWED_RANGE("second_chroma_qp_index_offset", p_pps->second_chroma_qp_index_offset, -12, 12);
    } else {
        /// When second_chroma_qp_index_offset is not present, it shall be inferred to be equal to chroma_qp_index_offset.
        p_pps->second_chroma_qp_index_offset = p_pps->chroma_qp_index_offset;
    }

    // Setup default scaling list if needed
    if (p_pps->pic_scaling_matrix_present_flag == 0) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 16; j++) {
                p_pps->scaling_list_4x4[i][j] = p_sps->scaling_list_4x4[i][j];
            }
        }
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 64; j++) {
                p_pps->scaling_list_8x8[i][j] = p_sps->scaling_list_8x8[i][j];
            }
        }
    } else {
        // 4 x 4
        for (int i = 0; i < 6; i++) {
            if (p_pps->pic_scaling_list_present_flag[i] == 0) {
                if (i == 0) {
                    if ( p_sps->seq_scaling_matrix_present_flag == 0) { // fall back rule set A
                        for (int j = 0; j < 16; j++) {
                            p_pps->scaling_list_4x4[i][j] = Default_4x4_Intra[j];
                        }
                    } else { // fall back rule set B
                        for (int j = 0; j < 16; j++) {
                            p_pps->scaling_list_4x4[i][j] = p_sps->scaling_list_4x4[i][j];
                        }
                    }
                } else if (i == 3) {
                    if (p_sps->seq_scaling_matrix_present_flag == 0) { // fall back rule set A
                        for (int j = 0; j < 16; j++) {
                            p_pps->scaling_list_4x4[i][j] = Default_4x4_Inter[j];
                        }
                    } else { // fall back rule set B
                        for (int j = 0; j < 16; j++) {
                            p_pps->scaling_list_4x4[i][j] = p_sps->scaling_list_4x4[i][j];
                        }
                    }
                } else {
                    for (int j = 0; j < 16; j++) {
                        p_pps->scaling_list_4x4[i][j] = p_pps->scaling_list_4x4[i - 1][j];
                    }
                }
            } else {
                if (p_pps->use_default_scaling_matrix_4x4_flag[i]) {
                    if( i < 3 ) {
                        for (int j = 0; j < 16; j++) {
                            p_pps->scaling_list_4x4[i][j] = Default_4x4_Intra[j];
                        }
                    } else {
                        for (int j = 0; j < 16; j++) {
                            p_pps->scaling_list_4x4[i][j] = Default_4x4_Inter[j];
                        }
                    }
                }
            }
        }

        // 8 x 8
        for (int i = 0; i < 6; i++) {
            if (p_pps->pic_scaling_list_present_flag[i + 6] == 0) {
                if (i == 0) {
                    if ( p_sps->seq_scaling_matrix_present_flag == 0) { // fall back rule set A
                        for (int j = 0; j < 64; j++) {
                            p_pps->scaling_list_8x8[i][j] = Default_8x8_Intra[j];
                        }
                    } else { // fall back rule set B
                        for (int j = 0; j < 64; j++) {
                            p_pps->scaling_list_8x8[i][j] = p_sps->scaling_list_8x8[i][j];
                        }
                    }
                } else if (i == 1) {
                    if ( p_sps->seq_scaling_matrix_present_flag == 0) { // fall back rule set A
                        for (int j = 0; j < 64; j++) {
                            p_pps->scaling_list_8x8[i][j] = Default_8x8_Inter[j];
                        }
                    } else { // fall back rule set B
                        for (int j = 0; j < 64; j++) {
                            p_pps->scaling_list_8x8[i][j] = p_sps->scaling_list_8x8[i][j];
                        }
                    }
                } else {
                    for (int j = 0; j < 64; j++) {
                        p_pps->scaling_list_8x8[i][j] = p_pps->scaling_list_8x8[i - 2][j];
                    }
                }
            } else {
                if (p_pps->use_default_scaling_matrix_8x8_flag[i]) {
                    if ( i % 2 == 0) {
                        for (int j = 0; j < 64; j++) {
                            p_pps->scaling_list_8x8[i][j] = Default_8x8_Intra[j];
                        }
                    } else {
                        for (int j = 0; j < 64; j++) {
                            p_pps->scaling_list_8x8[i][j] = Default_8x8_Inter[j];
                        }
                    }
                }
            }
        }
    }

    p_pps->is_received = 1;  // confirm PPS with pic_parameter_set_id received (but not activated)

#if DBGINFO
    PrintPps(p_pps);
#endif // DBGINFO
    return PARSER_OK;
}

ParserResult AvcVideoParser::ParseSliceHeader(uint8_t *p_stream, size_t stream_size_in_byte, AvcSliceHeader *p_slice_header) {
    int i;
    size_t offset = 0;  // current bit offset
    AvcSeqParameterSet *p_sps = nullptr;
    AvcPicParameterSet *p_pps = nullptr;

    curr_has_mmco_5_ = 0;
    memset(p_slice_header, 0, sizeof(AvcSliceHeader));

    p_slice_header->first_mb_in_slice = Parser::ExpGolomb::ReadUe(p_stream, offset); // range check below
    p_slice_header->slice_type = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("slice_type", p_slice_header->slice_type, 0, 9);

    // Set active SPS and PPS for the current slice
    int32_t active_pps_id = Parser::ExpGolomb::ReadUe(p_stream, offset);
    CHECK_ALLOWED_RANGE("pic_parameter_set_id", active_pps_id, 0, 255);
    if (pps_list_[active_pps_id].is_received == 0) {
        logger_.ErrorLog(MakeMsg("Empty PPS is referred."));
        return PARSER_WRONG_STATE;
    }
    active_pps_id_ = active_pps_id;
    p_slice_header->pic_parameter_set_id = active_pps_id;
    p_pps = &pps_list_[active_pps_id_];

    int32_t active_sps_id = p_pps->seq_parameter_set_id;
    if (sps_list_[active_sps_id].is_received == 0) {
        logger_.ErrorLog(MakeMsg("Empty SPS is referred."));
        return PARSER_WRONG_STATE;
    }
    if (active_sps_id_ != p_pps->seq_parameter_set_id) {
        active_sps_id_ = p_pps->seq_parameter_set_id;
        p_sps = &sps_list_[active_sps_id_];
        if ( p_sps->is_received == 0) {
            logger_.ErrorLog(MakeMsg("Empty SPS is referred."));
            return PARSER_WRONG_STATE;
        }
        new_seq_activated_ = true;  // Note: clear this flag after the actions are taken.
    }
    p_sps = &sps_list_[active_sps_id_];

    // Check video dimension change
    uint32_t curr_pic_width = (p_sps->pic_width_in_mbs_minus1 + 1) * AVC_MACRO_BLOCK_SIZE;
    uint32_t curr_pic_height = (2 - p_sps->frame_mbs_only_flag) * (p_sps->pic_height_in_map_units_minus1 + 1) * AVC_MACRO_BLOCK_SIZE;
    if ( pic_width_ != curr_pic_width || pic_height_ != curr_pic_height) {
        pic_width_ = curr_pic_width;
        pic_height_ = curr_pic_height;
        // Take care of the case where a new SPS replaces the old SPS with the same id but with different dimensions
        new_seq_activated_ = true;  // Note: clear this flag after the actions are taken.
    }
    // Check DPB buffer size change
    uint32_t dpb_size = p_sps->max_num_ref_frames + 1;
    if (dpb_buffer_.dpb_size != dpb_size) {
        // Need to flush DPB before changing size
        ParserResult ret;
        if ((ret = FlushDpb()) != PARSER_OK) {
            return ret;
        }
        dpb_buffer_.dpb_size = dpb_size;
        dpb_buffer_.dpb_size = dpb_buffer_.dpb_size > AVC_MAX_DPB_FRAMES ? AVC_MAX_DPB_FRAMES : dpb_buffer_.dpb_size;
        CheckAndAdjustDecBufPoolSize(dpb_buffer_.dpb_size);
        new_seq_activated_ = true;
    }

    // Set frame rate if available
    if (new_seq_activated_) {
        if (p_sps->vui_seq_parameters.timing_info_present_flag) {
            frame_rate_.numerator = p_sps->vui_seq_parameters.time_scale;
            frame_rate_.denominator = 2 * p_sps->vui_seq_parameters.num_units_in_tick;
        } else {
            frame_rate_.numerator = 0;
            frame_rate_.denominator = 0;
        }
    }

    if (p_sps->separate_colour_plane_flag == 1) {
        p_slice_header->colour_plane_id = Parser::ReadBits(p_stream, offset, 2);
    }
    p_slice_header->frame_num = Parser::ReadBits(p_stream, offset, p_sps->log2_max_frame_num_minus4 + 4);

    if (p_sps->frame_mbs_only_flag != 1) {
        p_slice_header->field_pic_flag = Parser::GetBit(p_stream, offset);
        if (p_slice_header->field_pic_flag == 1)
        {
            p_slice_header->bottom_field_flag = Parser::GetBit(p_stream, offset);
        }
    } else {
        p_slice_header->field_pic_flag = 0;
        p_slice_header->bottom_field_flag = 0;
    }
    int mbaff_frame_flag = p_sps->mb_adaptive_frame_field_flag && !p_slice_header->field_pic_flag; // MbaffFrameFlag
    int frame_height_in_mbs = (2 - p_sps->frame_mbs_only_flag) * (p_sps->pic_height_in_map_units_minus1 + 1); // FrameHeightInMbs
    int pic_height_in_mbs = frame_height_in_mbs / (1 + p_slice_header->field_pic_flag); // PicHeightInMbs
    int pic_size_in_mbs = (p_sps->pic_width_in_mbs_minus1 + 1) * pic_height_in_mbs; // PicSizeInMbs
    if (mbaff_frame_flag == 0) {
        CHECK_ALLOWED_RANGE("first_mb_in_slice", p_slice_header->first_mb_in_slice, 0, pic_size_in_mbs - 1);
    } else {
        CHECK_ALLOWED_RANGE("first_mb_in_slice", p_slice_header->first_mb_in_slice, 0, pic_size_in_mbs / 2 - 1);
    }

    if ( nal_unit_header_.nal_ref_idc ) {
        curr_ref_pic_bottom_field_ = p_slice_header->bottom_field_flag;
    }
    
    if (nal_unit_header_.nal_unit_type == kAvcNalTypeSlice_IDR) {
        p_slice_header->idr_pic_id = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("idr_pic_id", p_slice_header->idr_pic_id, 0, 65535);
    }

    if (p_sps->pic_order_cnt_type == 0) {
        p_slice_header->pic_order_cnt_lsb = Parser::ReadBits(p_stream, offset, p_sps->log2_max_pic_order_cnt_lsb_minus4 + 4);
        if (p_pps->bottom_field_pic_order_in_frame_present_flag == 1 && p_slice_header->field_pic_flag != 1 ) {
            p_slice_header->delta_pic_order_cnt_bottom = Parser::ExpGolomb::ReadSe(p_stream, offset);
        }
    }

    if (p_sps->pic_order_cnt_type == 1 && p_sps->delta_pic_order_always_zero_flag != 1) {
        p_slice_header->delta_pic_order_cnt[0] = Parser::ExpGolomb::ReadSe(p_stream, offset);
        if (p_pps->bottom_field_pic_order_in_frame_present_flag == 1 && p_slice_header->field_pic_flag != 1) {
            p_slice_header->delta_pic_order_cnt[1] = Parser::ExpGolomb::ReadSe(p_stream, offset);
        }
    }

    if (p_pps->redundant_pic_cnt_present_flag == 1) {
        p_slice_header->redundant_pic_cnt = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("redundant_pic_cnt", p_slice_header->redundant_pic_cnt, 0, 127);
    }

    if (p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6 ) { // B-Slice
        p_slice_header->direct_spatial_mv_pred_flag = Parser::GetBit(p_stream, offset);
    }

    if (p_slice_header->slice_type == kAvcSliceTypeP || p_slice_header->slice_type == kAvcSliceTypeP_5 ||
        p_slice_header->slice_type == kAvcSliceTypeSP || p_slice_header->slice_type == kAvcSliceTypeSP_8 ||
        p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6) {
        p_slice_header->num_ref_idx_active_override_flag = Parser::GetBit(p_stream, offset);
        if (p_slice_header->num_ref_idx_active_override_flag == 1) {
            p_slice_header->num_ref_idx_l0_active_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
            if (p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6) {
                p_slice_header->num_ref_idx_l1_active_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
            }
        } else {
            p_slice_header->num_ref_idx_l0_active_minus1 = p_pps->num_ref_idx_l0_default_active_minus1;
            p_slice_header->num_ref_idx_l1_active_minus1 = p_pps->num_ref_idx_l1_default_active_minus1;
        }
        if (p_slice_header->field_pic_flag == 0) {
            CHECK_ALLOWED_RANGE("num_ref_idx_l0_active_minus1", p_slice_header->num_ref_idx_l0_active_minus1, 0, 15);
            CHECK_ALLOWED_RANGE("num_ref_idx_l1_active_minus1", p_slice_header->num_ref_idx_l1_active_minus1, 0, 15);
        } else {
            CHECK_ALLOWED_RANGE("num_ref_idx_l0_active_minus1", p_slice_header->num_ref_idx_l0_active_minus1, 0, 31);
            CHECK_ALLOWED_RANGE("num_ref_idx_l1_active_minus1", p_slice_header->num_ref_idx_l1_active_minus1, 0, 31);
        }
}

    // Bail out for NAL unit type 20/21
    if ( nal_unit_header_.nal_unit_type == 21 || nal_unit_header_.nal_unit_type == 21) {
        return PARSER_NOT_SUPPORTED;
    }

    // Ref picture list modification
    int modification_of_pic_nums_idc;
    int max_frame_num = 1 << (p_sps->log2_max_frame_num_minus4 + 4); // MaxFrameNum
    int max_pic_num = p_slice_header->field_pic_flag ? 2 * max_frame_num : max_frame_num;
    if (p_slice_header->slice_type != kAvcSliceTypeI && p_slice_header->slice_type != kAvcSliceTypeSI &&
        p_slice_header->slice_type != kAvcSliceTypeI_7 && p_slice_header->slice_type != kAvcSliceTypeSI_9) {
        p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l0 = Parser::GetBit(p_stream, offset);
        if (p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l0 == 1) {
            i = 0;
            do {
                modification_of_pic_nums_idc = Parser::ExpGolomb::ReadUe(p_stream, offset);
                CHECK_ALLOWED_RANGE("modification_of_pic_nums_idc", modification_of_pic_nums_idc, 0, 3);
                p_slice_header->ref_pic_list.modification_l0[i].modification_of_pic_nums_idc = modification_of_pic_nums_idc;
                if (modification_of_pic_nums_idc == 0 || modification_of_pic_nums_idc == 1) {
                    p_slice_header->ref_pic_list.modification_l0[i].abs_diff_pic_num_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
                    CHECK_ALLOWED_RANGE("abs_diff_pic_num_minus1", p_slice_header->ref_pic_list.modification_l0[i].abs_diff_pic_num_minus1, 0, max_pic_num - 1);
                } else if (modification_of_pic_nums_idc == 2) {
                    p_slice_header->ref_pic_list.modification_l0[i].long_term_pic_num = Parser::ExpGolomb::ReadUe(p_stream, offset);
                }
                i++;
            } while (modification_of_pic_nums_idc != 3);
        }
    }

    if (p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6) {
        p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l1 = Parser::GetBit(p_stream, offset);
        if (p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l1 == 1) {
            i = 0;
            do {
                modification_of_pic_nums_idc = Parser::ExpGolomb::ReadUe(p_stream, offset);
                CHECK_ALLOWED_RANGE("modification_of_pic_nums_idc", modification_of_pic_nums_idc, 0, 3);
                p_slice_header->ref_pic_list.modification_l1[i].modification_of_pic_nums_idc = modification_of_pic_nums_idc;
                if (modification_of_pic_nums_idc == 0 || modification_of_pic_nums_idc == 1) {
                    p_slice_header->ref_pic_list.modification_l1[i].abs_diff_pic_num_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
                    CHECK_ALLOWED_RANGE("abs_diff_pic_num_minus1", p_slice_header->ref_pic_list.modification_l1[i].abs_diff_pic_num_minus1, 0, max_pic_num - 1);
                } else if(modification_of_pic_nums_idc == 2) {
                    p_slice_header->ref_pic_list.modification_l1[i].long_term_pic_num = Parser::ExpGolomb::ReadUe(p_stream, offset);
                }
                i++;
            } while (modification_of_pic_nums_idc != 3);
        }
    }

    // Prediction weight table
    if ((p_pps->weighted_pred_flag == 1 &&
            ((p_slice_header->slice_type == kAvcSliceTypeP || p_slice_header->slice_type == kAvcSliceTypeP_5) ||
            (p_slice_header->slice_type == kAvcSliceTypeSP || p_slice_header->slice_type == kAvcSliceTypeSP_8))) ||
        (p_pps->weighted_bipred_idc == 1 &&
            (p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6))) {
        p_slice_header->pred_weight_table.luma_log2_weight_denom = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("luma_log2_weight_denom", p_slice_header->pred_weight_table.luma_log2_weight_denom, 0, 7);
        
        int chroma_array_type = p_sps->separate_colour_plane_flag == 0 ? p_sps->chroma_format_idc : 0;
        if (chroma_array_type != 0) {
            p_slice_header->pred_weight_table.chroma_log2_weight_denom = Parser::ExpGolomb::ReadUe(p_stream, offset);
            CHECK_ALLOWED_RANGE("chroma_log2_weight_denom", p_slice_header->pred_weight_table.chroma_log2_weight_denom, 0, 7);
        }
        
        for (i = 0; i <= p_slice_header->num_ref_idx_l0_active_minus1; i++) {
            p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l0_flag = Parser::GetBit(p_stream, offset);
            if (p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l0_flag == 1) {
                p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l0 = Parser::ExpGolomb::ReadSe(p_stream, offset);
                CHECK_ALLOWED_RANGE("luma_weight_l0", p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l0, -128, 127);
                p_slice_header->pred_weight_table.weight_factor[i].luma_offset_l0 = Parser::ExpGolomb::ReadSe(p_stream, offset);
                CHECK_ALLOWED_RANGE("luma_offset_l0", p_slice_header->pred_weight_table.weight_factor[i].luma_offset_l0, -128, 127);
            } else {
                p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l0 = 1 << p_slice_header->pred_weight_table.luma_log2_weight_denom;
                p_slice_header->pred_weight_table.weight_factor[i].luma_offset_l0 = 0;
            }
            
            if (chroma_array_type != 0) {
                p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l0_flag = Parser::GetBit(p_stream, offset);
                if (p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l0_flag == 1) {
                    for (int j = 0; j < 2; j++) {
                        p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l0[j] = Parser::ExpGolomb::ReadSe(p_stream, offset);
                        CHECK_ALLOWED_RANGE("chroma_weight_l0", p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l0[j], -128, 127);
                        p_slice_header->pred_weight_table.weight_factor[i].chroma_offset_l0[j] = Parser::ExpGolomb::ReadSe(p_stream, offset);
                        CHECK_ALLOWED_RANGE("chroma_offset_l0", p_slice_header->pred_weight_table.weight_factor[i].chroma_offset_l0[j], -128, 127);
                    }
                } else {
                    for (int j = 0; j < 2; j++) {
                        p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l0[j] = 1 << p_slice_header->pred_weight_table.chroma_log2_weight_denom;
                        p_slice_header->pred_weight_table.weight_factor[i].chroma_offset_l0[j] = 0;
                    }
                }
            }
        }
        
        if (p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6) {
            for (int i = 0; i <= p_slice_header->num_ref_idx_l1_active_minus1; i++) {
                p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l1_flag = Parser::GetBit(p_stream, offset);
                if (p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l1_flag == 1) {
                    p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l1 = Parser::ExpGolomb::ReadSe(p_stream, offset);
                    CHECK_ALLOWED_RANGE("luma_weight_l1", p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l1, -128, 127);
                    p_slice_header->pred_weight_table.weight_factor[i].luma_offset_l1 = Parser::ExpGolomb::ReadSe(p_stream, offset);
                    CHECK_ALLOWED_RANGE("luma_offset_l1", p_slice_header->pred_weight_table.weight_factor[i].luma_offset_l1, -128, 127);
                } else {
                    p_slice_header->pred_weight_table.weight_factor[i].luma_weight_l1 = 1 << p_slice_header->pred_weight_table.luma_log2_weight_denom;
                    p_slice_header->pred_weight_table.weight_factor[i].luma_offset_l1 = 0;
                }

                if (chroma_array_type != 0 ) {
                    p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l1_flag = Parser::GetBit(p_stream, offset);
                    if (p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l1_flag == 1) {
                        for (int j = 0; j < 2; j++) {
                            p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l1[j] = Parser::ExpGolomb::ReadSe(p_stream, offset);
                            CHECK_ALLOWED_RANGE("chroma_weight_l1", p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l1[j], -128, 127);
                            p_slice_header->pred_weight_table.weight_factor[i].chroma_offset_l1[j] = Parser::ExpGolomb::ReadSe(p_stream, offset);
                            CHECK_ALLOWED_RANGE("chroma_offset_l1", p_slice_header->pred_weight_table.weight_factor[i].chroma_offset_l1[j], -128, 127);
                        }
                    } else {
                        for (int j = 0; j < 2; j++) {
                            p_slice_header->pred_weight_table.weight_factor[i].chroma_weight_l1[j] = 1 << p_slice_header->pred_weight_table.chroma_log2_weight_denom;
                            p_slice_header->pred_weight_table.weight_factor[i].chroma_offset_l1[j] = 0;
                        }
                    }
                }
            }
        }
    }

    // Decoded reference picture marking.
    int memory_management_control_operation;
    if (nal_unit_header_.nal_ref_idc != 0) {
        if (nal_unit_header_.nal_unit_type == kAvcNalTypeSlice_IDR) {
            p_slice_header->dec_ref_pic_marking.no_output_of_prior_pics_flag = Parser::GetBit(p_stream, offset);
            p_slice_header->dec_ref_pic_marking.long_term_reference_flag = Parser::GetBit(p_stream, offset);
        } else {
            p_slice_header->dec_ref_pic_marking.adaptive_ref_pic_marking_mode_flag = Parser::GetBit(p_stream, offset);
            if (p_slice_header->dec_ref_pic_marking.adaptive_ref_pic_marking_mode_flag == 1) {
                i = 0;
                do {
                    memory_management_control_operation = Parser::ExpGolomb::ReadUe(p_stream, offset);
                    CHECK_ALLOWED_RANGE("memory_management_control_operation", memory_management_control_operation, 0, 6);
                    p_slice_header->dec_ref_pic_marking.mmco[i].memory_management_control_operation = memory_management_control_operation;
                    
                    if (memory_management_control_operation == 1 || memory_management_control_operation == 3) {
                        p_slice_header->dec_ref_pic_marking.mmco[i].difference_of_pic_nums_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
                    }
                    if (memory_management_control_operation == 2) {
                        p_slice_header->dec_ref_pic_marking.mmco[i].long_term_pic_num = Parser::ExpGolomb::ReadUe(p_stream, offset);
                    }
                    if (memory_management_control_operation == 3 || memory_management_control_operation == 6) {
                        p_slice_header->dec_ref_pic_marking.mmco[i].long_term_frame_idx = Parser::ExpGolomb::ReadUe(p_stream, offset);
                    }
                    if (memory_management_control_operation == 4) {
                        p_slice_header->dec_ref_pic_marking.mmco[i].max_long_term_frame_idx_plus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
                        CHECK_ALLOWED_RANGE("max_long_term_frame_idx_plus1", p_slice_header->dec_ref_pic_marking.mmco[i].max_long_term_frame_idx_plus1, 0, p_sps->max_num_ref_frames);
                    }
                    if ( memory_management_control_operation == 5) {
                        curr_has_mmco_5_ = 1;
                    }
                    i++;
                } while (memory_management_control_operation != 0);
                p_slice_header->dec_ref_pic_marking.mmco_count = i - 1;
            }
        }
    }

    if (p_pps->entropy_coding_mode_flag == 1 &&
        p_slice_header->slice_type != kAvcSliceTypeI && p_slice_header->slice_type != kAvcSliceTypeSI &&
        p_slice_header->slice_type != kAvcSliceTypeI_7 && p_slice_header->slice_type != kAvcSliceTypeSI_9) {
        p_slice_header->cabac_init_idc = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("cabac_init_idc", p_slice_header->cabac_init_idc, 0, 2);
    }
    p_slice_header->slice_qp_delta = Parser::ExpGolomb::ReadSe(p_stream, offset);
    int slice_qp_y = 26 + p_pps->pic_init_qp_minus26 + p_slice_header->slice_qp_delta; // SliceQPY
    CHECK_ALLOWED_RANGE("SliceQPY", slice_qp_y, static_cast<int>(-6 * p_sps->bit_depth_luma_minus8), 51);
    if (p_slice_header->slice_type == kAvcSliceTypeSP || p_slice_header->slice_type == kAvcSliceTypeSI ||
        p_slice_header->slice_type == kAvcSliceTypeSP_8 || p_slice_header->slice_type == kAvcSliceTypeSI_9) {
        if (p_slice_header->slice_type == kAvcSliceTypeSP || p_slice_header->slice_type == kAvcSliceTypeSP_8) {
            p_slice_header->sp_for_switch_flag = Parser::GetBit(p_stream, offset);
        }
        p_slice_header->slice_qs_delta = Parser::ExpGolomb::ReadSe(p_stream, offset);
        int qs_y = 26 + p_pps->pic_init_qs_minus26 + p_slice_header->slice_qs_delta; // QSY
        CHECK_ALLOWED_RANGE("QSY", qs_y, 0, 51);
    }

    if (p_pps->deblocking_filter_control_present_flag == 1) {
        p_slice_header->disable_deblocking_filter_idc = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("disable_deblocking_filter_idc", p_slice_header->disable_deblocking_filter_idc, 0, 2);
        if (p_slice_header->disable_deblocking_filter_idc != 1) {
            p_slice_header->slice_alpha_c0_offset_div2 = Parser::ExpGolomb::ReadSe(p_stream, offset);
            CHECK_ALLOWED_RANGE("slice_alpha_c0_offset_div2", p_slice_header->slice_alpha_c0_offset_div2, -6, 6);
            p_slice_header->slice_beta_offset_div2 = Parser::ExpGolomb::ReadSe(p_stream, offset);
            CHECK_ALLOWED_RANGE("slice_beta_offset_div2", p_slice_header->slice_beta_offset_div2, -6, 6);
        }
    }
    if (p_pps->num_slice_groups_minus1 > 0 && p_pps->slice_group_map_type >= 3 && p_pps->slice_group_map_type <= 5) {
        int size = ceil(log2((double)(p_sps->pic_height_in_map_units_minus1+1) / (double)(p_pps->slice_group_change_rate_minus1+1) + 1));
        p_slice_header->slice_group_change_cycle = Parser::ReadBits(p_stream, offset, size);
    }

#if DBGINFO
    PrintSliceHeader(p_slice_header);
#endif // DBGINFO
    return PARSER_OK;
}

ParserResult AvcVideoParser::GetScalingList(uint8_t *p_stream, size_t &offset, uint32_t *scaling_list, uint32_t list_size, uint32_t *use_default_scaling_matrix_flag) {
    int32_t last_scale, next_scale, delta_scale;

    last_scale = 8;
    next_scale = 8;
    for (int j = 0; j < list_size; j++) {
        if (next_scale != 0) {
            delta_scale = Parser::ExpGolomb::ReadSe(p_stream, offset);
            CHECK_ALLOWED_RANGE("delta_scale", delta_scale, -128, 127);
            next_scale = (last_scale + delta_scale + 256) % 256;
            *use_default_scaling_matrix_flag = (j == 0 && next_scale == 0);
        }
        
        scaling_list[j] = (next_scale == 0) ? last_scale : next_scale;
        last_scale = scaling_list[j];
    }
    return PARSER_OK;
}

ParserResult AvcVideoParser::GetVuiParameters(uint8_t *p_stream, size_t &offset, AvcVuiSeqParameters *p_vui_params) {
    p_vui_params->aspect_ratio_info_present_flag = Parser::GetBit(p_stream, offset);
    if (p_vui_params->aspect_ratio_info_present_flag == 1) {
        p_vui_params->aspect_ratio_idc = Parser::ReadBits(p_stream, offset, 8);
        if (p_vui_params->aspect_ratio_idc == 255 /*Extended_SAR*/) {
            p_vui_params->sar_width = Parser::ReadBits(p_stream, offset, 16);
            p_vui_params->sar_height = Parser::ReadBits(p_stream, offset, 16);
        }
    }

    p_vui_params->overscan_info_present_flag = Parser::GetBit(p_stream, offset);
    if (p_vui_params->overscan_info_present_flag == 1) {
        p_vui_params->overscan_appropriate_flag = Parser::GetBit(p_stream, offset);
    }

    p_vui_params->video_signal_type_present_flag = Parser::GetBit(p_stream, offset);
    if (p_vui_params->video_signal_type_present_flag == 1) {
        p_vui_params->video_format = Parser::ReadBits(p_stream, offset, 3);
        p_vui_params->video_full_range_flag = Parser::GetBit(p_stream, offset);
        p_vui_params->colour_description_present_flag = Parser::GetBit(p_stream, offset);
        if (p_vui_params->colour_description_present_flag == 1) {
            p_vui_params->colour_primaries = Parser::ReadBits(p_stream, offset, 8);
            p_vui_params->transfer_characteristics = Parser::ReadBits(p_stream, offset, 8);
            p_vui_params->matrix_coefficients = Parser::ReadBits(p_stream, offset, 8);
        }
    }

    p_vui_params->chroma_loc_info_present_flag = Parser::GetBit(p_stream, offset);
    if (p_vui_params->chroma_loc_info_present_flag == 1) {
        p_vui_params->chroma_sample_loc_type_top_field = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("chroma_sample_loc_type_top_field", p_vui_params->chroma_sample_loc_type_top_field, 0, 5);
        p_vui_params->chroma_sample_loc_type_bottom_field = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("chroma_sample_loc_type_bottom_field", p_vui_params->chroma_sample_loc_type_bottom_field, 0, 5);
    }

    p_vui_params->timing_info_present_flag = Parser::GetBit(p_stream, offset);
    if (p_vui_params->timing_info_present_flag == 1) {
        p_vui_params->num_units_in_tick = Parser::ReadBits(p_stream, offset, 32);
        p_vui_params->time_scale = Parser::ReadBits(p_stream, offset, 32);
        p_vui_params->fixed_frame_rate_flag = Parser::GetBit(p_stream, offset);
    }
    
    p_vui_params->nal_hrd_parameters_present_flag = Parser::GetBit(p_stream, offset);
    if (p_vui_params->nal_hrd_parameters_present_flag == 1 ) {
        p_vui_params->nal_hrd_parameters.cpb_cnt_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("cpb_cnt_minus1", p_vui_params->nal_hrd_parameters.cpb_cnt_minus1, 0, 31);
        p_vui_params->nal_hrd_parameters.bit_rate_scale = Parser::ReadBits(p_stream, offset, 4);
        p_vui_params->nal_hrd_parameters.cpb_size_scale = Parser::ReadBits(p_stream, offset, 4);
        for (int SchedSelIdx = 0; SchedSelIdx <= p_vui_params->nal_hrd_parameters.cpb_cnt_minus1; SchedSelIdx ++) {
            p_vui_params->nal_hrd_parameters.bit_rate_value_minus1[SchedSelIdx] = Parser::ExpGolomb::ReadUe(p_stream, offset);
            p_vui_params->nal_hrd_parameters.cpb_size_value_minus1[SchedSelIdx] = Parser::ExpGolomb::ReadUe(p_stream, offset);
            p_vui_params->nal_hrd_parameters.cbr_flag[SchedSelIdx] = Parser::ReadBits(p_stream, offset, 1);
        }
        p_vui_params->nal_hrd_parameters.initial_cpb_removal_delay_length_minus1 = Parser::ReadBits(p_stream, offset, 5);
        p_vui_params->nal_hrd_parameters.cpb_removal_delay_length_minus1 = Parser::ReadBits(p_stream, offset, 5);
        p_vui_params->nal_hrd_parameters.dpb_output_delay_length_minus1 = Parser::ReadBits(p_stream, offset, 5);
        p_vui_params->nal_hrd_parameters.time_offset_length = Parser::ReadBits(p_stream, offset, 5);
    }
    
    p_vui_params->vcl_hrd_parameters_present_flag = Parser::GetBit(p_stream, offset);
    if (p_vui_params->vcl_hrd_parameters_present_flag == 1) {
        p_vui_params->vcl_hrd_parameters.cpb_cnt_minus1 = Parser::ExpGolomb::ReadUe(p_stream, offset);
        p_vui_params->vcl_hrd_parameters.bit_rate_scale = Parser::ReadBits(p_stream, offset, 4);
        p_vui_params->vcl_hrd_parameters.cpb_size_scale = Parser::ReadBits(p_stream, offset, 4);
        for (int SchedSelIdx = 0; SchedSelIdx <= p_vui_params->vcl_hrd_parameters.cpb_cnt_minus1; SchedSelIdx ++) {
            p_vui_params->vcl_hrd_parameters.bit_rate_value_minus1[SchedSelIdx] = Parser::ExpGolomb::ReadUe(p_stream, offset);
            p_vui_params->vcl_hrd_parameters.cpb_size_value_minus1[SchedSelIdx] = Parser::ExpGolomb::ReadUe(p_stream, offset);
            p_vui_params->vcl_hrd_parameters.cbr_flag[SchedSelIdx] = Parser::GetBit(p_stream, offset);
        }
        p_vui_params->vcl_hrd_parameters.initial_cpb_removal_delay_length_minus1 = Parser::ReadBits(p_stream, offset, 5);
        p_vui_params->vcl_hrd_parameters.cpb_removal_delay_length_minus1 = Parser::ReadBits(p_stream, offset, 5);
        p_vui_params->vcl_hrd_parameters.dpb_output_delay_length_minus1 = Parser::ReadBits(p_stream, offset, 5);
        p_vui_params->vcl_hrd_parameters.time_offset_length = Parser::ReadBits(p_stream, offset, 5);
    }
    if (p_vui_params->nal_hrd_parameters_present_flag == 1 || p_vui_params->vcl_hrd_parameters_present_flag == 1) {
        p_vui_params->low_delay_hrd_flag = Parser::GetBit(p_stream, offset);
    }
    
    p_vui_params->pic_struct_present_flag = Parser::GetBit(p_stream, offset);
    p_vui_params->bitstream_restriction_flag = Parser::GetBit(p_stream, offset);
    if (p_vui_params->bitstream_restriction_flag) {
        p_vui_params->motion_vectors_over_pic_boundaries_flag = Parser::GetBit(p_stream, offset);
        p_vui_params->max_bytes_per_pic_denom = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("max_bytes_per_pic_denom", p_vui_params->max_bytes_per_pic_denom, 0, 16);
        p_vui_params->max_bits_per_mb_denom = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("max_bits_per_mb_denom", p_vui_params->max_bits_per_mb_denom, 0, 16);
        p_vui_params->log2_max_mv_length_horizontal = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("log2_max_mv_length_horizontal", p_vui_params->log2_max_mv_length_horizontal, 0, 15);
        p_vui_params->log2_max_mv_length_vertical = Parser::ExpGolomb::ReadUe(p_stream, offset);
        CHECK_ALLOWED_RANGE("log2_max_mv_length_vertical", p_vui_params->log2_max_mv_length_vertical, 0, 15);
        p_vui_params->max_num_reorder_frames = Parser::ExpGolomb::ReadUe(p_stream, offset);
        p_vui_params->max_dec_frame_buffering = Parser::ExpGolomb::ReadUe(p_stream, offset);
    }
    return PARSER_OK;
}

bool AvcVideoParser::MoreRbspData(uint8_t *p_stream, size_t stream_size_in_byte, size_t bit_offset) {
    bool more_rbsp_bits = false;
    uint8_t curr_byte = p_stream[bit_offset >> 3];
    uint8_t next_bytes[3];
    uint32_t next_byte_offset = (bit_offset >> 3) + 1;
    int bit_offset_in_byte = bit_offset % 8;
    
    /// If the following bytes are not start code, we have more RBSP data. If we don't have enough bytes
    /// in the stream, pad with 0.
    next_bytes[0] = next_byte_offset < stream_size_in_byte ? p_stream[next_byte_offset] : 0;
    next_bytes[1] = (next_byte_offset + 1) < stream_size_in_byte ? p_stream[next_byte_offset + 1] : 0;
    next_bytes[2] = (next_byte_offset + 2) < stream_size_in_byte ? p_stream[next_byte_offset + 2] : 0;

    if ( (next_bytes[0] == 0x00 && next_bytes[1] == 0x00 && next_bytes[2] == 0x00) ||  // padding zero bytes
         (next_bytes[0] == 0x00 && next_bytes[1] == 0x00 && next_bytes[2] == 0x01) ) { // start code
        /// Continue checking the existence of the trailing bits in the current byte.
    } else {
        return true;
    }
        
    /// Check if RBSP trailing bits immediately follow
    if ((bit_offset_in_byte) == 0) {
        if (curr_byte == 0x80) { // RBSP trailing bits
            more_rbsp_bits = false;
        } else {
            more_rbsp_bits = true;
        }
    } else {
        uint8_t curr_bit = curr_byte & (0x80 >> bit_offset_in_byte);
        if ( curr_bit == 0 ) {
            more_rbsp_bits = true; // rbsp_stop_one_bit has to be 1.
        } else {
            /// If this is the last bit, need to grab the next byte
            if ( bit_offset_in_byte == 7 ) {
                if ( next_bytes[0] != 0 ) {
                    more_rbsp_bits = true;
                } else {
                    more_rbsp_bits = false;
                }
            } else {
                bit_offset_in_byte++;
                for (int i = bit_offset_in_byte; i < 8; i++) {
                    curr_bit = curr_byte & (0x80 >> i);
                    if ( curr_bit ) {
                        more_rbsp_bits = true;
                        break;
                    }
                }
            }
        }
    }
    
    return more_rbsp_bits;
}

void AvcVideoParser::InitDpb() {
    memset(&dpb_buffer_, 0, sizeof(DecodedPictureBuffer));
    for (int i = 0; i < AVC_MAX_DPB_FRAMES; i++) {
        dpb_buffer_.frame_buffer_list[i].pic_idx = i;
        dpb_buffer_.frame_buffer_list[i].is_reference = kUnusedForReference;
        dpb_buffer_.frame_buffer_list[i].use_status = kNotUsed;
        dpb_buffer_.frame_buffer_list[i].pic_output_flag = 0;
        dpb_buffer_.field_pic_list[i * 2].pic_idx = i;
        dpb_buffer_.field_pic_list[i * 2 + 1].pic_idx = i;
        dpb_buffer_.field_pic_list[i * 2].is_reference = kUnusedForReference;
        dpb_buffer_.field_pic_list[i * 2 + 1].is_reference = kUnusedForReference;
        dpb_buffer_.field_pic_list[i * 2].use_status = kNotUsed;
        dpb_buffer_.field_pic_list[i * 2 + 1].use_status = kNotUsed;
        dpb_buffer_.field_pic_list[i * 2].pic_output_flag = 0;
        dpb_buffer_.field_pic_list[i * 2 + 1].pic_output_flag = 0;
    }
    dpb_buffer_.dpb_size = 0;
    dpb_buffer_.dpb_fullness = 0;
    dpb_buffer_.num_short_term = 0;
    dpb_buffer_.num_long_term = 0;
    dpb_buffer_.num_short_term_ref_fields = 0;
    dpb_buffer_.num_long_term_ref_fields = 0;
    dpb_buffer_.num_pics_needed_for_output = 0;
}

// 8.2.1 Decoding process for picture order count
void AvcVideoParser::CalculateCurrPoc() {
    AvcSeqParameterSet *p_sps = &sps_list_[active_sps_id_];
    AvcSliceHeader *p_slice_header = &slice_info_list_[0].slice_header;
    int frame_num_offset; // FrameNumOffset

    int max_pic_order_cnt_lsb = 1 << (p_sps->log2_max_pic_order_cnt_lsb_minus4 + 4); // MaxPicOrderCntLsb
    int max_frame_num = 1 << (p_sps->log2_max_frame_num_minus4 + 4); // max_frame_num

    if (p_sps->pic_order_cnt_type == 0) {
        if (slice_nal_unit_header_.nal_unit_type == kAvcNalTypeSlice_IDR) {
            prev_pic_order_cnt_msb_ = 0;
            prev_pic_order_cnt_lsb_ = 0;
        } else {
            if (prev_has_mmco_5_) {
                if (prev_ref_pic_bottom_field_) {
                    prev_pic_order_cnt_msb_ = 0;
                    prev_pic_order_cnt_lsb_ = 0;
                } else {
                    prev_pic_order_cnt_msb_ = 0;
                    prev_pic_order_cnt_lsb_ = prev_top_field_order_cnt_;
                }
            }
        }

        int pic_order_cnt_msb;
		if ( (p_slice_header->pic_order_cnt_lsb < prev_pic_order_cnt_lsb_) && ((prev_pic_order_cnt_lsb_ - p_slice_header->pic_order_cnt_lsb) >= (max_pic_order_cnt_lsb / 2))) {
			pic_order_cnt_msb = prev_pic_order_cnt_msb_ + max_pic_order_cnt_lsb;
        } else if ((p_slice_header->pic_order_cnt_lsb > prev_pic_order_cnt_lsb_) && ( (p_slice_header->pic_order_cnt_lsb - prev_pic_order_cnt_lsb_) > (max_pic_order_cnt_lsb / 2))) {
			pic_order_cnt_msb = prev_pic_order_cnt_msb_ - max_pic_order_cnt_lsb;
        } else {
			pic_order_cnt_msb = prev_pic_order_cnt_msb_;
        }

		if (!p_slice_header->field_pic_flag || !p_slice_header->bottom_field_flag) {
			curr_pic_.top_field_order_cnt = pic_order_cnt_msb + p_slice_header->pic_order_cnt_lsb;
        }
		if (!p_slice_header->field_pic_flag) {
			curr_pic_.bottom_field_order_cnt = curr_pic_.top_field_order_cnt + p_slice_header->delta_pic_order_cnt_bottom;
        } else if (p_slice_header->bottom_field_flag) {
			curr_pic_.bottom_field_order_cnt = pic_order_cnt_msb + p_slice_header->pic_order_cnt_lsb;
        }
        if (slice_nal_unit_header_.nal_ref_idc) {
			prev_pic_order_cnt_msb_ = pic_order_cnt_msb;
            prev_pic_order_cnt_lsb_ = p_slice_header->pic_order_cnt_lsb;
            prev_top_field_order_cnt_ = curr_pic_.top_field_order_cnt;
        }
    } else if (p_sps->pic_order_cnt_type == 1) {
        int abs_frame_num; // absFrameNum

		if (slice_nal_unit_header_.nal_unit_type == kAvcNalTypeSlice_IDR){
			frame_num_offset = 0;
        } else {
            if (prev_has_mmco_5_) {
                prev_frame_num_offset_ = 0;
                prev_frame_num_ = 0;
            }
            if (prev_frame_num_ > p_slice_header->frame_num) {
                frame_num_offset = prev_frame_num_offset_ + max_frame_num;
            } else {
                frame_num_offset = prev_frame_num_offset_;
            }
        }
        
        if (p_sps->num_ref_frames_in_pic_order_cnt_cycle) {
            abs_frame_num = frame_num_offset + p_slice_header->frame_num;
        } else {
            abs_frame_num = 0;
        }
        if ((!slice_nal_unit_header_.nal_ref_idc) && abs_frame_num > 0) {
            abs_frame_num--;
        }
        
        int expected_delta_per_pic_order_cnt_cycle = 0; // ExpectedDeltaPerPicOrderCntCycle
        if (p_sps->num_ref_frames_in_pic_order_cnt_cycle) {
            for (int i = 0; i < p_sps->num_ref_frames_in_pic_order_cnt_cycle; i++) {
                expected_delta_per_pic_order_cnt_cycle += p_sps->offset_for_ref_frame[i];
            }
        }
        
        int expected_pic_order_cnt; // expectedPicOrderCnt
        if( abs_frame_num > 0 ) {
            int pic_order_cnt_cycle_cnt = (abs_frame_num - 1) / p_sps->num_ref_frames_in_pic_order_cnt_cycle; // picOrderCntCycleCnt
            int frame_num_in_pic_order_cnt_cycle = (abs_frame_num - 1) % p_sps->num_ref_frames_in_pic_order_cnt_cycle; // frameNumInPicOrderCntCycle
            expected_pic_order_cnt = pic_order_cnt_cycle_cnt * expected_delta_per_pic_order_cnt_cycle;
            for (int i = 0; i <= frame_num_in_pic_order_cnt_cycle; i++) {
                expected_pic_order_cnt += p_sps->offset_for_ref_frame[i];
            }
        } else {
            expected_pic_order_cnt = 0;
        }
        if (!slice_nal_unit_header_.nal_ref_idc ) {
            expected_pic_order_cnt += p_sps->offset_for_non_ref_pic;
        }
        
        if (!p_slice_header->field_pic_flag) {
            curr_pic_.top_field_order_cnt = expected_pic_order_cnt + p_slice_header->delta_pic_order_cnt[0];
            curr_pic_.bottom_field_order_cnt = curr_pic_.top_field_order_cnt + p_sps->offset_for_top_to_bottom_field + p_slice_header->delta_pic_order_cnt[1];
        } else if (p_slice_header->bottom_field_flag) {
            curr_pic_.bottom_field_order_cnt = expected_pic_order_cnt + p_sps->offset_for_top_to_bottom_field + p_slice_header->delta_pic_order_cnt[0];
        } else {
            curr_pic_.top_field_order_cnt = expected_pic_order_cnt + p_slice_header->delta_pic_order_cnt[0];
        }
        
        prev_frame_num_ = p_slice_header->frame_num;
        prev_frame_num_offset_ = frame_num_offset;
    } else if (p_sps->pic_order_cnt_type == 2) {
        if (slice_nal_unit_header_.nal_unit_type == kAvcNalTypeSlice_IDR) {
            frame_num_offset = 0;
            curr_pic_.top_field_order_cnt = 0;
            curr_pic_.bottom_field_order_cnt = 0;
        } else {
            if (prev_has_mmco_5_) {
                prev_frame_num_offset_ = 0;
            }
            
            if (prev_frame_num_ > p_slice_header->frame_num) {
                frame_num_offset = prev_frame_num_offset_ + max_frame_num;
            } else {
                frame_num_offset = prev_frame_num_offset_;
            }
            
            int temp_pic_order_cnt; // tempPicOrderCnt
            if (slice_nal_unit_header_.nal_ref_idc == 0) {
                temp_pic_order_cnt = 2 * (frame_num_offset + p_slice_header->frame_num) - 1;
            } else {
                temp_pic_order_cnt = 2 * (frame_num_offset + p_slice_header->frame_num);
            }
            
            if (!p_slice_header->field_pic_flag) {
                curr_pic_.top_field_order_cnt = temp_pic_order_cnt;
                curr_pic_.bottom_field_order_cnt = temp_pic_order_cnt;
            } else if (p_slice_header->bottom_field_flag) {
                curr_pic_.bottom_field_order_cnt = temp_pic_order_cnt;
            } else {
                curr_pic_.top_field_order_cnt = temp_pic_order_cnt;
            }
        }
        
        prev_frame_num_ = p_slice_header->frame_num;
        prev_frame_num_offset_ = frame_num_offset;
	}

    if (p_slice_header->field_pic_flag) {
        curr_pic_.pic_order_cnt = p_slice_header->bottom_field_flag ? curr_pic_.bottom_field_order_cnt : curr_pic_.top_field_order_cnt;
    } else {
        curr_pic_.pic_order_cnt = curr_pic_.top_field_order_cnt <= curr_pic_.bottom_field_order_cnt ? curr_pic_.top_field_order_cnt : curr_pic_.bottom_field_order_cnt;
    }
}

// 8.2.5.2 Decoding process for gaps in frame_num
ParserResult AvcVideoParser::DecodeFrameNumGaps() {
    AvcSeqParameterSet *p_sps = &sps_list_[active_sps_id_];
    if (p_sps->gaps_in_frame_num_value_allowed_flag == 0) {
        return PARSER_OK;
    }

    AvcSliceHeader *p_slice_header = &slice_info_list_[0].slice_header;
    int max_frame_num = 1 << (p_sps->log2_max_frame_num_minus4 + 4); // MaxFrameNum

    if (slice_nal_unit_header_.nal_unit_type == kAvcNalTypeSlice_IDR) {
        prev_ref_frame_num_ = 0;
    } else if ((p_slice_header->frame_num != prev_ref_frame_num_) && (p_slice_header->frame_num != ((prev_ref_frame_num_ + 1) % max_frame_num))) {
        logger_.ErrorLog(MakeMsg("Support for gaps in frame_num is currently disabled."));
        return PARSER_NOT_SUPPORTED;
        #if 0
        int unused_short_term_frame_num = (prev_ref_frame_num_ + 1) % max_frame_num;
        while (unused_short_term_frame_num != p_slice_header->frame_num) {
            AvcPicture non_existing_pic = {0};
            non_existing_pic.frame_num = unused_short_term_frame_num;
            non_existing_pic.is_reference = kUsedForShortTerm;
            non_existing_pic.pic_output_flag = 0;

            // Calculate POC
            int frame_num_offset; // FrameNumOffset
            if (p_sps->pic_order_cnt_type == 0) {
                // -1 is to try to avoid generating POC for the non-existing reference frames the same value as the existing ref.
                // Need to implement a more robust way.
                non_existing_pic.top_field_order_cnt = prev_top_field_order_cnt_ - 1;
                non_existing_pic.bottom_field_order_cnt = non_existing_pic.top_field_order_cnt + p_slice_header->delta_pic_order_cnt_bottom - 1;

                non_existing_pic.pic_order_cnt = non_existing_pic.top_field_order_cnt <= non_existing_pic.bottom_field_order_cnt ? non_existing_pic.top_field_order_cnt : non_existing_pic.bottom_field_order_cnt;
            } else if (p_sps->pic_order_cnt_type == 1) {
                int abs_frame_num; // absFrameNum
                if (prev_has_mmco_5_) {
                    prev_frame_num_offset_ = 0;
                    prev_frame_num_ = 0;
                }
                if (prev_frame_num_ > p_slice_header->frame_num) {
                    frame_num_offset = prev_frame_num_offset_ + max_frame_num;
                } else {
                    frame_num_offset = prev_frame_num_offset_;
                }

                if (p_sps->num_ref_frames_in_pic_order_cnt_cycle) {
                    abs_frame_num = frame_num_offset + p_slice_header->frame_num;
                } else {
                    abs_frame_num = 0;
                }
                if ((!slice_nal_unit_header_.nal_ref_idc) && abs_frame_num > 0) {
                    abs_frame_num--;
                }

                int expected_delta_per_pic_order_cnt_cycle = 0; // ExpectedDeltaPerPicOrderCntCycle
                if (p_sps->num_ref_frames_in_pic_order_cnt_cycle) {
                    for (int i = 0; i < p_sps->num_ref_frames_in_pic_order_cnt_cycle; i++) {
                        expected_delta_per_pic_order_cnt_cycle += p_sps->offset_for_ref_frame[i];
                    }
                }

                int expected_pic_order_cnt; // expectedPicOrderCnt
                if( abs_frame_num > 0 ) {
                    int pic_order_cnt_cycle_cnt = (abs_frame_num - 1) / p_sps->num_ref_frames_in_pic_order_cnt_cycle; // picOrderCntCycleCnt
                    int frame_num_in_pic_order_cnt_cycle = (abs_frame_num - 1) % p_sps->num_ref_frames_in_pic_order_cnt_cycle; // frameNumInPicOrderCntCycle
                    expected_pic_order_cnt = pic_order_cnt_cycle_cnt * expected_delta_per_pic_order_cnt_cycle;
                    for (int i = 0; i <= frame_num_in_pic_order_cnt_cycle; i++) {
                        expected_pic_order_cnt += p_sps->offset_for_ref_frame[i];
                    }
                } else {
                    expected_pic_order_cnt = 0;
                }
                if (!slice_nal_unit_header_.nal_ref_idc ) {
                    expected_pic_order_cnt += p_sps->offset_for_non_ref_pic;
                }

                if (!p_slice_header->field_pic_flag) {
                    non_existing_pic.top_field_order_cnt = expected_pic_order_cnt + p_slice_header->delta_pic_order_cnt[0];
                    non_existing_pic.bottom_field_order_cnt = non_existing_pic.top_field_order_cnt + p_sps->offset_for_top_to_bottom_field + p_slice_header->delta_pic_order_cnt[1];
                } else if (p_slice_header->bottom_field_flag) {
                    non_existing_pic.bottom_field_order_cnt = expected_pic_order_cnt + p_sps->offset_for_top_to_bottom_field + p_slice_header->delta_pic_order_cnt[0];
                } else {
                    non_existing_pic.top_field_order_cnt = expected_pic_order_cnt + p_slice_header->delta_pic_order_cnt[0];
                }

                prev_frame_num_ = p_slice_header->frame_num;
                prev_frame_num_offset_ = frame_num_offset;
            } else if (p_sps->pic_order_cnt_type == 2) {
                if (prev_has_mmco_5_) {
                    prev_frame_num_offset_ = 0;
                }
                if (prev_frame_num_ > non_existing_pic.frame_num) {
                    frame_num_offset = prev_frame_num_offset_ + max_frame_num;
                } else {
                    frame_num_offset = prev_frame_num_offset_;
                }

                int temp_pic_order_cnt; // tempPicOrderCnt
                // nal_ref_idc inferred to be not equal to 0
                temp_pic_order_cnt = 2 * (frame_num_offset + non_existing_pic.frame_num);
                // field_pic_flag inferred to be equal to 0
                non_existing_pic.top_field_order_cnt = temp_pic_order_cnt;
                non_existing_pic.bottom_field_order_cnt = temp_pic_order_cnt;
                non_existing_pic.pic_order_cnt = non_existing_pic.top_field_order_cnt <= non_existing_pic.bottom_field_order_cnt ? non_existing_pic.top_field_order_cnt : non_existing_pic.bottom_field_order_cnt;

                prev_frame_num_ = non_existing_pic.frame_num;
                prev_frame_num_offset_ = frame_num_offset;
            }

            // Decode picture numbers. 8.2.4.1
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForShortTerm) {
                    // Eq. 8-27
                    if (p_ref_pic->frame_num > non_existing_pic.frame_num) {
                        p_ref_pic->frame_num_wrap = p_ref_pic->frame_num - max_frame_num;
                    } else {
                        p_ref_pic->frame_num_wrap = p_ref_pic->frame_num;
                    }

                    if (non_existing_pic.pic_structure == kFrame) {
                        p_ref_pic->pic_num = p_ref_pic->frame_num_wrap;  // Eq. 8-28
                    } else if (((non_existing_pic.pic_structure == kTopField) && (p_ref_pic->pic_structure == kTopField)) || ((non_existing_pic.pic_structure == kBottomField) && (p_ref_pic->pic_structure == kBottomField))) {
                        p_ref_pic->pic_num = 2 * p_ref_pic->frame_num_wrap + 1;  // Eq. 8-30
                    } else {
                        p_ref_pic->pic_num = 2 * p_ref_pic->frame_num_wrap;  // Eq. 8-31
                    }
                } else if (p_ref_pic->is_reference == kUsedForLongTerm) {
                    if (non_existing_pic.pic_structure == kFrame) {
                        p_ref_pic->long_term_pic_num = p_ref_pic->long_term_frame_idx;  // Eq. 8-29
                    } else if (((non_existing_pic.pic_structure == kTopField) && (p_ref_pic->pic_structure == kTopField)) || ((non_existing_pic.pic_structure == kBottomField) && (p_ref_pic->pic_structure == kBottomField))) {
                        p_ref_pic->long_term_pic_num = 2 * p_ref_pic->long_term_frame_idx + 1;  // Eq. 8-32
                    } else {
                        p_ref_pic->long_term_pic_num = 2 * p_ref_pic->long_term_frame_idx;  // Eq. 8-33
                    }
                }
            }

            // Sliding window decoded reference picture marking. 8.2.5.3
            if (dpb_buffer_.num_short_term + dpb_buffer_.num_long_term == p_sps->max_num_ref_frames) {
                int32_t min_frame_num_wrap = 0xFFFFFF; // more than the largest possible value of FrameNumWrap (2 ^ 16)
                int min_index = AVC_MAX_DPB_FRAMES;
                for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                    if (dpb_buffer_.frame_buffer_list[i].is_reference == kUsedForShortTerm) {
                        if (dpb_buffer_.frame_buffer_list[i].frame_num_wrap < min_frame_num_wrap) {
                            min_frame_num_wrap = dpb_buffer_.frame_buffer_list[i].frame_num_wrap;
                            min_index = i;
                        }
                    }
                }
                if (min_index < dpb_buffer_.dpb_size) {
                    dpb_buffer_.frame_buffer_list[min_index].is_reference = kUnusedForReference;
                } else {
                    logger_.ErrorLog(MakeMsg("Could not find any short term ref picture."));
                    return PARSER_FAIL;
                }
                dpb_buffer_.num_short_term--;
            }

            // Insert the non-existing short-term reference picture to DPB
            if (dpb_buffer_.dpb_fullness == dpb_buffer_.dpb_size) {
                if (BumpPicFromDpb() != PARSER_OK) {
                        return PARSER_FAIL;
                }
            }

            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                if (dpb_buffer_.frame_buffer_list[i].use_status == kNotUsed) {
                    break;
                }
            }
            if (i < dpb_buffer_.dpb_size) {
                non_existing_pic.pic_idx = dpb_buffer_.frame_buffer_list[i].pic_idx;
                non_existing_pic.use_status = kFrameUsedForDecode;
                dpb_buffer_.frame_buffer_list[i] = non_existing_pic;
                dpb_buffer_.dpb_fullness++;
                dpb_buffer_.num_short_term++;
            } else {
                logger_.ErrorLog(MakeMsg("Could not find any free frame buffer in DPB."));
                return PARSER_FAIL;
            }

            // Update prev_ref_frame_num_ to value of frame_num for the last of the "non-existing" reference frames inferred 
            // by the decoding process for gaps in frame_num specified in clause 8.2.5.2.
            prev_ref_frame_num_ = unused_short_term_frame_num;

            unused_short_term_frame_num = (unused_short_term_frame_num + 1) % max_frame_num;
        }
        if (slice_nal_unit_header_.nal_ref_idc) {
            prev_ref_frame_num_ = p_slice_header->frame_num;
        }
        #endif
    } else {
        if (slice_nal_unit_header_.nal_ref_idc) {
            prev_ref_frame_num_ = p_slice_header->frame_num;
        }
    }

    return PARSER_OK;
}

// 8.2.4
static inline int ComparePicNumDesc(const void *p_pic_info_1, const void *p_pic_info_2) {
    int pic_num_1 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_1))->pic_num;
    int pic_num_2 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_2))->pic_num;

    if (pic_num_1 < pic_num_2) {
        return 1;
    } else if (pic_num_1 > pic_num_2) {
        return -1;
    } else {
        return 0;
    }
}

static inline int ComparePocDesc(const void *p_pic_info_1, const void *p_pic_info_2) {
    int poc_1 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_1))->pic_order_cnt;
    int poc_2 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_2))->pic_order_cnt;

    if (poc_1 < poc_2) {
        return 1;
    } else if (poc_1 > poc_2) {
        return -1;
    } else {
        return 0;
    }
}

static inline int ComparePocAsc(const void *p_pic_info_1, const void *p_pic_info_2) {
    int poc_1 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_1))->pic_order_cnt;
    int poc_2 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_2))->pic_order_cnt;

    if (poc_1 < poc_2) {
        return -1;
    } else if (poc_1 > poc_2) {
        return 1;
    } else {
        return 0;
    }
}

static inline int CompareLongTermPicNumAsc(const void *p_pic_info_1, const void *p_pic_info_2) {
    int long_term_pic_num_1 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_1))->long_term_pic_num;
    int long_term_pic_num_2 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_2))->long_term_pic_num;

    if (long_term_pic_num_1 < long_term_pic_num_2) {
        return -1;
    } else if (long_term_pic_num_1 > long_term_pic_num_2) {
        return 1;
    } else {
        return 0;
    }
}

static inline int CompareFrameNumWrapDesc(const void *p_pic_info_1, const void *p_pic_info_2) {
    int frame_num_wrap_1 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_1))->frame_num_wrap;
    int frame_num_wrap_2 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_2))->frame_num_wrap;

    if (frame_num_wrap_1 < frame_num_wrap_2) {
        return 1;
    } else if (frame_num_wrap_1 > frame_num_wrap_2) {
        return -1;
    } else {
        return 0;
    }
}

static inline int CompareLongTermFrameIdxAsc(const void *p_pic_info_1, const void *p_pic_info_2) {
    int long_term_frame_idx_1 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_1))->long_term_frame_idx;
    int long_term_frame_idx_2 = (reinterpret_cast<const AvcVideoParser::AvcPicture*>(p_pic_info_2))->long_term_frame_idx;

    if (long_term_frame_idx_1 < long_term_frame_idx_2) {
        return -1;
    } else if (long_term_frame_idx_1 > long_term_frame_idx_2) {
        return 1;
    } else {
        return 0;
    }
}

// 8.2.4 Decoding process for reference picture lists construction
ParserResult AvcVideoParser::SetupReflist(AvcSliceInfo *p_slice_info) {
    AvcSeqParameterSet *p_sps = &sps_list_[active_sps_id_];
    int max_frame_num = 1 << (p_sps->log2_max_frame_num_minus4 + 4); // MaxFrameNum
    AvcSliceHeader *p_slice_header = &p_slice_info->slice_header;
    int i;

    memset(p_slice_info->ref_list_0_, 0, sizeof(p_slice_info->ref_list_0_));
    memset(p_slice_info->ref_list_1_, 0, sizeof(p_slice_info->ref_list_1_));

    // 8.2.4.1. Calculate picture numbers. Only do it once.
    if (num_slices_ == 0) {
        if (p_slice_header->field_pic_flag == 0) {
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForShortTerm) {
                    // Eq. 8-27
                    if (p_ref_pic->frame_num > curr_pic_.frame_num) {
                        p_ref_pic->frame_num_wrap = p_ref_pic->frame_num - max_frame_num;
                    } else {
                        p_ref_pic->frame_num_wrap = p_ref_pic->frame_num;
                    }
                    p_ref_pic->pic_num = p_ref_pic->frame_num_wrap;  // Eq. 8-28
                } else if (p_ref_pic->is_reference == kUsedForLongTerm) {
                    p_ref_pic->long_term_pic_num = p_ref_pic->long_term_frame_idx;  // Eq. 8-29
                }
            }
        } else {
            for (i = 0; i < dpb_buffer_.dpb_size * 2; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.field_pic_list[i];
                if (p_ref_pic->is_reference == kUsedForShortTerm) {
                    // Eq. 8-27
                    if (p_ref_pic->frame_num > curr_pic_.frame_num) {
                        p_ref_pic->frame_num_wrap = p_ref_pic->frame_num - max_frame_num;
                    } else {
                        p_ref_pic->frame_num_wrap = p_ref_pic->frame_num;
                    }
                    // Assign to the corresponding frame
                    dpb_buffer_.frame_buffer_list[i / 2].frame_num_wrap = p_ref_pic->frame_num_wrap;
                    if (((curr_pic_.pic_structure == kTopField) && (p_ref_pic->pic_structure == kTopField)) || ((curr_pic_.pic_structure == kBottomField) && (p_ref_pic->pic_structure == kBottomField))) {
                        p_ref_pic->pic_num = 2 * p_ref_pic->frame_num_wrap + 1;  // Eq. 8-30
                    } else {
                        p_ref_pic->pic_num = 2 * p_ref_pic->frame_num_wrap;  // Eq. 8-31
                    }
                } else if (p_ref_pic->is_reference == kUsedForLongTerm) {
                    if (((curr_pic_.pic_structure == kTopField) && (p_ref_pic->pic_structure == kTopField)) || ((curr_pic_.pic_structure == kBottomField) && (p_ref_pic->pic_structure == kBottomField))) {
                        p_ref_pic->long_term_pic_num = 2 * p_ref_pic->long_term_frame_idx + 1;  // Eq. 8-32
                    } else {
                        p_ref_pic->long_term_pic_num = 2 * p_ref_pic->long_term_frame_idx;  // Eq. 8-33
                    }
                }
            }
        }
    }

    if (p_slice_header->slice_type == kAvcSliceTypeI || p_slice_header->slice_type == kAvcSliceTypeSI || p_slice_header->slice_type == kAvcSliceTypeI_7 || p_slice_header->slice_type == kAvcSliceTypeSI_9) {
        // We still need to do 8.2.4.1 above for I pictures but will not go furhter.
        return PARSER_OK;
    }

    // 8.2.4.2 Initialisation process for reference picture lists
    if (p_slice_header->slice_type == kAvcSliceTypeP || p_slice_header->slice_type == kAvcSliceTypeP_5) {
        if (curr_pic_.pic_structure == kFrame) { // 8.2.4.2.1 Initialisation process for the reference picture list for P and SP slices in frames
            // Group short term ref pictures
            int ref_index = 0;
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForShortTerm) {
                    p_slice_info->ref_list_0_[ref_index] = *p_ref_pic;
                    ref_index++;
                }
            }
            // Group long term ref pictures
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForLongTerm) {
                    p_slice_info->ref_list_0_[ref_index] = *p_ref_pic;
                    ref_index++;
                }
            }
            // Sort short term refs with descending order of pic_num
            if (dpb_buffer_.num_short_term > 1) {
                qsort((void*)p_slice_info->ref_list_0_, dpb_buffer_.num_short_term, sizeof(AvcPicture), ComparePicNumDesc);
            }
            // Sort long term refs with ascending order of long_term_pic_num
            if (dpb_buffer_.num_long_term > 1) {
                qsort((void*)&p_slice_info->ref_list_0_[dpb_buffer_.num_short_term], dpb_buffer_.num_long_term, sizeof(AvcPicture), CompareLongTermPicNumAsc);
            }
        } else { // 8.2.4.2.2 Initialisation process for the reference picture list for P and SP slices in fields
            // Construct and sort refFrameList0ShortTerm
            AvcPicture ref_frame_list0_short_term[AVC_MAX_REF_FRAME_NUM] = {{0}};
            int index = 0;
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                if (dpb_buffer_.field_pic_list[i * 2].is_reference == kUsedForShortTerm || dpb_buffer_.field_pic_list[i * 2 + 1].is_reference == kUsedForShortTerm) {
                    // frame_buffer_list[i] should have the corresponding FrameNumWrap
                    ref_frame_list0_short_term[index] = dpb_buffer_.frame_buffer_list[i];
                    index++;
                }
            }
            if (index > 1) {
                qsort((void*)ref_frame_list0_short_term, index, sizeof(AvcPicture), CompareFrameNumWrapDesc);
            }

            FillFieldRefList(ref_frame_list0_short_term, index, kUsedForShortTerm, curr_pic_.pic_structure, p_slice_info->ref_list_0_, &dpb_buffer_.num_short_term_ref_fields);

            // Construct and sort refFrameList0LongTerm
            AvcPicture ref_frame_list0_long_term[AVC_MAX_REF_FRAME_NUM] = {{0}};
            index = 0;
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                if (dpb_buffer_.field_pic_list[i * 2].is_reference == kUsedForLongTerm || dpb_buffer_.field_pic_list[i * 2 + 1].is_reference == kUsedForLongTerm) {
                    // frame_buffer_list[i] should have the corresponding LongTermFrameIdx
                    ref_frame_list0_long_term[index] = dpb_buffer_.frame_buffer_list[i];
                    index++;
                }
            }
            if (index > 1) {
                qsort((void*)ref_frame_list0_long_term, index, sizeof(AvcPicture), CompareLongTermFrameIdxAsc);
            }
            if (index > 0) {
                FillFieldRefList(ref_frame_list0_long_term, index, kUsedForLongTerm, curr_pic_.pic_structure, &p_slice_info->ref_list_0_[dpb_buffer_.num_short_term_ref_fields], &dpb_buffer_.num_long_term_ref_fields);
            }
        }
    } else {
        if (curr_pic_.pic_structure == kFrame) { // 8.2.4.2.3 Initialisation process for reference picture lists for B slices in frames
            // RefPicList0
            int num_short_term_smaller = 0;
            int num_short_term_greater = 0;
            int num_long_term = 0;
            int ref_index = 0;
            // Group short term ref pictures that have smaller POC than the current picture
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForShortTerm && p_ref_pic->pic_order_cnt < curr_pic_.pic_order_cnt) {
                    p_slice_info->ref_list_0_[ref_index] = *p_ref_pic;
                    num_short_term_smaller++;
                    ref_index++;
                }
            }
            // Sort in descending order of POC
            if (num_short_term_smaller > 1) {
                qsort((void*)p_slice_info->ref_list_0_, num_short_term_smaller, sizeof(AvcPicture), ComparePocDesc);
            }

            // Group short term ref pictures that have greater POC than the current picture
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForShortTerm && p_ref_pic->pic_order_cnt > curr_pic_.pic_order_cnt) {
                    p_slice_info->ref_list_0_[ref_index] = *p_ref_pic;
                    num_short_term_greater++;
                    ref_index++;
                }
            }
            // Sort in ascending order of POC
            if (num_short_term_greater > 1) {
                qsort((void*)&p_slice_info->ref_list_0_[num_short_term_smaller], num_short_term_greater, sizeof(AvcPicture), ComparePocAsc);
            }

            // Group long term ref pictures
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForLongTerm) {
                    p_slice_info->ref_list_0_[ref_index] = *p_ref_pic;
                    num_long_term++;
                    ref_index++;
                }
            }
            // Sort long term refs with ascending order of long_term_pic_num
            if (num_long_term > 1) {
                qsort((void*)&p_slice_info->ref_list_0_[num_short_term_smaller + num_short_term_greater], num_long_term, sizeof(AvcPicture), CompareLongTermPicNumAsc);
            }

            // RefPicList1
            num_short_term_smaller = 0;
            num_short_term_greater = 0;
            num_long_term = 0;
            ref_index = 0;

            // Group short term ref pictures that have greater POC than the current picture
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForShortTerm && p_ref_pic->pic_order_cnt > curr_pic_.pic_order_cnt) {
                    p_slice_info->ref_list_1_[ref_index] = *p_ref_pic;
                    num_short_term_greater++;
                    ref_index++;
                }
            }
            // Sort in ascending order of POC
            if (num_short_term_greater > 1) {
                qsort((void*)p_slice_info->ref_list_1_, num_short_term_greater, sizeof(AvcPicture), ComparePocAsc);
            }

            // Group short term ref pictures that have smaller POC than the current picture
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForShortTerm && p_ref_pic->pic_order_cnt < curr_pic_.pic_order_cnt) {
                    p_slice_info->ref_list_1_[ref_index] = *p_ref_pic;
                    num_short_term_smaller++;
                    ref_index++;
                }
            }
            // Sort in descending order of POC
            if (num_short_term_smaller > 1) {
                qsort((void*)&p_slice_info->ref_list_1_[num_short_term_greater], num_short_term_smaller, sizeof(AvcPicture), ComparePocDesc);
            }
 
            // Group long term ref pictures
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                AvcPicture *p_ref_pic = &dpb_buffer_.frame_buffer_list[i];
                if (p_ref_pic->is_reference == kUsedForLongTerm) {
                    p_slice_info->ref_list_1_[ref_index] = *p_ref_pic;
                    num_long_term++;
                    ref_index++;
                }
            }
            // Sort long term refs with ascending order of long_term_pic_num
            if (num_long_term > 1) {
                qsort((void*)&p_slice_info->ref_list_1_[num_short_term_smaller + num_short_term_greater], num_long_term, sizeof(AvcPicture), CompareLongTermPicNumAsc);
            }
        } else { // 8.2.4.2.4 Initialisation process for reference picture lists for B slices in fields
            // ===========
            // RefPicList0
            // ===========
            // Construct and sort refFrameList0ShortTerm
            AvcPicture ref_frame_list0_short_term[AVC_MAX_REF_FRAME_NUM] = {{0}};
            int num_short_term_smaller = 0;
            int num_short_term_greater = 0;
            int index = 0;
            // Group short term ref pictures that have smaller POC than the current picture
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                if ((dpb_buffer_.field_pic_list[i * 2].is_reference == kUsedForShortTerm || dpb_buffer_.field_pic_list[i * 2 + 1].is_reference == kUsedForShortTerm) && dpb_buffer_.frame_buffer_list[i].pic_order_cnt < curr_pic_.pic_order_cnt) {
                    ref_frame_list0_short_term[index] = dpb_buffer_.frame_buffer_list[i];
                    num_short_term_smaller++;
                    index++;
                }
            }
            // Sort in descending order of POC
            if (num_short_term_smaller > 1) {
                qsort((void*)ref_frame_list0_short_term, num_short_term_smaller, sizeof(AvcPicture), ComparePocDesc);
            }
            // Group short term ref pictures that have greater POC than the current picture
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                if ((dpb_buffer_.field_pic_list[i * 2].is_reference == kUsedForShortTerm || dpb_buffer_.field_pic_list[i * 2 + 1].is_reference == kUsedForShortTerm) && dpb_buffer_.frame_buffer_list[i].pic_order_cnt > curr_pic_.pic_order_cnt) {
                    ref_frame_list0_short_term[index] = dpb_buffer_.frame_buffer_list[i];
                    num_short_term_greater++;
                    index++;
                }
            }
            // Sort in ascending order of POC
            if (num_short_term_greater > 1) {
                qsort((void*)&ref_frame_list0_short_term[num_short_term_smaller], num_short_term_greater, sizeof(AvcPicture), ComparePocAsc);
            }

            FillFieldRefList(ref_frame_list0_short_term, num_short_term_smaller + num_short_term_greater, kUsedForShortTerm, curr_pic_.pic_structure, p_slice_info->ref_list_0_, &dpb_buffer_.num_short_term_ref_fields);

            // Construct and sort refFrameListLongTerm
            AvcPicture ref_frame_list_long_term[AVC_MAX_REF_FRAME_NUM] = {{0}};
            int num_long_term = 0;
            index = 0;
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                if (dpb_buffer_.field_pic_list[i * 2].is_reference == kUsedForLongTerm || dpb_buffer_.field_pic_list[i * 2 + 1].is_reference == kUsedForLongTerm) {
                    // frame_buffer_list[i] should have the corresponding LongTermFrameIdx
                    ref_frame_list_long_term[index] = dpb_buffer_.frame_buffer_list[i];
                    num_long_term++;
                    index++;
                }
            }
            if (index > 1) {
                qsort((void*)ref_frame_list_long_term, num_long_term, sizeof(AvcPicture), CompareLongTermFrameIdxAsc);
            }
            if (num_long_term > 0) {
                FillFieldRefList(ref_frame_list_long_term, num_long_term, kUsedForLongTerm, curr_pic_.pic_structure, &p_slice_info->ref_list_0_[dpb_buffer_.num_short_term_ref_fields], &dpb_buffer_.num_long_term_ref_fields);
            }

            // ===========
            // RefPicList1
            // ===========
            // Construct and sort refFrameList1ShortTerm
            AvcPicture ref_frame_list1_short_term[AVC_MAX_REF_FRAME_NUM] = {{0}};
            num_short_term_smaller = 0;
            num_short_term_greater = 0;
            index = 0;
            // Group short term ref pictures that have greater POC than the current picture
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                if ((dpb_buffer_.field_pic_list[i * 2].is_reference == kUsedForShortTerm || dpb_buffer_.field_pic_list[i * 2 + 1].is_reference == kUsedForShortTerm) && dpb_buffer_.frame_buffer_list[i].pic_order_cnt > curr_pic_.pic_order_cnt) {
                    ref_frame_list1_short_term[index] = dpb_buffer_.frame_buffer_list[i];
                    num_short_term_greater++;
                    index++;
                }
            }
            // Sort in ascending order of POC
            if (num_short_term_greater > 1) {
                qsort((void*)ref_frame_list1_short_term, num_short_term_greater, sizeof(AvcPicture), ComparePocAsc);
            }
            // Group short term ref pictures that have smaller POC than the current picture
            for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                if ((dpb_buffer_.field_pic_list[i * 2].is_reference == kUsedForShortTerm || dpb_buffer_.field_pic_list[i * 2 + 1].is_reference == kUsedForShortTerm) && dpb_buffer_.frame_buffer_list[i].pic_order_cnt < curr_pic_.pic_order_cnt) {
                    ref_frame_list1_short_term[index] = dpb_buffer_.frame_buffer_list[i];
                    num_short_term_smaller++;
                    index++;
                }
            }
            // Sort in descending order of POC
            if (num_short_term_smaller > 1) {
                qsort((void*)&ref_frame_list1_short_term[num_short_term_greater], num_short_term_smaller, sizeof(AvcPicture), ComparePocDesc);
            }

            uint32_t num_ref_fields;
            FillFieldRefList(ref_frame_list1_short_term, num_short_term_smaller + num_short_term_greater, kUsedForShortTerm, curr_pic_.pic_structure, p_slice_info->ref_list_1_, &num_ref_fields);
            if (num_long_term > 0) {
                FillFieldRefList(ref_frame_list_long_term, num_long_term, kUsedForLongTerm, curr_pic_.pic_structure, &p_slice_info->ref_list_1_[num_ref_fields], &num_ref_fields);
            }
        }
    }

    // 8.2.4.3 Modification process for reference picture lists
    if (p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l0 == 1) {
        AvcPicture *ref_pic_list_x = p_slice_info->ref_list_0_; // RefPicListX
        AvcListMod *p_list_mod = p_slice_header->ref_pic_list.modification_l0;
        int num_ref_idx_lx_active = p_slice_header->num_ref_idx_l0_active_minus1 + 1;
        ParserResult ret;
        if ((ret = ModifiyRefList(ref_pic_list_x, p_list_mod, num_ref_idx_lx_active, p_slice_header)) != PARSER_OK) {
            return ret;
        }
    }

    if (p_slice_header->slice_type == kAvcSliceTypeB || p_slice_header->slice_type == kAvcSliceTypeB_6) {
        if (p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l1 == 1) {
            AvcPicture *ref_pic_list_x = p_slice_info->ref_list_1_; // RefPicListX
            AvcListMod *p_list_mod = p_slice_header->ref_pic_list.modification_l1;
            int num_ref_idx_lx_active = p_slice_header->num_ref_idx_l1_active_minus1 + 1;
            ParserResult ret;
            if ((ret = ModifiyRefList(ref_pic_list_x, p_list_mod, num_ref_idx_lx_active, p_slice_header)) != PARSER_OK) {
                return ret;
            }
        }
    }
    return PARSER_OK;
}

void AvcVideoParser::FillFieldRefList(AvcPicture *ref_frame_list_x, int num_ref_frames, int ref_type, int curr_field_parity, AvcPicture *ref_pic_list_x, uint32_t *num_fields_filled) {
    int index_same_parity = 0;
    int index_opposite_parity = 0;
    int index_field_ref_list = 0;
    int index, i;
    bool found;

    while (index_same_parity < num_ref_frames || index_opposite_parity < num_ref_frames) {
        // First look for the next same parity field if present
        found = false;
        while (index_same_parity < num_ref_frames) {
            // pic_idx is also the array element index of dpb_buffer_.frame_buffer_list
            index = ref_frame_list_x[index_same_parity].pic_idx;
            for (i = 0; i < 2; i++) {
                if (dpb_buffer_.field_pic_list[index * 2 + i].is_reference == ref_type && dpb_buffer_.field_pic_list[index * 2 + i].pic_structure == curr_field_parity) {
                    ref_pic_list_x[index_field_ref_list] = dpb_buffer_.field_pic_list[index * 2 + i];
                    index_field_ref_list++;
                    found = true;
                }
            }
            index_same_parity++;
            if (found) {
                break;
            }
        }

        // Then look for the next opposite parity field if present
        found = false;
        while (index_opposite_parity < num_ref_frames) {
            // pic_idx is also the array element index of dpb_buffer_.frame_buffer_list
            index = ref_frame_list_x[index_opposite_parity].pic_idx;
            for (i = 0; i < 2; i++) {
                if (dpb_buffer_.field_pic_list[index * 2 + i].is_reference == ref_type && dpb_buffer_.field_pic_list[index * 2 + i].pic_structure != curr_field_parity) {
                    ref_pic_list_x[index_field_ref_list] = dpb_buffer_.field_pic_list[index * 2 + i];
                    index_field_ref_list++;
                    found = true;
                }
            }
            index_opposite_parity++;
            if (found) {
                break;
            }
        }
    }
    *num_fields_filled = index_field_ref_list;
}

ParserResult AvcVideoParser::ModifiyRefList(AvcPicture *ref_pic_list_x, AvcListMod *p_list_mod, int num_ref_idx_lx_active, AvcSliceHeader *p_slice_header) {
    AvcSeqParameterSet *p_sps = &sps_list_[active_sps_id_];
    int ref_idx_lx = 0; // refIdxLX
    int curr_pic_num = p_slice_header->field_pic_flag ? 2 * p_slice_header->frame_num + 1 : p_slice_header->frame_num; // CurrPicNum
    int pic_num_lx_pred = curr_pic_num; // picNumLXPred
    int max_frame_num = 1 << (p_sps->log2_max_frame_num_minus4 + 4); // MaxFrameNum
    int max_pic_num = p_slice_header->field_pic_flag ? 2 * max_frame_num : max_frame_num;
    int num_short_term_pics = curr_pic_.pic_structure == kFrame ? dpb_buffer_.num_short_term : dpb_buffer_.num_short_term_ref_fields;
    int num_long_term_pics = curr_pic_.pic_structure == kFrame ? dpb_buffer_.num_long_term : dpb_buffer_.num_long_term_ref_fields;
    AvcPicture ref_pic_list_mod[AVC_MAX_REF_PICTURE_NUM + 1];
    int i, c_idx, n_idx;

    memcpy(ref_pic_list_mod, ref_pic_list_x, sizeof(AvcPicture) * num_ref_idx_lx_active);

    while (p_list_mod->modification_of_pic_nums_idc != 3) {
        if (p_list_mod->modification_of_pic_nums_idc < 2) {
            // 8.2.4.3.1 Modification process of reference picture lists for short-term reference pictures
            int pic_num_lx_no_wrap; // picNumLXNoWrap
            if (p_list_mod->modification_of_pic_nums_idc == 0) {
                // (8-34)
                if (pic_num_lx_pred < (p_list_mod->abs_diff_pic_num_minus1 + 1)) {
                    pic_num_lx_no_wrap = pic_num_lx_pred - (p_list_mod->abs_diff_pic_num_minus1 + 1) + max_pic_num;
                } else {
                    pic_num_lx_no_wrap = pic_num_lx_pred - (p_list_mod->abs_diff_pic_num_minus1 + 1);
                }
            } else {
                // (8-35)
                if ((pic_num_lx_pred + (p_list_mod->abs_diff_pic_num_minus1 + 1)) >= max_pic_num) {
                    pic_num_lx_no_wrap = pic_num_lx_pred + (p_list_mod->abs_diff_pic_num_minus1 + 1) - max_pic_num;
                } else {
                    pic_num_lx_no_wrap = pic_num_lx_pred + (p_list_mod->abs_diff_pic_num_minus1 + 1);
                }
            }
            // (8-36)
            pic_num_lx_pred = pic_num_lx_no_wrap;
            int pic_num_lx; // picNumLX
            if (pic_num_lx_no_wrap > curr_pic_num) {
                pic_num_lx = pic_num_lx_no_wrap - max_pic_num;
            } else {
                pic_num_lx = pic_num_lx_no_wrap;
            }
            // Shift up
            for (c_idx = num_ref_idx_lx_active; c_idx > ref_idx_lx; c_idx--) {
                ref_pic_list_mod[c_idx] = ref_pic_list_mod[c_idx - 1];
            }
            // (8-37)
            // Find short-term reference picture with PicNum equal to pic_num_lx
            for (i = 0; i < num_short_term_pics; i++) {
                if (ref_pic_list_x[i].is_reference == kUsedForShortTerm && ref_pic_list_x[i].pic_num == pic_num_lx) {
                    break;
                }
            }
            if (i == num_short_term_pics) {
                logger_.ErrorLog(MakeMsg("Could not find a short-term reference with the modified pic num."));
                return PARSER_OUT_OF_RANGE;
            }
            ref_pic_list_mod[ref_idx_lx] = ref_pic_list_x[i];
            ref_idx_lx++;
            n_idx = ref_idx_lx;
            for (c_idx = ref_idx_lx; c_idx <= num_ref_idx_lx_active; c_idx++) {
                int pic_num_f = ref_pic_list_mod[c_idx].is_reference == kUsedForShortTerm ? ref_pic_list_mod[c_idx].pic_num : max_pic_num;
                if (pic_num_f != pic_num_lx) {
                    ref_pic_list_mod[n_idx++] = ref_pic_list_mod[c_idx];
                }
            }

        } else if (p_list_mod->modification_of_pic_nums_idc == 2) {
            // 8.2.4.3.2 Modification process of reference picture lists for long-term reference pictures
            // Shift up
            for (c_idx = num_ref_idx_lx_active; c_idx > ref_idx_lx; c_idx--) {
                ref_pic_list_mod[c_idx] = ref_pic_list_mod[c_idx - 1];
            }
            // (8-38)
            // Find long-term reference picture with LongTermPicNum equal to long_term_pic_num
            for (i = num_short_term_pics; i < num_short_term_pics + num_long_term_pics; i++) {
                if (ref_pic_list_x[i].is_reference == kUsedForLongTerm && ref_pic_list_x[i].long_term_pic_num == p_list_mod->long_term_pic_num) {
                    break;
                }
            }
            if (i == num_short_term_pics + num_long_term_pics) {
                logger_.ErrorLog(MakeMsg("Could not find long-term reference with the modified long term pic num."));
                return PARSER_OUT_OF_RANGE;
            }
            ref_pic_list_mod[ref_idx_lx] = ref_pic_list_x[i];
            ref_idx_lx++;
            n_idx = ref_idx_lx;
            for (c_idx = ref_idx_lx; c_idx <= num_ref_idx_lx_active; c_idx++) {
                int long_term_pic_num_f = ref_pic_list_mod[c_idx].is_reference == kUsedForLongTerm ? ref_pic_list_mod[c_idx].long_term_pic_num : 2 * (max_long_term_frame_idx_ + 1);
                if (long_term_pic_num_f != p_list_mod->long_term_pic_num) {
                    ref_pic_list_mod[n_idx++] = ref_pic_list_mod[c_idx];
                }
            }
        }
        p_list_mod = &p_slice_header->ref_pic_list.modification_l0[ref_idx_lx];
    }

    memcpy(ref_pic_list_x, ref_pic_list_mod, sizeof(AvcPicture) * num_ref_idx_lx_active);
    return PARSER_OK;
}

ParserResult AvcVideoParser::CheckDpbAndOutput() {
    // If DPB is full, bump one picture out
    if (dpb_buffer_.dpb_fullness == dpb_buffer_.dpb_size) {
        if (BumpPicFromDpb() != PARSER_OK) {
                return PARSER_FAIL;
        }
    }
    // Output decoded pictures from DPB if any are ready
    if (pfn_display_picture_cb_ && num_output_pics_ > 0) {
        if (OutputDecodedPictures(false) != PARSER_OK) {
            return PARSER_FAIL;
        }
    }
    return PARSER_OK;
}

ParserResult AvcVideoParser::FindFreeInDecBufPool() {
    int dec_buf_index;

    if (curr_pic_.pic_structure == kFrame || !second_field_) {
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

        curr_pic_.dec_buf_idx = dec_buf_index;
        if ( curr_pic_.pic_structure != kFrame) {
                first_field_dec_buf_idx_ = dec_buf_index;
        }
    } else {
        curr_pic_.dec_buf_idx = first_field_dec_buf_idx_;
    }

    return PARSER_OK;
}

ParserResult AvcVideoParser::FindFreeBufInDpb() {
    int i;
    if (curr_pic_.pic_structure == kFrame || !second_field_) {
        if (dpb_buffer_.dpb_fullness == dpb_buffer_.dpb_size) {
            if (BumpPicFromDpb() != PARSER_OK) {
                    return PARSER_FAIL;
            }
        }

        for (i = 0; i < dpb_buffer_.dpb_size; i++) {
            if (dpb_buffer_.frame_buffer_list[i].use_status == kNotUsed) {
                break;
            }
        }
        if (i < dpb_buffer_.dpb_size) {
            curr_pic_.pic_idx = dpb_buffer_.frame_buffer_list[i].pic_idx;
            if (curr_pic_.pic_structure == kFrame) {
                curr_pic_.use_status = kFrameUsedForDecode;
            } else if (curr_pic_.pic_structure == kTopField) {
                curr_pic_.use_status = kTopFieldUsedForDecode;
            } else {
                curr_pic_.use_status = kBottomFieldUsedForDecode;
            }
        } else {
            logger_.ErrorLog(MakeMsg("Could not find any free frame buffer in DPB."));
            return PARSER_FAIL;
        }

        if ( curr_pic_.pic_structure != kFrame) {
            first_field_pic_idx_ = i;
        }
    } else {
        curr_pic_.pic_idx = first_field_pic_idx_;
        if (curr_pic_.pic_structure == kTopField) {
            curr_pic_.use_status = kTopFieldUsedForDecode;
        } else {
            curr_pic_.use_status = kBottomFieldUsedForDecode;
        }
    }

    return PARSER_OK;
}

// 8.2.5 Decoded reference picture marking process
ParserResult AvcVideoParser::MarkDecodedRefPics() {
    AvcSeqParameterSet *p_sps = &sps_list_[active_sps_id_];
    AvcSliceHeader *p_slice_header = &slice_info_list_[0].slice_header;
    int i;

    if (slice_nal_unit_header_.nal_ref_idc == 0) {
        return PARSER_OK;
    }

    if (slice_nal_unit_header_.nal_unit_type == kAvcNalTypeSlice_IDR) {  // 8.2.5.1: 1. & 2.
        // Mark all reference pictures as "unused for reference
        for (i = 0; i < AVC_MAX_DPB_FRAMES; i++) {
            dpb_buffer_.frame_buffer_list[i].is_reference = kUnusedForReference;
            dpb_buffer_.field_pic_list[i * 2].is_reference = kUnusedForReference;
            dpb_buffer_.field_pic_list[i * 2 + 1].is_reference = kUnusedForReference;
        }
        // Output the remaining picutres in DPB
        if (FlushDpb() != PARSER_OK) {
            return PARSER_FAIL;
        }
        dpb_buffer_.num_long_term = 0;
        dpb_buffer_.num_short_term = 0;
        dpb_buffer_.num_short_term_ref_fields = 0;
        dpb_buffer_.num_long_term_ref_fields = 0;

        dpb_buffer_.dpb_fullness = 0;
        if (p_slice_header->dec_ref_pic_marking.long_term_reference_flag) {
            curr_pic_.is_reference = kUsedForLongTerm;
            curr_pic_.long_term_frame_idx = 0;
            max_long_term_frame_idx_ = 0;
        } else {
            curr_pic_.is_reference = kUsedForShortTerm;
            max_long_term_frame_idx_ = NO_LONG_TERM_FRAME_INDICES;
        }
    } else {
        curr_pic_.is_reference = kUsedForShortTerm;  // 8.2.5.1: 3.
        if (p_slice_header->dec_ref_pic_marking.adaptive_ref_pic_marking_mode_flag == 1) {
            // 8.2.5.4 Adaptive memory control decoded reference picture marking process
            for (i = 0; i < p_slice_header->dec_ref_pic_marking.mmco_count; i++) {
                AvcMmco *p_mmco = &p_slice_header->dec_ref_pic_marking.mmco[i];
                switch (p_mmco->memory_management_control_operation) {
                    case 1: { // 8.2.5.4.1 Marking process of a short-term reference picture as "unused for reference"
                        int curr_pic_num = p_slice_header->field_pic_flag ? 2 * p_slice_header->frame_num + 1 : p_slice_header->frame_num;
                        int pic_num_x = curr_pic_num - (p_mmco->difference_of_pic_nums_minus1 + 1);
                        if (p_slice_header->field_pic_flag) {
                            for (int j = 0; j < dpb_buffer_.dpb_size * 2; j++) {
                                if (dpb_buffer_.field_pic_list[j].is_reference == kUsedForShortTerm && dpb_buffer_.field_pic_list[j].pic_num == pic_num_x) {
                                    dpb_buffer_.field_pic_list[j].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_short_term_ref_fields--;
                                    dpb_buffer_.frame_buffer_list[j / 2].is_reference = kUnusedForReference;
                                    if (dpb_buffer_.field_pic_list[(j / 2) * 2].is_reference == kUnusedForReference && dpb_buffer_.field_pic_list[(j / 2) * 2 + 1].is_reference == kUnusedForReference) {
                                        dpb_buffer_.num_short_term--;
                                    }
                                    break;
                                }
                            }
                        } else {
                            for (int j = 0; j < dpb_buffer_.dpb_size; j++) {
                                if (dpb_buffer_.frame_buffer_list[j].is_reference == kUsedForShortTerm && dpb_buffer_.frame_buffer_list[j].pic_num == pic_num_x) {
                                    dpb_buffer_.frame_buffer_list[j].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_short_term--;
                                    if (dpb_buffer_.field_pic_list[j * 2].is_reference == kUsedForShortTerm) {
                                        dpb_buffer_.field_pic_list[j * 2].is_reference = kUnusedForReference;
                                        dpb_buffer_.num_short_term_ref_fields--;
                                    }
                                    if (dpb_buffer_.field_pic_list[j * 2 + 1].is_reference == kUsedForShortTerm) {
                                        dpb_buffer_.field_pic_list[j * 2 + 1].is_reference = kUnusedForReference;
                                        dpb_buffer_.num_short_term_ref_fields--;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    break;

                    case 2: { // 8.2.5.4.2 Marking process of a long-term reference picture as "unused for reference"
                        if (p_slice_header->field_pic_flag) {
                            for (int j = 0; j < dpb_buffer_.dpb_size * 2; j++) {
                                if (dpb_buffer_.field_pic_list[j].is_reference == kUsedForLongTerm && dpb_buffer_.field_pic_list[j].long_term_pic_num == p_mmco->long_term_pic_num) {
                                    dpb_buffer_.field_pic_list[j].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_long_term_ref_fields--;
                                    dpb_buffer_.frame_buffer_list[j / 2].is_reference = kUnusedForReference;
                                    if (dpb_buffer_.field_pic_list[(j / 2) * 2].is_reference == kUnusedForReference && dpb_buffer_.field_pic_list[(j / 2) * 2 + 1].is_reference == kUnusedForReference) {
                                        dpb_buffer_.num_long_term--;
                                    }
                                    break;
                                }
                            }
                        } else {
                            for (int j = 0; j < dpb_buffer_.dpb_size; j++) {
                                if (dpb_buffer_.frame_buffer_list[j].is_reference == kUsedForLongTerm && dpb_buffer_.frame_buffer_list[j].long_term_pic_num == p_mmco->long_term_pic_num) {
                                    dpb_buffer_.frame_buffer_list[j].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_long_term--;
                                    if (dpb_buffer_.field_pic_list[j * 2].is_reference == kUsedForLongTerm) {
                                        dpb_buffer_.field_pic_list[j * 2].is_reference = kUnusedForReference;
                                        dpb_buffer_.num_long_term_ref_fields--;
                                    }
                                    if (dpb_buffer_.field_pic_list[j * 2 + 1].is_reference == kUsedForLongTerm) {
                                        dpb_buffer_.field_pic_list[j * 2 + 1].is_reference = kUnusedForReference;
                                        dpb_buffer_.num_long_term_ref_fields--;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    break;

                    case 3: { // Assignment process of a LongTermFrameIdx to a short-term reference picture
                        for (int j = 0; j < dpb_buffer_.dpb_size; j++) {
                            if (dpb_buffer_.frame_buffer_list[j].is_reference == kUsedForLongTerm && dpb_buffer_.frame_buffer_list[j].long_term_frame_idx == p_mmco->long_term_frame_idx) {
                                dpb_buffer_.frame_buffer_list[j].is_reference = kUnusedForReference;
                                dpb_buffer_.num_long_term--;
                                if (dpb_buffer_.field_pic_list[j * 2].is_reference == kUsedForLongTerm) {
                                    dpb_buffer_.field_pic_list[j * 2].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_long_term_ref_fields--;
                                }
                                if (dpb_buffer_.field_pic_list[j * 2 + 1].is_reference == kUsedForLongTerm) {
                                    dpb_buffer_.field_pic_list[j * 2 + 1].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_long_term_ref_fields--;
                                }
                                break;
                            }
                        }

                        int curr_pic_num = p_slice_header->field_pic_flag ? 2 * p_slice_header->frame_num + 1 : p_slice_header->frame_num;
                        int pic_num_x = curr_pic_num - (p_mmco->difference_of_pic_nums_minus1 + 1);
                        for (int j = 0; j < dpb_buffer_.dpb_size * 2; j++) {
                            if (dpb_buffer_.field_pic_list[j].is_reference == kUsedForLongTerm && dpb_buffer_.field_pic_list[j].long_term_frame_idx == p_mmco->long_term_frame_idx) {
                                if (dpb_buffer_.field_pic_list[(j / 2) * 2].pic_num != pic_num_x && dpb_buffer_.field_pic_list[(j / 2) * 2 + 1].pic_num != pic_num_x) {
                                    dpb_buffer_.field_pic_list[j].is_reference = kUnusedForReference;
                                }
                                break;
                            }
                        }

                        if (p_slice_header->field_pic_flag) {
                            for (int j = 0; j < dpb_buffer_.dpb_size * 2; j++) {
                                if (dpb_buffer_.field_pic_list[j].is_reference == kUsedForShortTerm && dpb_buffer_.field_pic_list[j].pic_num == pic_num_x) {
                                    dpb_buffer_.field_pic_list[j].is_reference = kUsedForLongTerm;
                                    dpb_buffer_.field_pic_list[j].long_term_frame_idx = p_mmco->long_term_frame_idx;
                                    dpb_buffer_.num_short_term_ref_fields--;
                                    dpb_buffer_.num_long_term_ref_fields++;
                                    if (dpb_buffer_.field_pic_list[(j / 2) * 2].is_reference == kUsedForLongTerm && dpb_buffer_.field_pic_list[(j / 2) * 2 + 1].is_reference == kUsedForLongTerm ) {
                                        dpb_buffer_.frame_buffer_list[j / 2].is_reference = kUsedForLongTerm;
                                        dpb_buffer_.frame_buffer_list[j / 2].long_term_frame_idx = p_mmco->long_term_frame_idx;
                                        dpb_buffer_.num_short_term--;
                                        dpb_buffer_.num_long_term++;
                                    }
                                    break;
                                }
                            }
                        } else {
                            for (int j = 0; j < dpb_buffer_.dpb_size; j++) {
                                if (dpb_buffer_.frame_buffer_list[j].is_reference == kUsedForShortTerm && dpb_buffer_.frame_buffer_list[j].pic_num == pic_num_x) {
                                    dpb_buffer_.frame_buffer_list[j].is_reference = kUsedForLongTerm;
                                    dpb_buffer_.frame_buffer_list[j].long_term_frame_idx = p_mmco->long_term_frame_idx;
                                    dpb_buffer_.num_short_term--;
                                    dpb_buffer_.num_long_term++;
                                    if (dpb_buffer_.field_pic_list[j * 2].is_reference == kUsedForShortTerm) {
                                        dpb_buffer_.field_pic_list[j * 2].is_reference = kUsedForLongTerm;
                                        dpb_buffer_.field_pic_list[j * 2].long_term_frame_idx = p_mmco->long_term_frame_idx;
                                        dpb_buffer_.num_short_term_ref_fields--;
                                        dpb_buffer_.num_long_term_ref_fields++;
                                    }
                                    if (dpb_buffer_.field_pic_list[j * 2 + 1].is_reference == kUsedForShortTerm) {
                                        dpb_buffer_.field_pic_list[j * 2 + 1].is_reference = kUsedForLongTerm;
                                        dpb_buffer_.field_pic_list[j * 2 + 1].long_term_frame_idx = p_mmco->long_term_frame_idx;
                                        dpb_buffer_.num_short_term_ref_fields--;
                                        dpb_buffer_.num_long_term_ref_fields++;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    break;

                    case 4: { // 8.2.5.4.4 Decoding process for MaxLongTermFrameIdx
                        if (p_mmco->max_long_term_frame_idx_plus1 == 0) {
                            max_long_term_frame_idx_ = NO_LONG_TERM_FRAME_INDICES;
                            for (int j = 0; j < dpb_buffer_.dpb_size; j++) {
                                if (dpb_buffer_.frame_buffer_list[j].is_reference == kUsedForLongTerm) {
                                    dpb_buffer_.frame_buffer_list[j].is_reference = kUnusedForReference;
                                }
                                if (dpb_buffer_.field_pic_list[j * 2].is_reference == kUsedForLongTerm) {
                                    dpb_buffer_.field_pic_list[j * 2].is_reference = kUnusedForReference;
                                }
                                if (dpb_buffer_.field_pic_list[j * 2 + 1].is_reference == kUsedForLongTerm) {
                                    dpb_buffer_.field_pic_list[j * 2 + 1].is_reference = kUnusedForReference;
                                }
                            }
                            dpb_buffer_.num_long_term = 0;
                            dpb_buffer_.num_long_term_ref_fields = 0;
                        } else {
                            max_long_term_frame_idx_ = p_mmco->max_long_term_frame_idx_plus1 - 1;
                            for (int j = 0; j < dpb_buffer_.dpb_size; j++) {
                                if (dpb_buffer_.frame_buffer_list[j].is_reference == kUsedForLongTerm && dpb_buffer_.frame_buffer_list[j].long_term_frame_idx > max_long_term_frame_idx_) {
                                    dpb_buffer_.frame_buffer_list[j].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_long_term--;
                                }
                                if (dpb_buffer_.field_pic_list[j * 2].is_reference == kUsedForLongTerm && dpb_buffer_.field_pic_list[j * 2].long_term_frame_idx > max_long_term_frame_idx_) {
                                    dpb_buffer_.field_pic_list[j * 2].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_long_term_ref_fields--;
                                }
                                if (dpb_buffer_.field_pic_list[j * 2 + 1].is_reference == kUsedForLongTerm && dpb_buffer_.field_pic_list[j * 2 + 1].long_term_frame_idx > max_long_term_frame_idx_) {
                                    dpb_buffer_.field_pic_list[j * 2 + 1].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_long_term_ref_fields--;
                                }
                            }
                        }
                    }
                    break;

                    case 5: { // 8.2.5.4.5 Marking process of all reference pictures as "unused for reference" and setting MaxLongTermFrameIdx to "no long-term frame indices"
                        for (int j = 0; j < dpb_buffer_.dpb_size; j++) {
                            dpb_buffer_.frame_buffer_list[j].is_reference = kUnusedForReference;
                            dpb_buffer_.field_pic_list[j * 2].is_reference = kUnusedForReference;
                            dpb_buffer_.field_pic_list[j * 2 + 1].is_reference = kUnusedForReference;
                        }
                        // Output the remaining picutres in DPB
                        if (FlushDpb() != PARSER_OK) {
                                return PARSER_FAIL;
                        }
                        max_long_term_frame_idx_ = NO_LONG_TERM_FRAME_INDICES;
                        dpb_buffer_.num_short_term = 0;
                        dpb_buffer_.num_long_term = 0;
                        dpb_buffer_.num_short_term_ref_fields = 0;
                        dpb_buffer_.num_long_term_ref_fields = 0;

                        curr_pic_.frame_num = 0;
                        curr_pic_.pic_num = 0;
                        prev_frame_num_ = 0;
                        prev_ref_frame_num_ = 0;
                        if (p_slice_header->field_pic_flag) {
                            if (p_slice_header->bottom_field_flag) {
                                curr_pic_.bottom_field_order_cnt = 0;
                                curr_pic_.pic_order_cnt = 0;
                            } else {
                                curr_pic_.top_field_order_cnt = 0;
                                curr_pic_.pic_order_cnt = 0;
                            }
                        } else {
                            curr_pic_.top_field_order_cnt -= curr_pic_.pic_order_cnt;
                            curr_pic_.bottom_field_order_cnt -= curr_pic_.pic_order_cnt;
                            curr_pic_.pic_order_cnt = curr_pic_.top_field_order_cnt <= curr_pic_.bottom_field_order_cnt ? curr_pic_.top_field_order_cnt : curr_pic_.bottom_field_order_cnt;
                        }
                    }
                    break;

                    case 6: { // 8.2.5.4.6 Process for assigning a long-term frame index to the current picture
                        for (int j = 0; j < dpb_buffer_.dpb_size; j++) {
                            if (dpb_buffer_.frame_buffer_list[j].is_reference == kUsedForLongTerm && dpb_buffer_.frame_buffer_list[j].long_term_frame_idx == p_mmco->long_term_frame_idx) {
                                dpb_buffer_.frame_buffer_list[j].is_reference = kUnusedForReference;
                                dpb_buffer_.num_long_term--;
                                if (dpb_buffer_.field_pic_list[j * 2].is_reference == kUsedForLongTerm) {
                                    dpb_buffer_.field_pic_list[j * 2].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_long_term_ref_fields--;
                                }
                                if (dpb_buffer_.field_pic_list[j * 2 + 1].is_reference == kUsedForLongTerm) {
                                    dpb_buffer_.field_pic_list[j * 2 + 1].is_reference = kUnusedForReference;
                                    dpb_buffer_.num_long_term_ref_fields--;
                                }
                                break;
                            }
                        }
                        for (int j = 0; j < dpb_buffer_.dpb_size * 2; j++) {
                            if (dpb_buffer_.field_pic_list[j].is_reference == kUsedForLongTerm && dpb_buffer_.field_pic_list[j].long_term_frame_idx == p_mmco->long_term_frame_idx && dpb_buffer_.field_pic_list[j].pic_idx != curr_pic_.pic_idx) {
                                dpb_buffer_.field_pic_list[j].is_reference = kUnusedForReference;
                                break;
                            }
                        }

                        curr_pic_.is_reference = kUsedForLongTerm;
                        curr_pic_.long_term_frame_idx = p_mmco->long_term_frame_idx;
                        if (p_slice_header->field_pic_flag && second_field_) {
                            int j = curr_pic_.pic_idx;
                            if (dpb_buffer_.field_pic_list[j * 2].is_reference == kUsedForLongTerm) {
                                dpb_buffer_.frame_buffer_list[j].is_reference = kUsedForLongTerm;
                                dpb_buffer_.frame_buffer_list[j].long_term_frame_idx = p_mmco->long_term_frame_idx;
                            }
                        }
                    }
                    break;

                    default: {
                        logger_.ErrorLog(MakeMsg("Invalid memory management control operation!"));
                        return PARSER_INVALID_ARG;
                    }
                }

            }
        } else {
            // 8.2.5.3 Sliding window decoded reference picture marking process
            if (p_slice_header->field_pic_flag && second_field_) {
                i = curr_pic_.pic_idx;
                if (dpb_buffer_.field_pic_list[i * 2].is_reference == kUsedForShortTerm) {
                    dpb_buffer_.frame_buffer_list[i].is_reference = kUsedForShortTerm;
                    return PARSER_OK;
                }
            }

            if (dpb_buffer_.num_short_term + dpb_buffer_.num_long_term == p_sps->max_num_ref_frames) {
                int32_t min_frame_num_wrap = 0xFFFFFF; // more than the largest possible value of FrameNumWrap (2 ^ 16)
                int min_index = AVC_MAX_DPB_FRAMES;
                for (i = 0; i < dpb_buffer_.dpb_size; i++) {
                    if (dpb_buffer_.frame_buffer_list[i].is_reference == kUsedForShortTerm) {
                        if (dpb_buffer_.frame_buffer_list[i].frame_num_wrap < min_frame_num_wrap) {
                            min_frame_num_wrap = dpb_buffer_.frame_buffer_list[i].frame_num_wrap;
                            min_index = i;
                        }
                    }
                }
                if (min_index < dpb_buffer_.dpb_size) {
                    dpb_buffer_.frame_buffer_list[min_index].is_reference = kUnusedForReference;
                    dpb_buffer_.field_pic_list[min_index * 2].is_reference = kUnusedForReference;
                    dpb_buffer_.field_pic_list[min_index * 2 + 1].is_reference = kUnusedForReference;
                } else {
                    logger_.ErrorLog(MakeMsg("Could not find any short term ref picture."));
                    return PARSER_FAIL;
                }
                dpb_buffer_.num_short_term--;
            }
        }
    }

    return PARSER_OK;
}

ParserResult AvcVideoParser::BumpPicFromDpb() {
    int32_t min_poc_no_ref = 0x7FFFFFFF;  // largest possible POC value 2^31 - 1
    int32_t min_poc_ref = 0x7FFFFFFF;  // largest possible POC value 2^31 - 1
    int min_poc_pic_idx_no_ref = AVC_MAX_DPB_FRAMES;
    int min_poc_pic_idx_ref = AVC_MAX_DPB_FRAMES;
    int i;

    for (i = 0; i < dpb_buffer_.dpb_size; i++) {
        if (dpb_buffer_.frame_buffer_list[i].use_status) {
            if (dpb_buffer_.frame_buffer_list[i].is_reference) {
                if (dpb_buffer_.frame_buffer_list[i].pic_order_cnt < min_poc_ref && dpb_buffer_.frame_buffer_list[i].pic_output_flag) {
                    // Find the min-poc ref pic that has not been output yet.
                    min_poc_ref = dpb_buffer_.frame_buffer_list[i].pic_order_cnt;
                    min_poc_pic_idx_ref = i;
                }
            } else {
                if (dpb_buffer_.frame_buffer_list[i].pic_order_cnt < min_poc_no_ref) {
                    min_poc_no_ref = dpb_buffer_.frame_buffer_list[i].pic_order_cnt;
                    min_poc_pic_idx_no_ref = i;
                }
            }
        }
    }
    if (min_poc_pic_idx_no_ref >= dpb_buffer_.dpb_size) {
        logger_.ErrorLog(MakeMsg("Error! Could not find a non-reference buffer to bump."));
        return PARSER_OUT_OF_RANGE;
    }

    // Output any ref pics before (lower POC) the non-ref pic to be bumped out.
    while (min_poc_ref < min_poc_no_ref) {
        dpb_buffer_.frame_buffer_list[min_poc_pic_idx_ref].pic_output_flag = 0;
        if (dpb_buffer_.num_pics_needed_for_output > 0) {
            dpb_buffer_.num_pics_needed_for_output--;
            // Insert into output/display picture list
            if (pfn_display_picture_cb_) {
                if (num_output_pics_ >= dec_buf_pool_size_) {
                    logger_.ErrorLog(MakeMsg("Error! Decode buffer pool overflow!"));
                    return PARSER_OUT_OF_RANGE;
                } else {
                    output_pic_list_[num_output_pics_] = dpb_buffer_.frame_buffer_list[min_poc_pic_idx_ref].dec_buf_idx;
                    num_output_pics_++;
                }
            }
        }
        min_poc_ref = 0x7FFFFFFF;
        min_poc_pic_idx_ref = AVC_MAX_DPB_FRAMES;
        for (i = 0; i < dpb_buffer_.dpb_size; i++) {
            if (dpb_buffer_.frame_buffer_list[i].pic_output_flag && dpb_buffer_.frame_buffer_list[i].use_status && dpb_buffer_.frame_buffer_list[i].is_reference && dpb_buffer_.frame_buffer_list[i].pic_order_cnt < min_poc_ref) {
                min_poc_ref = dpb_buffer_.frame_buffer_list[i].pic_order_cnt;
                min_poc_pic_idx_ref = i;
            }
        }
    }

    // Mark as "not needed for output"
    if (dpb_buffer_.frame_buffer_list[min_poc_pic_idx_no_ref].pic_output_flag) {
        dpb_buffer_.frame_buffer_list[min_poc_pic_idx_no_ref].pic_output_flag = 0;
        if (dpb_buffer_.num_pics_needed_for_output > 0) {
            dpb_buffer_.num_pics_needed_for_output--;
        }

        // Insert into output/display picture list
        if (pfn_display_picture_cb_) {
            if (num_output_pics_ >= dec_buf_pool_size_) {
                logger_.ErrorLog(MakeMsg("Error! Decode buffer pool overflow!"));
                return PARSER_OUT_OF_RANGE;
            } else {
                output_pic_list_[num_output_pics_] = dpb_buffer_.frame_buffer_list[min_poc_pic_idx_no_ref].dec_buf_idx;
                num_output_pics_++;
            }
        }
    }
    // Remove it from DPB and mark unused for decode in decode buffer pool
    dpb_buffer_.frame_buffer_list[min_poc_pic_idx_no_ref].use_status = kNotUsed;
    decode_buffer_pool_[dpb_buffer_.frame_buffer_list[min_poc_pic_idx_no_ref].dec_buf_idx].use_status &= ~kFrameUsedForDecode;
    if (dpb_buffer_.dpb_fullness > 0 ) {
        dpb_buffer_.dpb_fullness--;
    }

    return PARSER_OK;
}

ParserResult AvcVideoParser::InsertCurrPicIntoDpb() {
    int i;
    // We have reserved a spot in DPB already.
    for (i = 0; i < dpb_buffer_.dpb_size; i++) {
        if (dpb_buffer_.frame_buffer_list[i].pic_idx == curr_pic_.pic_idx) {
            break;
        }
    }
    if (i < dpb_buffer_.dpb_size) {
        if (curr_pic_.pic_structure == kFrame) {
            dpb_buffer_.frame_buffer_list[i] = curr_pic_;
            if (dpb_buffer_.frame_buffer_list[i].pic_output_flag) {
                    dpb_buffer_.num_pics_needed_for_output++;
            }
            dpb_buffer_.dpb_fullness++;
            if (curr_pic_.is_reference == kUsedForShortTerm) {
                dpb_buffer_.num_short_term++;
            } else if (curr_pic_.is_reference == kUsedForLongTerm) {
                dpb_buffer_.num_long_term++;
            }

            AvcSeqParameterSet *p_sps = &sps_list_[active_sps_id_];
            if (p_sps->frame_mbs_only_flag == 0) { // picture adaptive frame-field (PICAFF)
                dpb_buffer_.field_pic_list[i * 2] = curr_pic_;
                dpb_buffer_.field_pic_list[i * 2].pic_structure = kTopField;
                dpb_buffer_.field_pic_list[i * 2 + 1] = curr_pic_;
                dpb_buffer_.field_pic_list[i * 2 + 1].pic_structure = kBottomField;
            }
        }
        else {
            if (second_field_ == 0) {
                dpb_buffer_.field_pic_list[i * 2] = curr_pic_;
                dpb_buffer_.frame_buffer_list[i] = curr_pic_; // Store several parameters
                dpb_buffer_.frame_buffer_list[i].pic_structure = kFrame;
                dpb_buffer_.frame_buffer_list[i].pic_output_flag = 0;
            } else {
                dpb_buffer_.field_pic_list[i * 2 + 1] = curr_pic_;
                if (curr_pic_.pic_structure == kTopField) {
                    dpb_buffer_.frame_buffer_list[i].top_field_order_cnt = curr_pic_.top_field_order_cnt;
                } else {
                    dpb_buffer_.frame_buffer_list[i].bottom_field_order_cnt = curr_pic_.bottom_field_order_cnt;
                }
                dpb_buffer_.frame_buffer_list[i].pic_order_cnt = dpb_buffer_.frame_buffer_list[i].top_field_order_cnt <= dpb_buffer_.frame_buffer_list[i].bottom_field_order_cnt ? dpb_buffer_.frame_buffer_list[i].top_field_order_cnt : dpb_buffer_.frame_buffer_list[i].bottom_field_order_cnt;
                dpb_buffer_.frame_buffer_list[i].pic_output_flag = curr_pic_.pic_output_flag;
                dpb_buffer_.frame_buffer_list[i].use_status = kFrameUsedForDecode;
                if (dpb_buffer_.frame_buffer_list[i].pic_output_flag) {
                    dpb_buffer_.num_pics_needed_for_output++;
                }
                dpb_buffer_.dpb_fullness++;
                if (curr_pic_.is_reference == kUsedForShortTerm) {
                    dpb_buffer_.num_short_term++;
                } else if (curr_pic_.is_reference == kUsedForLongTerm) {
                    dpb_buffer_.num_long_term++;
                }
            }
        }

        // Mark as used in decode buffer pool
        decode_buffer_pool_[curr_pic_.dec_buf_idx].use_status |= kFrameUsedForDecode;
        if (pfn_display_picture_cb_ && curr_pic_.pic_output_flag) {
            decode_buffer_pool_[curr_pic_.dec_buf_idx].use_status |= kFrameUsedForDisplay;
        }
        decode_buffer_pool_[curr_pic_.dec_buf_idx].pic_order_cnt = curr_pic_.pic_order_cnt;
        decode_buffer_pool_[curr_pic_.dec_buf_idx].pts = curr_pts_;
    } else {
        logger_.ErrorLog(MakeMsg("Could not find the reserved frame buffer for the current picture in DPB."));
        return PARSER_FAIL;
    }

#if DBGINFO
    PrintDpb();
#endif // DBGINFO
    return PARSER_OK;
}

ParserResult AvcVideoParser::FlushDpb() {
    if (dpb_buffer_.num_pics_needed_for_output) {
        // Mark all reference pictures as "unused for reference
        for (int i = 0; i < AVC_MAX_DPB_FRAMES; i++) {
            dpb_buffer_.frame_buffer_list[i].is_reference = kUnusedForReference;
            dpb_buffer_.field_pic_list[i * 2].is_reference = kUnusedForReference;
            dpb_buffer_.field_pic_list[i * 2 + 1].is_reference = kUnusedForReference;
        }
        // Bump the remaining pictures
        while (dpb_buffer_.num_pics_needed_for_output) {
            if (BumpPicFromDpb() != PARSER_OK) {
                return PARSER_FAIL;
            }
        }
    }
    if (pfn_display_picture_cb_ && num_output_pics_ > 0) {
        if (OutputDecodedPictures(true) != PARSER_OK) {
            return PARSER_FAIL;
        }
    }

    // Empty DPB
    for (int i = 0; i < AVC_MAX_DPB_FRAMES; i++) {
        dpb_buffer_.frame_buffer_list[i].use_status = kNotUsed;
        dpb_buffer_.field_pic_list[i * 2].use_status = kNotUsed;
        dpb_buffer_.field_pic_list[i * 2 + 1].use_status = kNotUsed;
        decode_buffer_pool_[dpb_buffer_.frame_buffer_list[i].dec_buf_idx].use_status = kNotUsed;
    }
    return PARSER_OK;
}

#if DBGINFO
void AvcVideoParser::PrintSps(AvcSeqParameterSet *p_sps) {
    uint32_t i, j;
    
    MSG("=======================");
    MSG("Sequence parameter set: ");
    MSG("=======================");
    MSG("profile_idc = " << p_sps->profile_idc);
    MSG("level_idc = " << p_sps->level_idc);
    MSG("chroma_format_idc = " << p_sps->chroma_format_idc);
    MSG("separate_colour_plane_flag = " << p_sps->separate_colour_plane_flag);
    MSG("bit_depth_luma_minus8 = " << p_sps->bit_depth_luma_minus8);
    MSG("bit_depth_chroma_minus8 = " << p_sps->bit_depth_chroma_minus8);
    MSG("qpprime_y_zero_transform_bypass_flag = " << p_sps->qpprime_y_zero_transform_bypass_flag);
    MSG("seq_scaling_matrix_present_flag = " << p_sps->seq_scaling_matrix_present_flag);
    
    MSG_NO_NEWLINE("seq_scaling_list_present_flag[12]: ");
    for (i = 0; i < 12; i++) {
        MSG_NO_NEWLINE(" " << p_sps->seq_scaling_list_present_flag[i]);
    }
    MSG("");
    
    MSG("scaling_list_4x4[6][16]:");
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 16; j++) {
            MSG_NO_NEWLINE(" " << p_sps->scaling_list_4x4[i][j]);
        }
        MSG("");
    }
    MSG("");
    
    MSG("scaling_list_8x8[6][64]:");
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 64; j++) {
            MSG_NO_NEWLINE(" " << p_sps->scaling_list_8x8[i][j]);
        }
        MSG("");
    }
    MSG("");
    
    MSG_NO_NEWLINE("use_default_scaling_matrix_4x4_flag[6]: ");
    for (i = 0; i < 6; i++) {
        MSG_NO_NEWLINE(" " << p_sps->use_default_scaling_matrix_4x4_flag[i]);
    }
    MSG("");
    
    MSG_NO_NEWLINE("use_default_scaling_matrix_8x8_flag[6]: ");
    for (i = 0; i < 6; i++) {
        MSG_NO_NEWLINE(" " << p_sps->use_default_scaling_matrix_8x8_flag[i]);
    }
    MSG("");
    
    MSG("log2_max_frame_num_minus4 = " << p_sps->log2_max_frame_num_minus4);
    MSG("pic_order_cnt_type = " << p_sps->pic_order_cnt_type);
    MSG("log2_max_pic_order_cnt_lsb_minus4 = " << p_sps->log2_max_pic_order_cnt_lsb_minus4);
    MSG("delta_pic_order_always_zero_flag = " << p_sps->delta_pic_order_always_zero_flag);
    MSG("offset_for_non_ref_pic = " << p_sps->offset_for_non_ref_pic);
    MSG("offset_for_top_to_bottom_field = " << p_sps->offset_for_top_to_bottom_field);
    MSG("num_ref_frames_in_pic_order_cnt_cycle = " << p_sps->num_ref_frames_in_pic_order_cnt_cycle);
    MSG("offset_for_ref_frame[]: ....");
    MSG("max_num_ref_frames = " << p_sps->max_num_ref_frames);
    MSG("gaps_in_frame_num_value_allowed_flag = " << p_sps->gaps_in_frame_num_value_allowed_flag);
    MSG("pic_width_in_mbs_minus1 = " << p_sps->pic_width_in_mbs_minus1);
    MSG("pic_height_in_map_units_minus1 = " << p_sps->pic_height_in_map_units_minus1);
    MSG("frame_mbs_only_flag = " << p_sps->frame_mbs_only_flag);
    MSG("mb_adaptive_frame_field_flag = " << p_sps->mb_adaptive_frame_field_flag);
    MSG("direct_8x8_inference_flag = " << p_sps->direct_8x8_inference_flag);
    MSG("frame_cropping_flag = " << p_sps->frame_cropping_flag);
    MSG("frame_crop_left_offset = " << p_sps->frame_crop_left_offset);
    MSG("frame_crop_right_offset = " << p_sps->frame_crop_right_offset);
    MSG("frame_crop_top_offset = " << p_sps->frame_crop_top_offset);
    MSG("frame_crop_bottom_offset = " << p_sps->frame_crop_bottom_offset);
    MSG("vui_parameters_present_flag = " << p_sps->vui_parameters_present_flag);
    MSG("vui_seq_parameters: ....");
    MSG("");
}

void AvcVideoParser::PrintPps(AvcPicParameterSet *p_pps) {
    uint32_t i, j;
    
    MSG("=======================");
    MSG("Picture parameter set: ");
    MSG("=======================");
    MSG("pic_parameter_set_id = " << p_pps->pic_parameter_set_id);
    MSG("seq_parameter_set_id = " << p_pps->seq_parameter_set_id);
    MSG("entropy_coding_mode_flag = " << p_pps->entropy_coding_mode_flag);
    MSG("bottom_field_pic_order_in_frame_present_flag = " << p_pps->bottom_field_pic_order_in_frame_present_flag);
    MSG("num_slice_groups_minus1 = " << p_pps->num_slice_groups_minus1);
    MSG("slice_group_map_type = " << p_pps->slice_group_map_type);
    MSG("run_length_minus1[]: ....");
    MSG("top_left[]: ....");
    MSG("bottom_right[]: ....");
    MSG("slice_group_change_direction_flag = " << p_pps->slice_group_change_direction_flag);
    MSG("slice_group_change_rate_minus1 = " << p_pps->slice_group_change_rate_minus1);
    MSG("pic_size_in_map_units_minus1 = " << p_pps->pic_size_in_map_units_minus1);
    MSG("slice_group_id[]: ....");
    MSG("num_ref_idx_l0_default_active_minus1 = " << p_pps->num_ref_idx_l0_default_active_minus1);
    MSG("num_ref_idx_l1_default_active_minus1 = " << p_pps->num_ref_idx_l1_default_active_minus1);
    MSG("weighted_pred_flag = " << p_pps->weighted_pred_flag);
    MSG("weighted_bipred_idc = " << p_pps->weighted_bipred_idc);
    MSG("pic_init_qp_minus26 = " << p_pps->pic_init_qp_minus26);
    MSG("pic_init_qs_minus26 = " << p_pps->pic_init_qs_minus26);
    MSG("chroma_qp_index_offset = " << p_pps->chroma_qp_index_offset);
    MSG("deblocking_filter_control_present_flag = " << p_pps->deblocking_filter_control_present_flag);
    MSG("constrained_intra_pred_flag = " << p_pps->constrained_intra_pred_flag);
    MSG("redundant_pic_cnt_present_flag = " << p_pps->redundant_pic_cnt_present_flag);
    MSG("transform_8x8_mode_flag = " << p_pps->transform_8x8_mode_flag);
    MSG("pic_scaling_matrix_present_flag = " << p_pps->pic_scaling_matrix_present_flag);
    
    MSG_NO_NEWLINE("pic_scaling_list_present_flag[12]: ");
    for (i = 0; i < 12; i++) {
        MSG_NO_NEWLINE(" " << p_pps->pic_scaling_list_present_flag[i]);
    }
    MSG("");
    MSG("scaling_list_4x4[6][16]:");
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 16; j++) {
            MSG_NO_NEWLINE(" " << p_pps->scaling_list_4x4[i][j]);
        }
        MSG("");
    }
    MSG("");
    
    MSG("scaling_list_8x8[6][64]:");
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 64; j++) {
            MSG_NO_NEWLINE(" " << p_pps->scaling_list_8x8[i][j]);
        }
        MSG("");
    }
    MSG("");
    
    MSG_NO_NEWLINE("use_default_scaling_matrix_4x4_flag[6]: ");
    for (i = 0; i < 6; i++) {
        MSG_NO_NEWLINE(" " << p_pps->use_default_scaling_matrix_4x4_flag[i]);
    }
    MSG("");
    
    MSG_NO_NEWLINE("use_default_scaling_matrix_8x8_flag[6]: ");
    for (i = 0; i < 6; i++) {
        MSG_NO_NEWLINE(" " << p_pps->use_default_scaling_matrix_8x8_flag[i]);
    }
    MSG("");
    
    MSG("second_chroma_qp_index_offset = " << p_pps->second_chroma_qp_index_offset);
    MSG("");
}

void AvcVideoParser::PrintSliceHeader(AvcSliceHeader *p_slice_header) {
    uint32_t j;
        
    MSG("======================");
    MSG("Slice header");
    MSG("======================");
    MSG("first_mb_in_slice = " << p_slice_header->first_mb_in_slice);
    MSG("slice_type = " << p_slice_header->slice_type);
    MSG("pic_parameter_set_id = " << p_slice_header->pic_parameter_set_id);
    MSG("frame_num = " << p_slice_header->frame_num);
    MSG("field_pic_flag = " << p_slice_header->field_pic_flag);
    MSG("bottom_field_flag = " << p_slice_header->bottom_field_flag);
    MSG("idr_pic_id = " << p_slice_header->idr_pic_id);
    MSG("pic_order_cnt_lsb = " << p_slice_header->pic_order_cnt_lsb);
    MSG("delta_pic_order_cnt_bottom = " << p_slice_header->delta_pic_order_cnt_bottom);
    MSG("delta_pic_order_cnt[2] =  " << p_slice_header->delta_pic_order_cnt[0] << ", " << p_slice_header->delta_pic_order_cnt[1]);
    MSG("redundant_pic_cnt = " << p_slice_header->redundant_pic_cnt);
    MSG("direct_spatial_mv_pred_flag = " << p_slice_header->direct_spatial_mv_pred_flag);
    MSG("num_ref_idx_active_override_flag = " << p_slice_header->num_ref_idx_active_override_flag);
    MSG("num_ref_idx_l0_active_minus1 = " << p_slice_header->num_ref_idx_l0_active_minus1);
    MSG("num_ref_idx_l1_active_minus1 = " << p_slice_header->num_ref_idx_l1_active_minus1);
    
    MSG("Reference picture list modification:");
    MSG("ref_pic_list_modification_flag_l0 = " << p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l0);
    if ( p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l0 ){
        MSG("Modification operations for list 0: ");
        for (j = 0; j < AVC_MAX_REF_PICTURE_NUM; j++) {
            MSG_NO_NEWLINE("(" << p_slice_header->ref_pic_list.modification_l0[j].modification_of_pic_nums_idc << ", " << p_slice_header->ref_pic_list.modification_l0[j].abs_diff_pic_num_minus1 << ", " << p_slice_header->ref_pic_list.modification_l0[j].long_term_pic_num << ") ");
        }
        MSG("");
    }
    MSG("ref_pic_list_modification_flag_l1 = " << p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l1);
    if ( p_slice_header->ref_pic_list.ref_pic_list_modification_flag_l1 ) {
        MSG("Modification operations for list 1: ");
        for (j = 0; j < AVC_MAX_REF_PICTURE_NUM; j++) {
            MSG_NO_NEWLINE("(" << p_slice_header->ref_pic_list.modification_l1[j].modification_of_pic_nums_idc << ", " << p_slice_header->ref_pic_list.modification_l1[j].abs_diff_pic_num_minus1 << ", " << p_slice_header->ref_pic_list.modification_l1[j].long_term_pic_num << ") ");
        }
        MSG("");
    }
    
    MSG("pred_weight_table: ....");
    
    MSG("Decoded reference picture marking:");
    AvcDecRefPicMarking *refMarking = &(p_slice_header->dec_ref_pic_marking);
    MSG("no_output_of_prior_pics_flag = " << refMarking->no_output_of_prior_pics_flag);
    MSG("long_term_reference_flag = " << refMarking->long_term_reference_flag);
    MSG("adaptive_ref_pic_marking_mode_flag = " << refMarking->adaptive_ref_pic_marking_mode_flag);
    if ( refMarking->adaptive_ref_pic_marking_mode_flag ) {
        MSG("mmco_count = " << refMarking->mmco_count);
        for (j = 0; j < AVC_MAX_REF_PICTURE_NUM; j++) {
            MSG_NO_NEWLINE("(" << refMarking->mmco[j].memory_management_control_operation << ", " << refMarking->mmco[j].difference_of_pic_nums_minus1 << ", " << refMarking->mmco[j].long_term_pic_num << ", " << refMarking->mmco[j].long_term_frame_idx << ", " << refMarking->mmco[j].max_long_term_frame_idx_plus1 << ") ");
        }
        MSG("");
    }
    
    MSG("cabac_init_idc = " << p_slice_header->cabac_init_idc);
    MSG("slice_qp_delta = " << p_slice_header->slice_qp_delta);
    MSG("sp_for_switch_flag = " << p_slice_header->sp_for_switch_flag);
    MSG("slice_qs_delta = " << p_slice_header->slice_qs_delta);
    MSG("disable_deblocking_filter_idc = " << p_slice_header->disable_deblocking_filter_idc);
    MSG("slice_alpha_c0_offset_div2 = " << p_slice_header->slice_alpha_c0_offset_div2);
    MSG("slice_beta_offset_div2 = " << p_slice_header->slice_beta_offset_div2);
    MSG("slice_group_change_cycle = " << p_slice_header->slice_group_change_cycle);
}

void AvcVideoParser::PrintDpb() {
    uint32_t i;

    MSG("=======================");
    MSG("DPB buffer content: ");
    MSG("=======================");
    MSG("dpb_size = " << dpb_buffer_.dpb_size);
    MSG("num_short_term = " << dpb_buffer_.num_short_term);
    MSG("num_long_term = " << dpb_buffer_.num_long_term);
    MSG("num_short_term_ref_fields = " << dpb_buffer_.num_short_term_ref_fields);
    MSG("num_long_term_ref_fields = " << dpb_buffer_.num_long_term_ref_fields);
    MSG("second_field_ = " << second_field_);
    MSG("num_pics_needed_for_output = " << dpb_buffer_.num_pics_needed_for_output);
    MSG("dpb_fullness = " << dpb_buffer_.dpb_fullness);
    MSG("Frame buffer store:");
    for (i = 0; i < AVC_MAX_DPB_FRAMES; i++) {
        AvcPicture *p_buf = &dpb_buffer_.frame_buffer_list[i];
        MSG("Frame buffer " << i << ": pic_idx = " << p_buf->pic_idx << ", dec_buf_idx = " << p_buf->dec_buf_idx << ", pic_structure = " << p_buf->pic_structure << ", pic_order_cnt = " << p_buf->pic_order_cnt << ", top_field_order_cnt = " << p_buf->top_field_order_cnt << ", bottom_field_order_cnt = " << p_buf->bottom_field_order_cnt << ", frame_num = " << p_buf->frame_num << ", frame_num_wrap = " << p_buf->frame_num_wrap << ", pic_num = " << p_buf->pic_num << ", long_term_pic_num = " << p_buf->long_term_pic_num << ", long_term_frame_idx = " << p_buf->long_term_frame_idx << ", is_reference = " << p_buf->is_reference << ", use_status = " << p_buf->use_status << ", pic_output_flag = " << p_buf->pic_output_flag);
    }
    MSG("");
    MSG("Field picture store:");
    for (i = 0; i < AVC_MAX_DPB_FIELDS; i++) {
        AvcPicture *p_buf = &dpb_buffer_.field_pic_list[i];
        MSG("Field picture " << i << ": pic_idx = " << p_buf->pic_idx << ", pic_structure = " << p_buf->pic_structure << ", pic_order_cnt = " << p_buf->pic_order_cnt << ", top_field_order_cnt = " << p_buf->top_field_order_cnt << ", bottom_field_order_cnt = " << p_buf->bottom_field_order_cnt << ", frame_num = " << p_buf->frame_num << ", frame_num_wrap = " << p_buf->frame_num_wrap << ", pic_num = " << p_buf->pic_num << ", long_term_pic_num = " << p_buf->long_term_pic_num << ", long_term_frame_idx = " << p_buf->long_term_frame_idx << ", is_reference = " << p_buf->is_reference << ", use_status = " << p_buf->use_status << ", pic_output_flag = " << p_buf->pic_output_flag);
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

void AvcVideoParser::PrintVappiBufInfo() {
    RocdecAvcPicParams *p_pic_param = &dec_pic_params_.pic_params.avc;
    MSG("=======================");
    MSG("VAAPI Buffer Info: ");
    MSG("=======================");
    MSG("Current buffer:");
    MSG_NO_NEWLINE("pic_idx = " << p_pic_param->curr_pic.pic_idx << ", frame_idx = " << p_pic_param->curr_pic.frame_idx << ", top_field_order_cnt = " << p_pic_param->curr_pic.top_field_order_cnt << ", bottom_field_order_cnt = " << std::dec << p_pic_param->curr_pic.bottom_field_order_cnt);
    MSG(", flags = 0x" << std::hex << p_pic_param->curr_pic.flags);
    MSG(std::dec);

    MSG("Reference pictures:");
    for (int i = 0; i < 16; i++) {
        RocdecAvcPicture *p_ref_pic = &p_pic_param->ref_frames[i];
        MSG_NO_NEWLINE("Ref pic " << i << ": " << "pic_idx = " << p_ref_pic->pic_idx << ", frame_idx = " << p_ref_pic->frame_idx << ", top_field_order_cnt = " << p_ref_pic->top_field_order_cnt << ", bottom_field_order_cnt = " << p_ref_pic->bottom_field_order_cnt);
        MSG(", flags = 0x" << std::hex << p_ref_pic->flags);
        MSG_NO_NEWLINE(std::dec);
    }

    MSG("Slice ref lists:")
    for (int slice_index = 0; slice_index < num_slices_; slice_index++) {
        RocdecAvcSliceParams *p_slice_param = &slice_param_list_[slice_index];
        AvcSliceInfo *p_slice_info = &slice_info_list_[slice_index];
        MSG("Slice " << slice_index << " ref list 0:");
        for (int i = 0; i <= p_slice_info->slice_header.num_ref_idx_l0_active_minus1; i++) {
            MSG("Index " << i << ": pic_idx = " << p_slice_param->ref_pic_list_0[i].pic_idx << ", frame_idx = " << p_slice_param->ref_pic_list_0[i].frame_idx << ", top_poc = " << p_slice_param->ref_pic_list_0[i].top_field_order_cnt << ", bottom_poc = " << p_slice_param->ref_pic_list_0[i].bottom_field_order_cnt);
        }
        if (p_slice_info->slice_header.slice_type == kAvcSliceTypeB || p_slice_info->slice_header.slice_type == kAvcSliceTypeB_6) {
            MSG("Slice " << slice_index << " ref list 1: ");
            for (int i = 0; i <= p_slice_info->slice_header.num_ref_idx_l1_active_minus1; i++) {
                MSG("Index " << i << ": pic_idx = " << p_slice_param->ref_pic_list_1[i].pic_idx << ", frame_idx = " << p_slice_param->ref_pic_list_1[i].frame_idx << ", top_poc = " << p_slice_param->ref_pic_list_1[i].top_field_order_cnt << ", bottom_poc = " << p_slice_param->ref_pic_list_1[i].bottom_field_order_cnt);
            }
        }
        MSG("");
    }
}
#endif // DBGINFO
