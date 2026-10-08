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

#include "roc_video_parser.h"

RocVideoParser::RocVideoParser() {
    pic_count_ = 0;
    pic_width_ = 0;
    pic_height_ = 0;
    bit_depth_luma_minus8_ = 0;
    bit_depth_chroma_minus8_ = 0;
    new_seq_activated_ = false;
    frame_rate_.numerator = 0;
    frame_rate_.denominator = 0;
    curr_pts_ = 0;

    sei_rbsp_buf_ = nullptr;
    sei_rbsp_buf_size_ = 0;
    sei_payload_buf_ = nullptr;
    sei_payload_buf_size_ = 0;
    sei_message_list_.assign(INIT_SEI_MESSAGE_COUNT, {0});
}

RocVideoParser::~RocVideoParser() {
    if (sei_rbsp_buf_) {
        delete [] sei_rbsp_buf_;
    }
    if (sei_payload_buf_) {
        delete [] sei_payload_buf_;
    }
}

/**
 * @brief Initializes any parser related stuff for all parsers
 * 
 * @return rocDecStatus : ROCDEC_SUCCESS on success
 */
rocDecStatus RocVideoParser::Initialize(RocdecParserParams *pParams) {
    FunctionEntryLogWithArgs(g_rocdec_logger, RocDecFmtPtr(pParams));
    if(pParams == nullptr) {
        CriticalLog(g_rocdec_logger, ROCDEC_STR("Parser parameters are not set for the parser"));
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_NOT_INITIALIZED;
    }
    // Initialize callback function pointers
    pfn_sequence_cb_         = pParams->pfn_sequence_callback;     /**< Called before decoding frames and/or whenever there is a fmt change */
    pfn_decode_picture_cb_  = pParams->pfn_decode_picture;        /**< Called when a picture is ready to be decoded (decode order)         */
    pfn_display_picture_cb_ = pParams->pfn_display_picture;       /**< Called whenever a picture is ready to be displayed (display order)  */
    pfn_get_sei_message_cb_ = pParams->pfn_get_sei_msg;           /**< Called when all SEI messages are parsed for particular frame        */

    parser_params_ = *pParams;

    dec_buf_pool_size_ = parser_params_.max_num_decode_surfaces;
    decode_buffer_pool_.resize(dec_buf_pool_size_, {0});
    output_pic_list_.resize(dec_buf_pool_size_, 0xFF);
    InitDecBufPool();

    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

void RocVideoParser::InitDecBufPool() {
    for (int i = 0; i < dec_buf_pool_size_; i++) {
        decode_buffer_pool_[i].use_status = kNotUsed;
        decode_buffer_pool_[i].pic_order_cnt = 0;
        output_pic_list_[i] = 0xFF;
    }
    num_output_pics_ = 0;
}

void RocVideoParser::CheckAndAdjustDecBufPoolSize(int dpb_size) {
    int min_dec_buf_pool_size = dpb_size + (parser_params_.max_display_delay > DECODE_BUF_POOL_EXTENSION ? parser_params_.max_display_delay : DECODE_BUF_POOL_EXTENSION);
    // If DPB size decreases, we keep the existing pool and skip reconfiguration.
    if ( dec_buf_pool_size_ < min_dec_buf_pool_size) {
        dec_buf_pool_size_ = min_dec_buf_pool_size;
        decode_buffer_pool_.resize(dec_buf_pool_size_, {0});
        output_pic_list_.resize(dec_buf_pool_size_, 0xFF);
    }
}

ParserResult RocVideoParser::OutputDecodedPictures(bool no_delay) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(no_delay));
    RocdecParserDispInfo disp_info = {0};
    disp_info.progressive_frame = 1; // not used
    disp_info.top_field_first = 1; // not used

    int disp_delay = no_delay ? 0 : parser_params_.max_display_delay;
    if (num_output_pics_ > disp_delay) {
        int num_disp = num_output_pics_ - disp_delay;
        for (int i = 0; i < num_disp; i++) {
            disp_info.picture_index = output_pic_list_[i];
            disp_info.pts = decode_buffer_pool_[output_pic_list_[i]].pts;
            pfn_display_picture_cb_(parser_params_.user_data, &disp_info);
            decode_buffer_pool_[output_pic_list_[i]].use_status &= ~kFrameUsedForDisplay;
        }
        num_output_pics_ = disp_delay;
        // Shift the remaining frames to the top
        if (num_output_pics_) {
            for (int i = 0; i < num_output_pics_; i++) {
                output_pic_list_[i] = output_pic_list_[i + num_disp];
            }
        }
    }
    FunctionExitLog(g_rocdec_logger);
    return PARSER_OK;
}

ParserResult RocVideoParser::GetNalUnit() {
    bool start_code_found = false;

    nal_unit_size_ = 0;
    curr_start_code_offset_ = next_start_code_offset_;  // save the current start code offset

    // A start code is three bytes, so there is nothing to scan in a smaller buffer. The check
    // belongs here rather than in the loop condition: it also keeps pic_data_size_ - 2 from
    // wrapping, both operands being uint32_t, and it does not depend on the callers resetting
    // start_code_num_ for the return below to be reached.
    if (pic_data_size_ < 3) {
        ErrorLog(g_rocdec_logger, "Picture data is " + ROCDEC_TOSTR(pic_data_size_) + " bytes, too short to hold a start code.");
        return PARSER_INVALID_FORMAT;
    }

    // Search for the next start code
    while (curr_byte_offset_ < pic_data_size_ - 2) {
        if (pic_data_buffer_ptr_[curr_byte_offset_] == 0 && pic_data_buffer_ptr_[curr_byte_offset_ + 1] == 0 && pic_data_buffer_ptr_[curr_byte_offset_ + 2] == 0x01) {
            curr_start_code_offset_ = next_start_code_offset_;  // save the current start code offset

            start_code_found = true;
            start_code_num_++;
            next_start_code_offset_ = curr_byte_offset_;
            // Move the pointer 3 bytes forward
            curr_byte_offset_ += 3;

            // For the very first NAL unit, search for the next start code (or reach the end of frame)
            if (start_code_num_ == 1 ) {
                start_code_found = false;
                curr_start_code_offset_ = next_start_code_offset_;
                continue;
            } else {
                break;
            }
        }
        curr_byte_offset_++;
    }
    if (start_code_num_ == 0) {
        // No NAL unit in the frame data
        return PARSER_NOT_FOUND;
    }
    // Defensive; not reachable with the current callers, which reset both offsets per picture and
    // only ever assign them values the scan bound above already constrains. Kept because the
    // subtractions below are unsigned: an end offset below the start would yield a huge size
    // rather than a negative one, and the nal_unit_size_ floor the callers apply before copying
    // out of the NAL unit would not catch that.
    if (curr_start_code_offset_ > pic_data_size_ ||
        (start_code_found && next_start_code_offset_ < curr_start_code_offset_)) {
        ErrorLog(g_rocdec_logger, "Start code offsets are out of order for the current picture.");
        nal_unit_size_ = 0;
        return PARSER_INVALID_FORMAT;
    }
    if (start_code_found) {
        nal_unit_size_ = next_start_code_offset_ - curr_start_code_offset_;
        return PARSER_OK;
    } else {
        nal_unit_size_ = pic_data_size_ - curr_start_code_offset_;
        return PARSER_EOF;
    }
}

ParserResult RocVideoParser::EbspToRbsp(uint8_t *streamBuffer,size_t begin_bytepos, size_t end_bytepos, size_t *p_rbsp_size) {
    int count = 0;  // length of the current run of zero bytes, 0 to ZEROBYTES_SHORTSTARTCODE
    *p_rbsp_size = 0;
    // An end before the start describes no range at all. Reporting end_bytepos as the length and
    // PARSER_OK was the same mistake this function was changed to stop making, even though all
    // nine callers pass begin_bytepos of 0 and cannot reach it.
    if (end_bytepos < begin_bytepos) {
        return PARSER_INVALID_ARG;
    }
    uint8_t *streamBuffer_i = streamBuffer + begin_bytepos;
    uint8_t *streamBuffer_end = streamBuffer + end_bytepos;
    size_t reduce_count = 0;  // bytes discarded, subtracted from a size_t span below
    for (; streamBuffer_i != streamBuffer_end; ) { 
        //starting from begin_bytepos to avoid header information
        //in NAL unit, 0x000000, 0x000001 or 0x000002 shall not occur at any uint8_t-aligned position
        uint8_t tmp =* streamBuffer_i;
        if (count == ZEROBYTES_SHORTSTARTCODE) {
            if (tmp == 0x03) {
                //check the 4th uint8_t after 0x000003, except when cabac_zero_word is used, in which case the last three bytes of this NAL unit must be 0x000003
                if ((streamBuffer_i + 1 != streamBuffer_end) && (streamBuffer_i[1] > 0x03)) {
                    ErrorLog(g_rocdec_logger, "Malformed emulation prevention sequence in the NAL unit.");
                    return PARSER_INVALID_ARG;
                }
                //if cabac_zero_word is used, the final uint8_t of this NAL unit(0x03) is discarded, and the last two bytes of RBSP must be 0x0000
                if (streamBuffer_i + 1 == streamBuffer_end) {
                    reduce_count++;  // discarded as well, so it is not part of the RBSP
                    break;
                }
                memmove(streamBuffer_i, streamBuffer_i + 1, streamBuffer_end-streamBuffer_i - 1);
                streamBuffer_end--;
                reduce_count++;
                count = 0;
                tmp = *streamBuffer_i;
            } else if (tmp < 0x03) {
            }
        }
        if (tmp == 0x00) {
            count++;
        } else {
            count = 0;
        }
        streamBuffer_i++;
    }
    // Every discarded byte shortens the data, so the RBSP is the EBSP less reduce_count. Adding
    // it instead reported more than the EBSP ever held: 00 00 03 01 is 3 bytes of RBSP but was
    // reported as 5. For a full rbsp_buf_ that told the parse functions the buffer was larger
    // than it is, which is exactly the kind of bound the rest of this change relies on.
    *p_rbsp_size = end_bytepos - (begin_bytepos + reduce_count);
    return PARSER_OK;
}

ParserResult RocVideoParser::ParseSeiMessage(uint8_t *nalu, size_t size) {
    size_t offset = 0; // byte offset
    // Accumulated in size_t so that a long run of ff_bytes cannot wrap the running total before
    // it is range checked below.
    size_t payload_type;
    size_t payload_size;

    // SEI is supplemental and does not affect the decode, so a message running past the end of
    // the NAL unit stops the parse here and keeps whatever was read cleanly. The result is
    // reported for the record; the callers log it and carry on with the picture.
    do {
        payload_type = 0;
        while (offset < size && nalu[offset] == 0xFF) {
            payload_type += 255;  // ff_byte
            offset++;
        }
        if (offset >= size) {
            ErrorLog(g_rocdec_logger, "SEI payload type extends past the end of the NAL unit.");
            return PARSER_OUT_OF_RANGE;
        }
        payload_type += nalu[offset];  // last_payload_type_byte
        offset++;

        payload_size = 0;
        while (offset < size && nalu[offset] == 0xFF) {
            payload_size += 255;  // ff_byte
            offset++;
        }
        if (offset >= size) {
            ErrorLog(g_rocdec_logger, "SEI payload size extends past the end of the NAL unit.");
            return PARSER_OUT_OF_RANGE;
        }
        payload_size += nalu[offset];  // last_payload_size_byte
        offset++;

        if (payload_size > size - offset) {
            ErrorLog(g_rocdec_logger, "SEI payload size (" + ROCDEC_TOSTR(payload_size) + ") exceeds the " + ROCDEC_TOSTR(size - offset) + " bytes left in the NAL unit.");
            return PARSER_OUT_OF_RANGE;
        }

        // We start with INIT_SEI_MESSAGE_COUNT. Should be enough for normal use cases. If not, resize.
        if((sei_message_count_ + 1) > sei_message_list_.size()) {
            sei_message_list_.resize((sei_message_count_ + 1));
        }
        // Both fields of the public RocdecSeiMessage are narrower than the accumulators, so both
        // conversions are spelled out. sei_message_type is uint8_t, so a type above 255 truncates
        // and the value stored is not the one parsed; payload_type is still accumulated wide so
        // that the running total cannot wrap before the range checks above act on it.
        // sei_message_size is uint32_t and payload_size was bounded by the size - offset check
        // above, so that one is exact.
        sei_message_list_[sei_message_count_].sei_message_type = static_cast<uint8_t>(payload_type);
        sei_message_list_[sei_message_count_].sei_message_size = static_cast<uint32_t>(payload_size);

        if (sei_payload_buf_) {
            if ((payload_size + sei_payload_size_) > sei_payload_buf_size_) {
                // Grow geometrically. Fitting the capacity to exactly what is needed leaves it
                // equal to sei_payload_size_ once the payload below is appended, so every later
                // message carrying any payload re-enters this branch and copies the whole
                // accumulated payload again. The messages in one picture are only bounded by the
                // packet size, so that is quadratic in the packet size.
                size_t needed = sei_payload_size_ + payload_size;
                size_t new_size = sei_payload_buf_size_ ? sei_payload_buf_size_ : INIT_SEI_PAYLOAD_BUF_SIZE;
                while (new_size < needed) {
                    new_size *= 2;
                }
                size_t capacity = new_size >= needed ? new_size : needed;
                // sei_payload_buf_size_ is uint32_t, and it is what the allocation below is sized
                // from and what the two copies are bounded by. Narrowing a capacity past 4 GB
                // would under allocate and let both copies run past the new buffer, so fail here
                // rather than record a size that is not the one that was needed.
                if (capacity > 0xFFFFFFFFULL) {
                    ErrorLog(g_rocdec_logger, "SEI payload buffer would exceed the 4 GB size field.");
                    return PARSER_OUT_OF_RANGE;
                }
                sei_payload_buf_size_ = static_cast<uint32_t>(capacity);
                uint8_t *tmp_ptr = new uint8_t [sei_payload_buf_size_];
                memcpy(tmp_ptr, sei_payload_buf_, sei_payload_size_); // save the existing payload
                delete [] sei_payload_buf_;
                sei_payload_buf_ = tmp_ptr;
            }
        } else {
            // First payload, sei_payload_size_ is 0. The narrowing is explicit because
            // payload_size is size_t: it was checked above against what is left of the NAL unit,
            // which is itself bounded by the uint32_t size the NAL unit was copied with, so a
            // single payload always fits.
            sei_payload_buf_size_ = payload_size > INIT_SEI_PAYLOAD_BUF_SIZE ? static_cast<uint32_t>(payload_size) : INIT_SEI_PAYLOAD_BUF_SIZE;
            sei_payload_buf_ = new uint8_t [sei_payload_buf_size_];
        }
        // Append the current payload to sei_payload_buf_
        memcpy(sei_payload_buf_ + sei_payload_size_, nalu + offset, payload_size);

        // The running total cannot pass sei_payload_buf_size_, which the branches above just
        // sized to hold it and confirmed fits in uint32_t, so the narrowing is safe here.
        sei_payload_size_ += static_cast<uint32_t>(payload_size);
        sei_message_count_++;

        offset += payload_size;
    } while (offset < size && nalu[offset] != 0x80);
    return PARSER_OK;
}
