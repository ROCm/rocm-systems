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

#include <string.h>
#include "es_reader.h"
#include "hevc_defines.h"
#include "avc_defines.h"
#include "av1_defines.h"
#include "vp9_defines.h"

RocVideoESParser::RocVideoESParser(const char *input_file_path) {
    logger_.SetLogLevel(kRocDecLogError);

    p_stream_file_.open(input_file_path, std::ifstream::in | std::ifstream::binary);
    if (!p_stream_file_) {
        logger_.CriticalLog(MakeMsg("Failed to open the bitstream file."));
    }
    end_of_file_ = false;
    end_of_stream_ = false;
    read_ptr_ = 0;
    write_ptr_ = 0;
    curr_byte_offset_ = read_ptr_;
    pic_data_.assign(INIT_PIC_DATA_SIZE, 0);
    pic_data_size_ = 0;
    curr_pic_end_ = 0;
    next_pic_start_ = 0;
    num_pictures_ = 0;
    num_start_code_ = 0;
    curr_start_code_offset_ = 0;
    next_start_code_offset_ = 0;
    obu_byte_offset_ = 0;
    obu_size_ = 0;
    num_td_obus_ = 0;
    num_temp_units_ = 0;
    ivf_file_header_read_ = false;

    stream_type_ = ProbeStreamType();
    bit_depth_ = 8;
}

RocVideoESParser::~RocVideoESParser() {
    if (p_stream_file_) {
        p_stream_file_.close();
    }
}

int RocVideoESParser::GetDataSizeInRB() {
    if (read_ptr_ == write_ptr_) {
        return 0;
    } else if (read_ptr_ < write_ptr_) {
        return write_ptr_ - read_ptr_;
    } else {
        return BS_RING_SIZE - read_ptr_ + write_ptr_;
    }
}

int RocVideoESParser::FetchBitStream()
{
    int free_space;
    int read_size;
    int total_read_size = 0;

    // A full ring has BS_RING_SIZE - 1 bytes
    free_space = BS_RING_SIZE - 1 - GetDataSizeInRB();
    if (free_space == 0) {
        return 0;
    }
    
    // First fill the ending part of the ring
    if (write_ptr_ >= read_ptr_) {
        int fill_space = BS_RING_SIZE - (write_ptr_ == 0 ? 1 : write_ptr_);
        read_size = p_stream_file_.read(reinterpret_cast<char*>(&bs_ring_[write_ptr_]), fill_space).gcount();
        if (read_size > 0) {
            write_ptr_ = (write_ptr_ + read_size) % BS_RING_SIZE; // when we still have more bytes to fill, write_ptr_ becomes 0 to continue to the next step.
        }
        if (read_size < fill_space) {
            end_of_file_ = true;
        }
        total_read_size += read_size;
        if (end_of_file_) {
            return total_read_size;
        }
        free_space -= read_size;
        if (free_space == 0) {
            return total_read_size;
        }
    }
        
    // Continue filling the beginning part of the ring
    if (read_ptr_ > 0) {
        read_size = p_stream_file_.read(reinterpret_cast<char*>(&bs_ring_[write_ptr_]), free_space).gcount();
        if (read_size > 0) {
            write_ptr_ = (write_ptr_ + read_size) % BS_RING_SIZE;
        }
        if (read_size < free_space) {
            end_of_file_ = true;
        }
        total_read_size += read_size;
    }
    return total_read_size;
}

bool RocVideoESParser::GetByte(int offset, uint8_t *data) {
    offset = offset % BS_RING_SIZE;
    if (offset == write_ptr_) {
        if (FetchBitStream() == 0) {
            end_of_stream_ = true;
            return false;
        }
    }
    *data = bs_ring_[offset];
    return true;
}

bool RocVideoESParser::ReadBytes(int offset, int size, uint8_t *data) {
    offset = offset % BS_RING_SIZE;
    if (size > GetDataSizeInRB()) {
        if (FetchBitStream() == 0) {
            end_of_stream_ = true;
            return false;
        }
        if (size > GetDataSizeInRB()) {
            logger_.CriticalLog(MakeMsg("Could not read the requested bytes from ring buffer. Either ring buffer size is too small or not enough bytes left."));
            return false;
        }
    }
    if (offset + size > BS_RING_SIZE) {
        int part = BS_RING_SIZE - offset;
        memcpy(data, &bs_ring_[offset], part);
        memcpy(&data[part], &bs_ring_[0], size - part);
    } else {
        memcpy(data, &bs_ring_[offset], size);
    }
    return true;
}

void RocVideoESParser::SetReadPointer(int value) {
    read_ptr_ = value % BS_RING_SIZE;
}

bool RocVideoESParser::FindStartCode() {
    uint8_t three_bytes[3];
    int i;

    curr_start_code_offset_ = next_start_code_offset_;

    // Search for the next start code
    while (!end_of_stream_) {
        for (i = 0; i < 3; i++) {
            if (GetByte(curr_byte_offset_ + i, three_bytes + i) == false) {
                break;
            }
        }
        if (i < 3) {
            break;
        }

        if (three_bytes[0] == 0 && three_bytes[1] == 0 && three_bytes[2] == 0x01) {
            num_start_code_++;
            next_start_code_offset_ = curr_byte_offset_;
            // Move the pointer 3 bytes forward
            curr_byte_offset_ = (curr_byte_offset_ + 3) % BS_RING_SIZE;

            // For the very first NAL unit, search for the next start code (or reach the end of frame)
            if (num_start_code_ == 1) {
                curr_start_code_offset_ = next_start_code_offset_;
                continue;
            } else {
                break;
            }
        }
        curr_byte_offset_ = (curr_byte_offset_ + 1) % BS_RING_SIZE;
    }
    return num_start_code_ ? true : false;
}

void RocVideoESParser::CopyNalUnitFromRing() {
    int nal_start, nal_end_plus_1;
    int nal_size;
    nal_start = curr_start_code_offset_;
    nal_end_plus_1 = curr_start_code_offset_ != next_start_code_offset_ ? next_start_code_offset_ : write_ptr_;
    if (nal_end_plus_1 >= nal_start) {
        nal_size = nal_end_plus_1 - nal_start;
        if ((pic_data_size_ + nal_size) > pic_data_.size()) {
            pic_data_.resize(pic_data_.size() + nal_size);
        }
        memcpy(&pic_data_[pic_data_size_], &bs_ring_[nal_start], nal_size);
    } else { // wrap around
        nal_size = BS_RING_SIZE - nal_start + nal_end_plus_1;
        if ((pic_data_size_ + nal_size) > pic_data_.size()) {
            pic_data_.resize(pic_data_.size() + nal_size);
        }
        memcpy(&pic_data_[pic_data_size_], &bs_ring_[nal_start], BS_RING_SIZE - nal_start);
        memcpy(&pic_data_[pic_data_size_ + BS_RING_SIZE - nal_start], &bs_ring_[0], nal_end_plus_1);
    }
    pic_data_size_ += nal_size;
    SetReadPointer(nal_end_plus_1);
}

void RocVideoESParser::CheckHevcNalForSlice(int start_code_offset, int *slice_flag, int *first_slice_flag) {
    uint8_t nal_header_byte;
    GetByte(start_code_offset + 3, &nal_header_byte);
    uint8_t nal_unit_type = (nal_header_byte >> 1) & 0x3F;
    switch (nal_unit_type) {
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
            *slice_flag = 1;
            uint8_t slice_byte;
            GetByte(start_code_offset + 5, &slice_byte);
            *first_slice_flag = slice_byte >> 7; // first_slice_segment_in_pic_flag
            break;
        }

        default:
            *slice_flag = 0;
            *first_slice_flag = 0;
            break;
    }
}

void RocVideoESParser::CheckAvcNalForSlice(int start_code_offset, int *slice_flag, int *first_slice_flag) {
    uint8_t nal_header_byte;
    GetByte(start_code_offset + 3, &nal_header_byte);
    uint8_t nal_unit_type = nal_header_byte & 0x1F;
    switch (nal_unit_type) {
        case kAvcNalTypeSlice_IDR:
        case kAvcNalTypeSlice_Non_IDR:
        case kAvcNalTypeSlice_Data_Partition_A:
        case kAvcNalTypeSlice_Data_Partition_B:
        case kAvcNalTypeSlice_Data_Partition_C: {
            *slice_flag = 1;
            uint8_t slice_bytes[4]; // 4 bytes is enough to parse the Exp-Golomb codes for first_mb_in_slice
            for (int i = 0; i < 4; i++) {
                GetByte(start_code_offset + 4 + i, &slice_bytes[i]);
            }
            size_t offset = 0;
            int first_mb_in_slice = Parser::ExpGolomb::ReadUe(slice_bytes, offset);
            *first_slice_flag = first_mb_in_slice == 0;
            break;
        }

        default:
            *slice_flag = 0;
            *first_slice_flag = 0;
            break;
    }
}

int RocVideoESParser::GetPicDataAvcHevc(uint8_t **p_pic_data, int *pic_size) {
    int slice_nal_flag;
    int first_slice_flag = 0;
    int num_slices = 0;
    
    curr_pic_end_ = 0;
    // Check if we have already got some NAL units for the current picture from processing of the last picture
    if (next_pic_start_ > 0 && next_pic_start_ < pic_data_size_) {
        memcpy(&pic_data_[0], &pic_data_[next_pic_start_], pic_data_size_ - next_pic_start_);
        pic_data_size_ = pic_data_size_ - next_pic_start_;
        curr_pic_end_ = pic_data_size_;
        next_pic_start_ = 0;
    } else {
        pic_data_size_ = 0;
        next_pic_start_ = 0;
    }

    while (!end_of_stream_) {
        if (!FindStartCode()) {
            logger_.ErrorLog(MakeMsg("No start code in the bitstream."));
            break;
        }
        CopyNalUnitFromRing();
        if ( stream_type_ == kStreamTypeAvcElementary) {
            CheckAvcNalForSlice(curr_start_code_offset_, &slice_nal_flag, &first_slice_flag);
        } else {
            CheckHevcNalForSlice(curr_start_code_offset_, &slice_nal_flag, &first_slice_flag);
        }
        if (slice_nal_flag) {
            num_slices++;
            curr_pic_end_ = pic_data_size_; // update the current picture data end
        }

        if (curr_start_code_offset_ == next_start_code_offset_) {
            break; // end of stream
        } else if (num_slices) {
            if ( stream_type_ == kStreamTypeAvcElementary) {
                CheckAvcNalForSlice(next_start_code_offset_, &slice_nal_flag, &first_slice_flag); // peek the next NAL
            } else {
                CheckHevcNalForSlice(next_start_code_offset_, &slice_nal_flag, &first_slice_flag); // peek the next NAL
            }
            if (slice_nal_flag && first_slice_flag) {
                // Between two pictures, we can have non-slice NAL units which are associated with the next picutre
                if (curr_pic_end_ < pic_data_size_) {
                    next_pic_start_ = curr_pic_end_;
                }
                break; // hit the first slice of the next picture
            }
        }
    }

    *p_pic_data = pic_data_.data();
    if (num_slices) {
        num_pictures_++;
        *pic_size = curr_pic_end_;
    } else {
        *pic_size = 0;
    }
    return 0;
}

bool RocVideoESParser::ReadObuHeaderAndSize(int *obu_type) {
    uint8_t header_byte;
    int obu_extension_flag;

    obu_size_ = 0;
    obu_byte_offset_ = curr_byte_offset_;
    // Parser header
    if (GetByte(curr_byte_offset_, &header_byte) == false) {
        return false;
    }
    *obu_type = (header_byte >> 3) & 0x0F;
    obu_extension_flag = (header_byte >> 2) & 0x01;
    curr_byte_offset_ = (curr_byte_offset_ + 1) % BS_RING_SIZE;
    obu_size_++;
    if (obu_extension_flag) {
        curr_byte_offset_ = (curr_byte_offset_ + 1) % BS_RING_SIZE;
        obu_size_++;
    }
    // Parse size
    int len;
    uint32_t value = 0;
    uint8_t data_byte;
    for (len = 0; len < 8; ++len) {
        if (GetByte(curr_byte_offset_ + len, &data_byte) == false) {
            return false;
        }
        value |= (data_byte & 0x7F) << (len * 7);
        if ((data_byte & 0x80) == 0) {
            ++len;
            break;
        }
    }
    obu_size_ += len + value;
    curr_byte_offset_ = (curr_byte_offset_ + len + value) % BS_RING_SIZE;

    return true;
}

bool RocVideoESParser::CopyObuFromRing() {
    if (obu_size_ > GetDataSizeInRB()) {
        if (FetchBitStream() == 0) {
            end_of_stream_ = true;
            return false;
        }
        if (obu_size_ > GetDataSizeInRB()) {
            return false;
        }
    }
    if ((pic_data_size_ + obu_size_) > pic_data_.size()) {
        pic_data_.resize(pic_data_.size() + obu_size_);
    }
    int obu_end_offset = (obu_byte_offset_ + obu_size_) % BS_RING_SIZE;
    if (obu_end_offset >= obu_byte_offset_) {
        memcpy(&pic_data_[pic_data_size_], &bs_ring_[obu_byte_offset_], obu_size_);
    } else {
        memcpy(&pic_data_[pic_data_size_], &bs_ring_[obu_byte_offset_], BS_RING_SIZE - obu_byte_offset_);
        memcpy(&pic_data_[pic_data_size_ + BS_RING_SIZE - obu_byte_offset_], &bs_ring_[0], obu_end_offset);
    }
    pic_data_size_ += obu_size_;
    SetReadPointer(obu_end_offset);
    return true;
}

int RocVideoESParser::GetPicDataAv1(uint8_t **p_pic_data, int *pic_size) {
    int obu_type;
    pic_data_size_ = 0;

    while (!end_of_stream_) {
        if (!ReadObuHeaderAndSize(&obu_type)) {
            break;
        }
        CopyObuFromRing();
        if (obu_type == kObuTemporalDelimiter) {
            num_td_obus_++;
        if (num_td_obus_ > 1) {
            break;
        }
        }
    }

    *p_pic_data = pic_data_.data();
    *pic_size = pic_data_size_;
    num_temp_units_++;
    return 0;
}

int RocVideoESParser::GetPicDataIvf(uint8_t **p_pic_data, int *pic_size) {
    uint8_t frame_header[12];
    pic_data_size_ = 0;
    if (ReadBytes(curr_byte_offset_, 12, frame_header)) {
        curr_byte_offset_ = (curr_byte_offset_ + 12) % BS_RING_SIZE;
        SetReadPointer(curr_byte_offset_);
        int frame_size = frame_header[0] | (frame_header[1] << 8) | (frame_header[2] << 16) | (frame_header[3] << 24);
        if (frame_size > pic_data_.size()) {
            pic_data_.resize(frame_size);
        }
        if (ReadBytes(curr_byte_offset_, frame_size, pic_data_.data())) {
            pic_data_size_ = frame_size;
            curr_byte_offset_ = (curr_byte_offset_ + frame_size) % BS_RING_SIZE;
            SetReadPointer(curr_byte_offset_);
        }
    }
    *p_pic_data = pic_data_.data();
    *pic_size = pic_data_size_;
    return 0;
}

int RocVideoESParser::GetPicData(uint8_t **p_pic_data, int *pic_size, int64_t *pts) {
    *pts = 0;
    switch (stream_type_) {
        case kStreamTypeAvcElementary:
        case kStreamTypeHevcElementary:
            return GetPicDataAvcHevc(p_pic_data, pic_size);
        case kStreamTypeAv1Elementary:
            return GetPicDataAv1(p_pic_data, pic_size);
        case kStreamTypeAv1Ivf:
        case kStreamTypeVp9Ivf: {
            if (!ivf_file_header_read_) {
                uint8_t file_header[32];
                ReadBytes(curr_byte_offset_, 32, file_header);
                curr_byte_offset_ = (curr_byte_offset_ + 32) % BS_RING_SIZE;
                SetReadPointer(curr_byte_offset_);
                ivf_file_header_read_ = true;
            }
            return GetPicDataIvf(p_pic_data, pic_size);
        }
        default: {
            *p_pic_data = pic_data_.data();
            *pic_size = 0;
            return 0;
        }
    }
}

rocDecVideoCodec RocVideoESParser::GetCodecId() {
    switch (stream_type_) {
        case kStreamTypeAvcElementary:
            return rocDecVideoCodec_AVC;
        case kStreamTypeHevcElementary:
            return rocDecVideoCodec_HEVC;
        case kStreamTypeAv1Elementary:
        case kStreamTypeAv1Ivf:
            return rocDecVideoCodec_AV1;
        case kStreamTypeVp9Ivf:
            return rocDecVideoCodec_VP9;
        default:
            return rocDecVideoCodec_NumCodecs;
    }
}

int RocVideoESParser::ProbeStreamType() {
    int stream_type = kStreamTypeUnsupported;
    int stream_type_score = 0;
    uint8_t *stream_buf;
    int stream_size;

    stream_buf = static_cast<uint8_t*>(malloc(STREAM_PROBE_SIZE));
    p_stream_file_.seekg (0, p_stream_file_.beg);
    stream_size = p_stream_file_.read(reinterpret_cast<char*>(stream_buf), STREAM_PROBE_SIZE).gcount();
    // When the file size is smaller than STREAM_PROBE_SIZE, the fail bit is set. If we don't clear the state, further operations will fail.
    if (p_stream_file_.fail()) {
        p_stream_file_.clear();
    }

    for (int i = kStreamTypeAvcElementary; i < kStreamTypeNumSupported; i++) {
        int curr_score = 0;
        switch (i) {
            case kStreamTypeAvcElementary: {
                curr_score = CheckAvcEStream(stream_buf, stream_size);
                if (curr_score > STREAM_TYPE_SCORE_THRESHOLD && curr_score > stream_type_score) {
                    stream_type = kStreamTypeAvcElementary;
                    stream_type_score = curr_score;
                }
                break;
            }
            case kStreamTypeHevcElementary: {
                curr_score = CheckHevcEStream(stream_buf, stream_size);
                if (curr_score > STREAM_TYPE_SCORE_THRESHOLD && curr_score > stream_type_score) {
                    stream_type = kStreamTypeHevcElementary;
                    stream_type_score = curr_score;
                }
                break;
            }
            case kStreamTypeAv1Elementary: {
                curr_score = CheckAv1EStream(stream_buf, stream_size);
                if (curr_score > STREAM_TYPE_SCORE_THRESHOLD && curr_score > stream_type_score) {
                    stream_type = kStreamTypeAv1Elementary;
                    stream_type_score = curr_score;
                }
                break;
            }
            case kStreamTypeAv1Ivf: {
                curr_score = CheckIvfAv1Stream(stream_buf, stream_size);
                if (curr_score > STREAM_TYPE_SCORE_THRESHOLD && curr_score > stream_type_score) {
                    stream_type = kStreamTypeAv1Ivf;
                    stream_type_score = curr_score;
                }
                break;
            }
            case kStreamTypeVp9Ivf: {
                curr_score = CheckIvfVp9Stream(stream_buf, stream_size);
                if (curr_score > STREAM_TYPE_SCORE_THRESHOLD && curr_score > stream_type_score) {
                    stream_type = kStreamTypeVp9Ivf;
                    stream_type_score = curr_score;
                }
                break;
            }
        }
        if (stream_type_score >= STREAM_TYPE_HIGH_CONFIDENCE_SCORE_THRESHOLD) {
            break;
        }
    }

    if (stream_buf) {
        free(stream_buf);
    }
    p_stream_file_.seekg(0, std::ios::beg);
    return stream_type;
}

int RocVideoESParser::CheckAvcEStream(uint8_t *p_stream, int stream_size) {
    int score = 0;
    int curr_offset = 0;
    int num_start_codes = 0;
    int sps_present = 0;
    int pps_present = 0;
    int slice_present = 0;
    int idr_slice_present = 0;
    int first_slice_present = 0;
    size_t offset = 0;

    while (curr_offset < stream_size - 2) {
        if (p_stream[curr_offset] == 0 && p_stream[curr_offset + 1] == 0 && p_stream[curr_offset + 2] == 1) {
            num_start_codes++;
            uint8_t nal_header_byte = p_stream[curr_offset + 3];
            uint8_t nal_unit_type = nal_header_byte & 0x1F;
            uint8_t nal_rbsp[256];
            memcpy(nal_rbsp, p_stream + curr_offset + 4, 256);
            EbspToRbsp(nal_rbsp, 0, 256);
            switch (nal_unit_type) {
                case kAvcNalTypeSeq_Parameter_Set: {
                    offset = 0;
                    uint32_t profile_idc = Parser::ReadBits(nal_rbsp, offset, 8);
                    Parser::ReadBits(nal_rbsp, offset, 8);
                    uint32_t level_idc = Parser::ReadBits(nal_rbsp, offset, 8);
                    uint32_t seq_parameter_set_id = Parser::ExpGolomb::ReadUe(nal_rbsp, offset);
                    uint32_t chroma_format_idc;
                    if (profile_idc == 100 ||
                        profile_idc == 110 ||
                        profile_idc == 122 ||
                        profile_idc == 244 ||
                        profile_idc == 44  ||
                        profile_idc == 83  ||
                        profile_idc == 86  ||
                        profile_idc == 118 ||
                        profile_idc == 128 ||
                        profile_idc == 138 ||
                        profile_idc == 139 ||
                        profile_idc == 134 ||
                        profile_idc == 135) {
                        chroma_format_idc = Parser::ExpGolomb::ReadUe(p_stream, offset);
                        if (chroma_format_idc == 3) {
                            Parser::GetBit(p_stream, offset); // separate_colour_plane_flag
                        }
                        uint32_t bit_depth_luma = Parser::ExpGolomb::ReadUe(p_stream, offset) + 8;
                        uint32_t bit_depth_chroma = Parser::ExpGolomb::ReadUe(p_stream, offset) + 8;
                        bit_depth_ = bit_depth_luma > bit_depth_chroma ? bit_depth_luma : bit_depth_chroma;
                    } else {
                        chroma_format_idc = 1;
                        bit_depth_ = 8;
                    }

                    if (profile_idc > 0 && level_idc > 0 && seq_parameter_set_id >= 0 && seq_parameter_set_id <= 31 && chroma_format_idc >= 0 && chroma_format_idc <= 3 && bit_depth_ >= 8 && bit_depth_ <= 14) {
                        sps_present = 1;
                    }
                    break;
                }

                case kAvcNalTypePic_Parameter_Set: {
                    offset = 0;
                    uint32_t pic_parameter_set_id = Parser::ExpGolomb::ReadUe(nal_rbsp, offset);
                    uint32_t seq_parameter_set_id = Parser::ExpGolomb::ReadUe(nal_rbsp, offset);
                    if ( pic_parameter_set_id >= 0 && pic_parameter_set_id <= 255 && seq_parameter_set_id >= 0 && seq_parameter_set_id <= 31) {
                        pps_present = 1;
                    }
                    break;
                }

                case kAvcNalTypeSlice_IDR:
                    idr_slice_present = 1;
                case kAvcNalTypeSlice_Non_IDR:
                case kAvcNalTypeSlice_Data_Partition_A:
                case kAvcNalTypeSlice_Data_Partition_B:
                case kAvcNalTypeSlice_Data_Partition_C: {
                    slice_present = 1;
                    offset = 0;
                    uint32_t first_mb_in_slice = Parser::ExpGolomb::ReadUe(nal_rbsp, offset);
                    if ( first_mb_in_slice == 0) {
                        first_slice_present = 1;
                    }
                    break;
                }

                default:
                    break;
            }
            curr_offset += 4;
        } else {
            curr_offset++;
        }
    }
    if (num_start_codes == 0) {
        score = 0;
    } else {
        score = sps_present * 25 + pps_present * 25 + idr_slice_present * 20 + slice_present * 15 + first_slice_present * 15;
    }
    return score;
}

int RocVideoESParser::CheckHevcEStream(uint8_t *p_stream, int stream_size) {
    int score = 0;
    int curr_offset = 0;
    int num_start_codes = 0;
    int vps_present = 0;
    int sps_present = 0;
    int pps_present = 0;
    int slice_present = 0;
    int rap_slice_present = 0;
    int first_slice_present = 0;
    size_t offset = 0;

    while (curr_offset < stream_size - 2) {
        if (p_stream[curr_offset] == 0 && p_stream[curr_offset + 1] == 0 && p_stream[curr_offset + 2] == 1) {
            num_start_codes++;
            uint8_t nal_header_byte = p_stream[curr_offset + 3];
            uint8_t nal_unit_type = (nal_header_byte >> 1) & 0x3F;
            uint8_t nal_rbsp[256];
            memcpy(nal_rbsp, p_stream + curr_offset + 5, 256);
            EbspToRbsp(nal_rbsp, 0, 256);
            switch (nal_unit_type) {
                 case NAL_UNIT_VPS: {
                    offset = 16;
                    int vps_reserved_0xffff_16bits = Parser::ReadBits(nal_rbsp, offset, 16);
                    if (vps_reserved_0xffff_16bits == 0xFFFF) {
                        vps_present = 1;
                    }
                    break;
                }

                case NAL_UNIT_SPS: {
                    offset = 0;
                    Parser::ReadBits(nal_rbsp, offset, 4); // sps_video_parameter_set_id
                    uint32_t max_sub_layer_minus1 = Parser::ReadBits(nal_rbsp, offset, 3);
                    Parser::GetBit(nal_rbsp, offset); // sps_temporal_id_nesting_flag
                    // profile_tier_level()
                    int sub_layer_profile_present_flag[6];
                    int sub_layer_level_present_flag[6];
                    offset += 96;
                    for (int i = 0; i < max_sub_layer_minus1; i++) {
                        sub_layer_profile_present_flag[i] = Parser::GetBit(nal_rbsp, offset);
                        sub_layer_level_present_flag[i] = Parser::GetBit(nal_rbsp, offset);
                    }
                    if (max_sub_layer_minus1 > 0) {
                        for (int i = max_sub_layer_minus1; i < 8; i++) {
                            offset += 2;
                        }
                    }
                    for (int i = 0; i < max_sub_layer_minus1; i++) {
                        if (sub_layer_profile_present_flag[i]) {
                            offset += 88;
                        }
                        if (sub_layer_level_present_flag[i]) {
                            offset += 8;
                        }
                    }
                    uint32_t sps_seq_parameter_set_id = Parser::ExpGolomb::ReadUe(nal_rbsp, offset);
                    uint32_t chroma_format_idc = Parser::ExpGolomb::ReadUe(nal_rbsp, offset);
                    if (chroma_format_idc == 3) {
                        Parser::GetBit(nal_rbsp, offset); // separate_colour_plane_flag
                    }
                    Parser::ExpGolomb::ReadUe(nal_rbsp, offset); // pic_width_in_luma_samples
                    Parser::ExpGolomb::ReadUe(nal_rbsp, offset); // pic_height_in_luma_samples
                    int conformance_window_flag = Parser::GetBit(nal_rbsp, offset);
                    if (conformance_window_flag) {
                        Parser::ExpGolomb::ReadUe(nal_rbsp, offset); // conf_win_left_offset
                        Parser::ExpGolomb::ReadUe(nal_rbsp, offset); // conf_win_right_offset
                        Parser::ExpGolomb::ReadUe(nal_rbsp, offset); // conf_win_top_offset
                        Parser::ExpGolomb::ReadUe(nal_rbsp, offset); // conf_win_bottom_offset
                    }
                    uint32_t bit_depth_luma = Parser::ExpGolomb::ReadUe(nal_rbsp, offset) + 8;
                    uint32_t bit_depth_chroma = Parser::ExpGolomb::ReadUe(nal_rbsp, offset) + 8;
                    bit_depth_ = bit_depth_luma > bit_depth_chroma ? bit_depth_luma : bit_depth_chroma;
                    if (sps_seq_parameter_set_id >= 0 && sps_seq_parameter_set_id <= 15 && chroma_format_idc >= 0 && chroma_format_idc <= 3 && bit_depth_ >= 8 && bit_depth_ <= 16) {
                        sps_present = 1;
                    }

                    break;
                }

                case NAL_UNIT_PPS: {
                    offset = 0;
                    uint32_t pps_pic_parameter_set_id = Parser::ExpGolomb::ReadUe(nal_rbsp, offset);
                    uint32_t pps_seq_parameter_set_id = Parser::ExpGolomb::ReadUe(nal_rbsp, offset);
                    if ( pps_pic_parameter_set_id >= 0 && pps_pic_parameter_set_id <= 63 && pps_seq_parameter_set_id >= 0 && pps_seq_parameter_set_id <= 15) {
                        pps_present = 1;
                    }
                    break;
                }

                case NAL_UNIT_CODED_SLICE_BLA_W_LP:
                case NAL_UNIT_CODED_SLICE_BLA_W_RADL:
                case NAL_UNIT_CODED_SLICE_BLA_N_LP:
                case NAL_UNIT_CODED_SLICE_IDR_W_RADL:
                case NAL_UNIT_CODED_SLICE_IDR_N_LP:
                case NAL_UNIT_CODED_SLICE_CRA_NUT:
                    rap_slice_present = 1;
                case NAL_UNIT_CODED_SLICE_TRAIL_R:
                case NAL_UNIT_CODED_SLICE_TRAIL_N:
                case NAL_UNIT_CODED_SLICE_TLA_R:
                case NAL_UNIT_CODED_SLICE_TSA_N:
                case NAL_UNIT_CODED_SLICE_STSA_R:
                case NAL_UNIT_CODED_SLICE_STSA_N:
                case NAL_UNIT_CODED_SLICE_RADL_N:
                case NAL_UNIT_CODED_SLICE_RADL_R:
                case NAL_UNIT_CODED_SLICE_RASL_N:
                case NAL_UNIT_CODED_SLICE_RASL_R: {
                    offset = 0;
                    int first_slice_segment_in_pic_flag = Parser::GetBit(nal_rbsp, offset);
                    if (first_slice_segment_in_pic_flag) {
                        first_slice_present = 1;
                    }
                    if (nal_unit_type >= NAL_UNIT_CODED_SLICE_BLA_W_LP && nal_unit_type <= NAL_UNIT_RESERVED_IRAP_VCL23) {
                        offset++;
                    }
                    uint32_t slice_pic_parameter_set_id = Parser::ExpGolomb::ReadUe(nal_rbsp, offset);
                    if ( slice_pic_parameter_set_id >= 0 && slice_pic_parameter_set_id <= 63) {
                        slice_present = 1;
                    } else {
                        slice_present = 0;
                    }
                    if (!slice_present) {
                        rap_slice_present = 0;
                        first_slice_present = 0;
                    }
                    break;
                }

                default:
                    break;
            }
            curr_offset += 5;
        } else {
            curr_offset++;
        }
    }
    if (num_start_codes == 0) {
        score = 0;
    } else {
        score = vps_present * 15 + sps_present * 20 + pps_present * 20 + rap_slice_present * 15 + slice_present * 15 + first_slice_present * 15;
    }
    return score;
}

int RocVideoESParser::EbspToRbsp(uint8_t *streamBuffer, int begin_bytepos, int end_bytepos) {
    int count = 0;
    if (end_bytepos < begin_bytepos) {
        return end_bytepos;
    }
    uint8_t *streamBuffer_i = streamBuffer + begin_bytepos;
    uint8_t *streamBuffer_end = streamBuffer + end_bytepos;
    int reduce_count = 0;
    for (; streamBuffer_i != streamBuffer_end; ) {
        //starting from begin_bytepos to avoid header information
        //in NAL unit, 0x000000, 0x000001 or 0x000002 shall not occur at any uint8_t-aligned position
        uint8_t tmp =* streamBuffer_i;
        if (count == ZEROBYTES_SHORTSTARTCODE) {
            if (tmp == 0x03) {
                //check the 4th uint8_t after 0x000003, except when cabac_zero_word is used, in which case the last three bytes of this NAL unit must be 0x000003
                if ((streamBuffer_i + 1 != streamBuffer_end) && (streamBuffer_i[1] > 0x03)) {
                    return -1;
                }
                //if cabac_zero_word is used, the final uint8_t of this NAL unit(0x03) is discarded, and the last two bytes of RBSP must be 0x0000
                if (streamBuffer_i + 1 == streamBuffer_end) {
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
    return end_bytepos - begin_bytepos + reduce_count;
}

uint32_t RocVideoESParser::ReadUVLC(const uint8_t *p_stream, size_t &bit_offset) {
    int leading_zeros = 0;
    while (!Parser::GetBit(p_stream, bit_offset)) {
        ++leading_zeros;
    }
    // Maximum 32 bits.
    if (leading_zeros >= 32) {
        return 0xFFFFFFFF;
    }
    uint32_t base = (1u << leading_zeros) - 1;
    uint32_t value = Parser::ReadBits(p_stream, bit_offset, leading_zeros);
    return base + value;
}

int RocVideoESParser::CheckAv1EStream(uint8_t *p_stream, int stream_size) {
    int score = 0;
    uint8_t *obu_stream = p_stream;
    uint32_t curr_offset = 0;
    int temporal_delimiter_obu_present = 0;
    int seq_header_obu_present = 0;
    int frame_header_obu_present = 0;
    int frame_obu_present = 0;
    int tile_group_obu_present = 0;
    bool syntax_error = false;
    size_t offset = 0;

    while (curr_offset < stream_size) {
        // OBU header
        Av1ObuHeader obu_header;
        offset = 0;
        obu_stream = p_stream + curr_offset;
        obu_header.size = 1;
        if (Parser::GetBit(obu_stream, offset) != 0) {
            syntax_error = true;
            break;
        }
        obu_header.obu_type = Parser::ReadBits(obu_stream, offset, 4);
        obu_header.obu_extension_flag = Parser::GetBit(obu_stream, offset);
        obu_header.obu_has_size_field = Parser::GetBit(obu_stream, offset);
        if (!obu_header.obu_has_size_field) {
            syntax_error = true;
            break;
        }
        if (Parser::GetBit(obu_stream, offset) != 0) {
            syntax_error = true;
            break;
        }
        if (obu_header.obu_extension_flag) {
            obu_header.size += 1;
            obu_header.temporal_id = Parser::ReadBits(obu_stream, offset, 3);
            obu_header.spatial_id = Parser::ReadBits(obu_stream, offset, 2);
            if (Parser::ReadBits(obu_stream, offset, 3) != 0) {
                syntax_error = true;
                break;
            }
        }
        curr_offset += obu_header.size;
        obu_stream += obu_header.size;
        // OBU size
        int len;
        uint32_t obu_size = 0;
        for (len = 0; len < 8; ++len) {
            obu_size |= (obu_stream[len] & 0x7F) << (len * 7);
            if ((obu_stream[len] & 0x80) == 0) {
                ++len;
                break;
            }
        }
        curr_offset += len;
        obu_stream += len;

        switch (obu_header.obu_type) {
            case kObuTemporalDelimiter:
                temporal_delimiter_obu_present = 1;
                break;

            case kObuSequenceHeader: {
                Av1SequenceHeader seq_header = {0};
                offset = 0;
                seq_header.seq_profile = Parser::ReadBits(obu_stream, offset, 3);
                seq_header.still_picture = Parser::GetBit(obu_stream, offset);
                seq_header.reduced_still_picture_header = Parser::GetBit(obu_stream, offset);

                if (seq_header.reduced_still_picture_header) {
                    seq_header.timing_info_present_flag = 0;
                    seq_header.decoder_model_info_present_flag = 0;
                    seq_header.initial_display_delay_present_flag = 0;
                    seq_header.operating_points_cnt_minus_1 = 0;
                    seq_header.operating_point_idc[0] = 0;
                    seq_header.seq_level_idx[0] = Parser::ReadBits(obu_stream, offset, 5);
                    seq_header.seq_tier[0] = 0;
                    seq_header.decoder_model_present_for_this_op[0] = 0;
                    seq_header.initial_display_delay_present_for_this_op[0] = 0;
                } else {
                    seq_header.timing_info_present_flag = Parser::GetBit(obu_stream, offset);
                    if (seq_header.timing_info_present_flag) {
                        // timing_info()
                        seq_header.timing_info.num_units_in_display_tick = Parser::ReadBits(obu_stream, offset, 32);
                        seq_header.timing_info.time_scale = Parser::ReadBits(obu_stream, offset, 32);
                        seq_header.timing_info.equal_picture_interval = Parser::GetBit(obu_stream, offset);
                        if (seq_header.timing_info.equal_picture_interval) {
                            seq_header.timing_info.num_ticks_per_picture_minus_1 = ReadUVLC(obu_stream, offset);
                        }
                        seq_header.decoder_model_info_present_flag = Parser::GetBit(obu_stream, offset);
                        if (seq_header.decoder_model_info_present_flag) {
                            seq_header.decoder_model_info.buffer_delay_length_minus_1 = Parser::ReadBits(obu_stream, offset, 5);
                            seq_header.decoder_model_info.num_units_in_decoding_tick = Parser::ReadBits(obu_stream, offset, 32);
                            seq_header.decoder_model_info.buffer_removal_time_length_minus_1 = Parser::ReadBits(obu_stream, offset, 5);
                            seq_header.decoder_model_info.frame_presentation_time_length_minus_1 = Parser::ReadBits(obu_stream, offset, 5);
                        }
                    } else {
                        seq_header.decoder_model_info_present_flag = 0;
                    }
                    seq_header.initial_display_delay_present_flag = Parser::GetBit(obu_stream, offset);
                    seq_header.operating_points_cnt_minus_1 = Parser::ReadBits(obu_stream, offset, 5);
                    for (int i = 0; i < seq_header.operating_points_cnt_minus_1 + 1; i++) {
                        seq_header.operating_point_idc[i] = Parser::ReadBits(obu_stream, offset, 12);
                        seq_header.seq_level_idx[i] = Parser::ReadBits(obu_stream, offset, 5);
                        if (seq_header.seq_level_idx[i] > 7) {
                            seq_header.seq_tier[i] = Parser::GetBit(obu_stream, offset);
                        } else {
                            seq_header.seq_tier[i] = 0;
                        }
                        if (seq_header.decoder_model_info_present_flag) {
                            seq_header.decoder_model_present_for_this_op[i] = Parser::GetBit(obu_stream, offset);
                            if (seq_header.decoder_model_present_for_this_op[i]) {
                                seq_header.operating_parameters_info[i].decoder_buffer_delay = Parser::ReadBits(obu_stream, offset, seq_header.decoder_model_info.buffer_delay_length_minus_1 + 1);
                                seq_header.operating_parameters_info[i].encoder_buffer_delay = Parser::ReadBits(obu_stream, offset, seq_header.decoder_model_info.buffer_delay_length_minus_1 + 1);
                                seq_header.operating_parameters_info[i].low_delay_mode_flag = Parser::GetBit(obu_stream, offset);
                            }
                        } else {
                            seq_header.decoder_model_present_for_this_op[i] = 0;
                        }

                        if (seq_header.initial_display_delay_present_flag) {
                            seq_header.initial_display_delay_present_for_this_op[i] = Parser::GetBit(obu_stream, offset);
                            if (seq_header.initial_display_delay_present_for_this_op[i]) {
                                seq_header.initial_display_delay_minus_1[i] = Parser::ReadBits(obu_stream, offset, 4);
                            }
                        }
                    }
                }
                seq_header.frame_width_bits_minus_1 = Parser::ReadBits(obu_stream, offset, 4);
                seq_header.frame_height_bits_minus_1 = Parser::ReadBits(obu_stream, offset, 4);
                seq_header.max_frame_width_minus_1 = Parser::ReadBits(obu_stream, offset, seq_header.frame_width_bits_minus_1 + 1);
                seq_header.max_frame_height_minus_1 = Parser::ReadBits(obu_stream, offset, seq_header.frame_height_bits_minus_1 + 1);
                if (seq_header.reduced_still_picture_header) {
                    seq_header.frame_id_numbers_present_flag = 0;
                } else {
                    seq_header.frame_id_numbers_present_flag = Parser::GetBit(obu_stream, offset);
                }
                if (seq_header.frame_id_numbers_present_flag) {
                    seq_header.delta_frame_id_length_minus_2 = Parser::ReadBits(obu_stream, offset, 4);
                    seq_header.additional_frame_id_length_minus_1 = Parser::ReadBits(obu_stream, offset, 3);
                }
                seq_header.use_128x128_superblock = Parser::GetBit(obu_stream, offset);
                seq_header.enable_filter_intra = Parser::GetBit(obu_stream, offset);
                seq_header.enable_intra_edge_filter = Parser::GetBit(obu_stream, offset);

                if (seq_header.reduced_still_picture_header) {
                    seq_header.enable_interintra_compound = 0;
                    seq_header.enable_masked_compound = 0;
                    seq_header.enable_warped_motion = 0;
                    seq_header.enable_dual_filter = 0;
                    seq_header.enable_order_hint = 0;
                    seq_header.enable_jnt_comp = 0;
                    seq_header.enable_ref_frame_mvs = 0;
                    seq_header.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
                    seq_header.seq_force_integer_mv = SELECT_INTEGER_MV;
                    seq_header.order_hint_bits = 0;
                } else {
                    seq_header.enable_interintra_compound = Parser::GetBit(obu_stream, offset);
                    seq_header.enable_masked_compound = Parser::GetBit(obu_stream, offset);
                    seq_header.enable_warped_motion = Parser::GetBit(obu_stream, offset);
                    seq_header.enable_dual_filter = Parser::GetBit(obu_stream, offset);
                    seq_header.enable_order_hint = Parser::GetBit(obu_stream, offset);
                    if (seq_header.enable_order_hint) {
                        seq_header.enable_jnt_comp = Parser::GetBit(obu_stream, offset);
                        seq_header.enable_ref_frame_mvs = Parser::GetBit(obu_stream, offset);
                    } else {
                        seq_header.enable_jnt_comp = 0;
                        seq_header.enable_ref_frame_mvs = 0;
                    }
                    seq_header.seq_choose_screen_content_tools = Parser::GetBit(obu_stream, offset);
                    if (seq_header.seq_choose_screen_content_tools) {
                        seq_header.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
                    } else {
                        seq_header.seq_force_screen_content_tools = Parser::GetBit(obu_stream, offset);
                    }
                    if (seq_header.seq_force_screen_content_tools > 0) {
                        seq_header.seq_choose_integer_mv = Parser::GetBit(obu_stream, offset);
                        if (seq_header.seq_choose_integer_mv) {
                            seq_header.seq_force_integer_mv = SELECT_INTEGER_MV;
                        } else {
                            seq_header.seq_force_integer_mv = Parser::GetBit(obu_stream, offset);
                        }
                    } else {
                        seq_header.seq_force_integer_mv = SELECT_INTEGER_MV;
                    }

                    if (seq_header.enable_order_hint) {
                        seq_header.order_hint_bits_minus_1 = Parser::ReadBits(obu_stream, offset, 3);
                        seq_header.order_hint_bits = seq_header.order_hint_bits_minus_1 + 1;
                    } else {
                        seq_header.order_hint_bits = 0;
                    }
                }
                seq_header.enable_superres = Parser::GetBit(obu_stream, offset);
                seq_header.enable_cdef = Parser::GetBit(obu_stream, offset);
                seq_header.enable_restoration = Parser::GetBit(obu_stream, offset);
                seq_header.color_config.bit_depth = 8;
                seq_header.color_config.high_bitdepth = Parser::GetBit(obu_stream, offset);
                if (seq_header.seq_profile == 2 && seq_header.color_config.high_bitdepth) {
                    seq_header.color_config.twelve_bit = Parser::GetBit(obu_stream, offset);
                    seq_header.color_config.bit_depth = seq_header.color_config.twelve_bit ? 12 : 10;
                } else if (seq_header.seq_profile <= 2) {
                    seq_header.color_config.bit_depth = seq_header.color_config.high_bitdepth ? 10 : 8;
                }
                bit_depth_ = seq_header.color_config.bit_depth;
                if (seq_header.seq_profile >= 0 && seq_header.seq_profile <= 2) {
                    seq_header_obu_present = 1;
                }
                break;
            }

            case kObuFrameHeader:
                frame_header_obu_present = 1;
                break;

            case kObuFrame:
                frame_obu_present = 1;
                break;

            case kObuTileGroup:
                tile_group_obu_present = 1;
                break;
        }

        // Check before update
        if (obu_size > (stream_size - curr_offset)) {
            break;
        }
        curr_offset += obu_size;
    }
    if (syntax_error) {
        score = 0;
    } else {
        score = temporal_delimiter_obu_present * 25 + seq_header_obu_present * 25 + frame_obu_present * 50 + (frame_header_obu_present & tile_group_obu_present) * 50;
    }
    return score;
}

int RocVideoESParser::CheckIvfAv1Stream(uint8_t *p_stream, int stream_size) {
    static const char *IVF_SIGNATURE = "DKIF";
    static const char *AV1_FourCC = "AV01";
    static const int IvfFileHeaderSize = 32;
    static const int IvfFrameHeaderSize = 12;
    uint8_t *ptr = p_stream;
    int score = 0;

    // bytes 0-3: signature
    if (memcmp(IVF_SIGNATURE, ptr, 4) == 0) {
        ptr += 4;
        // bytes 4-5: version (should be 0). Little Endian.
        int ivf_version = ptr[0] | (ptr[1] << 8);
        if (ivf_version != 0) {
            score = 0;
        } else {
            ptr += 4;
            // bytes 8-11: codec FourCC (e.g., 'AV01')
            if (memcmp(AV1_FourCC, ptr, 4)) {
                score = 0;
            } else {
                ptr = p_stream + IvfFileHeaderSize;
                int frame_size = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
                ptr += IvfFrameHeaderSize;
                int size = stream_size - IvfFileHeaderSize - IvfFrameHeaderSize;
                size = frame_size < size ? frame_size : size;
                score = CheckAv1EStream(ptr, size);
            }
        }
    } else {
        score = 0;
    }
    return score;
}

int RocVideoESParser::CheckVp9EStream(uint8_t *p_stream, int stream_size) {
    int score = 0;
    size_t offset = 0; // bit offset
    Vp9UncompressedHeader uncomp_header;

    uncomp_header.frame_marker = Parser::ReadBits(p_stream, offset, 2);
    if (uncomp_header.frame_marker != 2) {
        return 0;
    }
    uncomp_header.profile_low_bit = Parser::GetBit(p_stream, offset);
    uncomp_header.profile_high_bit = Parser::GetBit(p_stream, offset);
    uncomp_header.profile = (uncomp_header.profile_high_bit << 1) + uncomp_header.profile_low_bit;
    if (uncomp_header.profile == 3) {
        uncomp_header.reserved_zero = Parser::GetBit(p_stream, offset);
        if (uncomp_header.reserved_zero) {
            return 0;
        }
    }
    uncomp_header.show_existing_frame = Parser::GetBit(p_stream, offset);
    if (uncomp_header.show_existing_frame) {
        return 0;
    }
    uncomp_header.frame_type = Parser::GetBit(p_stream, offset);
    uncomp_header.show_frame = Parser::GetBit(p_stream, offset);
    uncomp_header.error_resilient_mode = Parser::GetBit(p_stream, offset);
    if (uncomp_header.frame_type == kVp9KeyFrame) {
        uncomp_header.frame_sync_code.frame_sync_byte_0 = Parser::ReadBits(p_stream, offset, 8);
        if (uncomp_header.frame_sync_code.frame_sync_byte_0 != 0x49) {
            return 0;
        }
        uncomp_header.frame_sync_code.frame_sync_byte_1 = Parser::ReadBits(p_stream, offset, 8);
        if (uncomp_header.frame_sync_code.frame_sync_byte_1 != 0x83) {
            return 0;
        }
        uncomp_header.frame_sync_code.frame_sync_byte_2 = Parser::ReadBits(p_stream, offset, 8);
        if (uncomp_header.frame_sync_code.frame_sync_byte_2 != 0x42) {
            return 0;
        }
        score = 100;
    } else {
        return 0;
    }
    return score;
}

int RocVideoESParser::CheckIvfVp9Stream(uint8_t *p_stream, int stream_size) {
    static const char *IVF_SIGNATURE = "DKIF";
    static const char *VP9_FourCC = "VP90";
    static const int IvfFileHeaderSize = 32;
    static const int IvfFrameHeaderSize = 12;
    uint8_t *ptr = p_stream;
    int score = 0;

    // bytes 0-3: signature
    if (memcmp(IVF_SIGNATURE, ptr, 4) == 0) {
        ptr += 4;
        // bytes 4-5: version (should be 0). Little Endian.
        int ivf_version = ptr[0] | (ptr[1] << 8);
        if (ivf_version != 0) {
            score = 0;
        } else {
            ptr += 4;
            // bytes 8-11: codec FourCC (e.g., 'VP90')
            if (memcmp(VP9_FourCC, ptr, 4)) {
                score = 0;
            } else {
                ptr = p_stream + IvfFileHeaderSize;
                int frame_size = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
                ptr += IvfFrameHeaderSize;
                int size = stream_size - IvfFileHeaderSize - IvfFrameHeaderSize;
                size = frame_size < size ? frame_size : size;
                score = CheckVp9EStream(ptr, size);
            }
        }
    } else {
        score = 0;
    }
    return score;
}