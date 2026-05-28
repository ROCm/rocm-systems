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
#include "../../api/amd_detail/rocdecode_api_trace.h"

namespace rocdecode {
const RocDecodeDispatchTable* GetRocDecodeDispatchTable();
} //namespace rocdecode

rocDecStatus ROCDECAPI rocDecCreateVideoParser(RocdecVideoParser *parser_handle, RocdecParserParams *params) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_create_video_parser(parser_handle, params);
}
rocDecStatus ROCDECAPI rocDecParseVideoData(RocdecVideoParser parser_handle, RocdecSourceDataPacket *packet) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_parse_video_data(parser_handle, packet);
}
rocDecStatus ROCDECAPI rocDecDestroyVideoParser(RocdecVideoParser parser_handle) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_destroy_video_parser(parser_handle);
}
rocDecStatus ROCDECAPI rocDecCreateDecoder(rocDecDecoderHandle *decoder_handle, RocDecoderCreateInfo *decoder_create_info) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_create_decoder(decoder_handle, decoder_create_info);
}
rocDecStatus ROCDECAPI rocDecDestroyDecoder(rocDecDecoderHandle decoder_handle) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_destroy_decoder(decoder_handle);
}
rocDecStatus ROCDECAPI rocDecGetDecoderCaps(RocdecDecodeCaps *decode_caps) { 
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_get_gecoder_caps(decode_caps);
}
rocDecStatus ROCDECAPI rocDecDecodeFrame(rocDecDecoderHandle decoder_handle, RocdecPicParams *pic_params) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_decode_frame(decoder_handle, pic_params);
}
rocDecStatus ROCDECAPI rocDecGetDecodeStatus(rocDecDecoderHandle decoder_handle, int pic_idx, RocdecDecodeStatus *decode_status) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_get_decode_status(decoder_handle, pic_idx, decode_status);
}
rocDecStatus ROCDECAPI rocDecReconfigureDecoder(rocDecDecoderHandle decoder_handle, RocdecReconfigureDecoderInfo *reconfig_params) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_reconfigure_decoder(decoder_handle, reconfig_params);
}
rocDecStatus ROCDECAPI rocDecGetVideoFrame(rocDecDecoderHandle decoder_handle, int pic_idx, void *dev_mem_ptr[3], uint32_t *horizontal_pitch, RocdecProcParams *vid_postproc_params) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_get_video_frame(decoder_handle, pic_idx, dev_mem_ptr, horizontal_pitch, vid_postproc_params);
}
const char *ROCDECAPI rocDecGetErrorName(rocDecStatus rocdec_status) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_get_error_name(rocdec_status);
}
rocDecStatus ROCDECAPI rocDecCreateBitstreamReader(RocdecBitstreamReader *bs_reader_handle, const char *input_file_path) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_create_bitstream_reader(bs_reader_handle, input_file_path);
}
rocDecStatus ROCDECAPI rocDecGetBitstreamCodecType(RocdecBitstreamReader bs_reader_handle, rocDecVideoCodec *codec_type) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_get_bitstream_codec_type(bs_reader_handle, codec_type);
}
rocDecStatus ROCDECAPI rocDecGetBitstreamBitDepth(RocdecBitstreamReader bs_reader_handle, int *bit_depth) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_get_bitstream_bit_depth(bs_reader_handle, bit_depth);
}
rocDecStatus ROCDECAPI rocDecGetBitstreamPicData(RocdecBitstreamReader bs_reader_handle, uint8_t **pic_data, int *pic_size, int64_t *pts) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_get_bitstream_pic_data(bs_reader_handle, pic_data, pic_size, pts);
}
rocDecStatus ROCDECAPI rocDecDestroyBitstreamReader(RocdecBitstreamReader bs_reader_handle) {
    return rocdecode::GetRocDecodeDispatchTable()->pfn_rocdec_destroy_bitstream_reader(bs_reader_handle);
}

