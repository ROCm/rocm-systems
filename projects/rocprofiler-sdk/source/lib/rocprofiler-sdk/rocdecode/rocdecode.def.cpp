// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/rocdecode/defines.hpp"
#include "lib/rocprofiler-sdk/rocdecode/rocdecode.hpp"
#include "lib/rocprofiler-sdk/rocdecode/utils.hpp"

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocdecode.h>
#include <rocprofiler-sdk/rocdecode/table_id.h>

namespace rocprofiler
{
namespace rocdecode
{
template <>
struct rocdecode_domain_info<ROCPROFILER_ROCDECODE_TABLE_ID_LAST>
{
    using args_type              = rocprofiler_rocdecode_api_args_t;
    using retval_type            = rocprofiler_rocdecode_api_retval_t;
    using callback_data_type     = rocprofiler_callback_tracing_rocdecode_api_data_t;
    using buffer_data_type       = rocprofiler_buffer_tracing_rocdecode_api_record_t;
    using buffered_ext_data_type = rocprofiler_buffer_tracing_rocdecode_api_ext_record_t;
};

template <>
struct rocdecode_domain_info<ROCPROFILER_ROCDECODE_TABLE_ID_CORE>
: rocdecode_domain_info<ROCPROFILER_ROCDECODE_TABLE_ID_LAST>
{
    using enum_type                               = rocprofiler_marker_core_api_id_t;
    static constexpr auto callback_domain_idx     = ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API;
    static constexpr auto buffered_domain_idx     = ROCPROFILER_BUFFER_TRACING_ROCDECODE_API;
    static constexpr auto buffered_ext_domain_idx = ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT;
    static constexpr auto none                    = ROCPROFILER_ROCDECODE_API_ID_NONE;
    static constexpr auto last                    = ROCPROFILER_ROCDECODE_API_ID_LAST;
    static constexpr auto external_correlation_id_domain_idx =
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCDECODE_API;
};

}  // namespace rocdecode
}  // namespace rocprofiler

#if defined(ROCPROFILER_LIB_ROCPROFILER_SDK_ROCDECODE_ROCDECODE_CPP_IMPL) &&                       \
    ROCPROFILER_LIB_ROCPROFILER_SDK_ROCDECODE_ROCDECODE_CPP_IMPL == 1

// clang-format off
ROCDECODE_API_TABLE_LOOKUP_DEFINITION(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, rocdecode_api_func_table_t)

ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecCreateVideoParser, rocDecCreateVideoParser, pfn_rocdec_create_video_parser, parser_handle, params)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecParseVideoData, rocDecParseVideoData, pfn_rocdec_parse_video_data, parser_handle, packet)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecDestroyVideoParser, rocDecDestroyVideoParser, pfn_rocdec_destroy_video_parser, parser_handle)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecCreateDecoder, rocDecCreateDecoder, pfn_rocdec_create_decoder, decoder_handle, decoder_create_info)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecDestroyDecoder, rocDecDestroyDecoder, pfn_rocdec_destroy_decoder, decoder_handle)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecGetDecoderCaps, rocDecGetDecoderCaps, pfn_rocdec_get_gecoder_caps, decode_caps)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecDecodeFrame, rocDecDecodeFrame, pfn_rocdec_decode_frame, decoder_handle, pic_params)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecGetDecodeStatus, rocDecGetDecodeStatus, pfn_rocdec_get_decode_status, decoder_handle, pic_idx, decode_status)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecReconfigureDecoder, rocDecReconfigureDecoder, pfn_rocdec_reconfigure_decoder, decoder_handle, reconfig_params)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecGetVideoFrame, rocDecGetVideoFrame, pfn_rocdec_get_video_frame, decoder_handle, pic_idx, dev_mem_ptr, horizontal_pitch, vid_postproc_params)
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecGetErrorName, rocDecGetErrorName, pfn_rocdec_get_error_name, rocdec_status)

#if  ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION >= 1
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecCreateBitstreamReader, rocDecCreateBitstreamReader, pfn_rocdec_create_bitstream_reader, bs_reader_handle, input_file_path);
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecGetBitstreamCodecType, rocDecGetBitstreamCodecType, pfn_rocdec_get_bitstream_codec_type, bs_reader_handle, codec_type);
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecGetBitstreamBitDepth, rocDecGetBitstreamBitDepth, pfn_rocdec_get_bitstream_bit_depth, bs_reader_handle, bit_depth);
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecGetBitstreamPicData, rocDecGetBitstreamPicData, pfn_rocdec_get_bitstream_pic_data, bs_reader_handle, pic_data, pic_size, pts);
ROCDECODE_API_INFO_DEFINITION_V(ROCPROFILER_ROCDECODE_TABLE_ID_CORE, ROCPROFILER_ROCDECODE_API_ID_rocDecDestroyBitstreamReader, rocDecDestroyBitstreamReader, pfn_rocdec_destroy_bitstream_reader, bs_reader_handle);
#endif
#else
#    error                                                                                         \
        "Do not compile this file directly. It is included by lib/rocprofiler-sdk/rocdecode/rocdecode.cpp"
#endif
