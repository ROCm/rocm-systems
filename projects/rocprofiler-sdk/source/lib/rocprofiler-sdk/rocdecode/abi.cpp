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

#include "lib/rocprofiler-sdk/rocdecode/rocdecode.hpp"

#include "lib/common/abi.hpp"
#include "lib/common/defines.hpp"

#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/rocdecode.h>

namespace rocprofiler
{
namespace rocdecode
{
static_assert(ROCDECODE_RUNTIME_API_TABLE_MAJOR_VERSION == 0,
              "Major version updated for ROCDecode dispatch table");

#if ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 0
ROCP_SDK_ENFORCE_ABI_VERSIONING(::RocDecodeDispatchTable, 11);
#endif

#if ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 1
ROCP_SDK_ENFORCE_ABI_VERSIONING(::RocDecodeDispatchTable, 16);
#endif

ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_create_video_parser, 0)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_parse_video_data, 1)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_destroy_video_parser, 2)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_create_decoder, 3)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_destroy_decoder, 4)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_get_gecoder_caps, 5)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_decode_frame, 6)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_get_decode_status, 7)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_reconfigure_decoder, 8)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_get_video_frame, 9)
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_get_error_name, 10)

#if ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION >= 1
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_create_bitstream_reader, 11);
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_get_bitstream_codec_type, 12);
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_get_bitstream_bit_depth, 13);
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_get_bitstream_pic_data, 14);
ROCP_SDK_ENFORCE_ABI(::RocDecodeDispatchTable, pfn_rocdec_destroy_bitstream_reader, 15);
#endif

}  // namespace rocdecode
}  // namespace rocprofiler
