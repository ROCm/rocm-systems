// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/rocjpeg/rocjpeg.hpp"

#include "lib/common/abi.hpp"
#include "lib/common/defines.hpp"

#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/rocjpeg.h>

namespace rocprofiler
{
namespace rocjpeg
{
static_assert(ROCJPEG_RUNTIME_API_TABLE_MAJOR_VERSION == 0,
              "Major version updated for rocJPEG dispatch table");

ROCP_SDK_ENFORCE_ABI_VERSIONING(::RocJpegDispatchTable, 9)

ROCP_SDK_ENFORCE_ABI(::RocJpegDispatchTable, pfn_rocjpeg_stream_create, 0)
ROCP_SDK_ENFORCE_ABI(::RocJpegDispatchTable, pfn_rocjpeg_stream_parse, 1)
ROCP_SDK_ENFORCE_ABI(::RocJpegDispatchTable, pfn_rocjpeg_stream_destroy, 2)
ROCP_SDK_ENFORCE_ABI(::RocJpegDispatchTable, pfn_rocjpeg_create, 3)
ROCP_SDK_ENFORCE_ABI(::RocJpegDispatchTable, pfn_rocjpeg_destroy, 4)
ROCP_SDK_ENFORCE_ABI(::RocJpegDispatchTable, pfn_rocjpeg_get_image_info, 5)
ROCP_SDK_ENFORCE_ABI(::RocJpegDispatchTable, pfn_rocjpeg_decode, 6)
ROCP_SDK_ENFORCE_ABI(::RocJpegDispatchTable, pfn_rocjpeg_decode_batched, 7)
ROCP_SDK_ENFORCE_ABI(::RocJpegDispatchTable, pfn_rocjpeg_get_error_name, 8)

}  // namespace rocjpeg
}  // namespace rocprofiler
