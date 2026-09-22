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

#include "lib/rocprofiler-sdk/rocjpeg/defines.hpp"
#include "lib/rocprofiler-sdk/rocjpeg/rocjpeg.hpp"

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocjpeg.h>
#include <rocprofiler-sdk/rocjpeg/table_id.h>

namespace rocprofiler
{
namespace rocjpeg
{
template <>
struct rocjpeg_domain_info<ROCPROFILER_ROCJPEG_TABLE_ID_LAST>
{
    using args_type          = rocprofiler_rocjpeg_api_args_t;
    using retval_type        = rocprofiler_rocjpeg_api_retval_t;
    using callback_data_type = rocprofiler_callback_tracing_rocjpeg_api_data_t;
    using buffer_data_type   = rocprofiler_buffer_tracing_rocjpeg_api_record_t;
};

template <>
struct rocjpeg_domain_info<ROCPROFILER_ROCJPEG_TABLE_ID_CORE>
: rocjpeg_domain_info<ROCPROFILER_ROCJPEG_TABLE_ID_LAST>
{
    using enum_type                           = rocprofiler_marker_core_api_id_t;
    static constexpr auto callback_domain_idx = ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API;
    static constexpr auto buffered_domain_idx = ROCPROFILER_BUFFER_TRACING_ROCJPEG_API;
    static constexpr auto none                = ROCPROFILER_ROCJPEG_API_ID_NONE;
    static constexpr auto last                = ROCPROFILER_ROCJPEG_API_ID_LAST;
    static constexpr auto external_correlation_id_domain_idx =
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCJPEG_API;
};

}  // namespace rocjpeg
}  // namespace rocprofiler

#if defined(ROCPROFILER_LIB_ROCPROFILER_SDK_ROCJPEG_ROCJPEG_CPP_IMPL) &&                           \
    ROCPROFILER_LIB_ROCPROFILER_SDK_ROCJPEG_ROCJPEG_CPP_IMPL == 1

// clang-format off
ROCJPEG_API_TABLE_LOOKUP_DEFINITION(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, rocjpeg_api_func_table_t)

ROCJPEG_API_INFO_DEFINITION_V(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, ROCPROFILER_ROCJPEG_API_ID_rocJpegStreamCreate, rocJpegStreamCreate, pfn_rocjpeg_stream_create, jpeg_stream_handle)
ROCJPEG_API_INFO_DEFINITION_V(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, ROCPROFILER_ROCJPEG_API_ID_rocJpegStreamParse, rocJpegStreamParse, pfn_rocjpeg_stream_parse, data, length, jpeg_stream_handle)
ROCJPEG_API_INFO_DEFINITION_V(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, ROCPROFILER_ROCJPEG_API_ID_rocJpegStreamDestroy, rocJpegStreamDestroy, pfn_rocjpeg_stream_destroy, jpeg_stream_handle)
ROCJPEG_API_INFO_DEFINITION_V(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, ROCPROFILER_ROCJPEG_API_ID_rocJpegCreate, rocJpegCreate, pfn_rocjpeg_create, backend, device_id, handle)
ROCJPEG_API_INFO_DEFINITION_V(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, ROCPROFILER_ROCJPEG_API_ID_rocJpegDestroy, rocJpegDestroy, pfn_rocjpeg_destroy, handle)
ROCJPEG_API_INFO_DEFINITION_V(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, ROCPROFILER_ROCJPEG_API_ID_rocJpegGetImageInfo, rocJpegGetImageInfo, pfn_rocjpeg_get_image_info, handle, jpeg_stream_handle, num_components, subsampling, widths, heights)
ROCJPEG_API_INFO_DEFINITION_V(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, ROCPROFILER_ROCJPEG_API_ID_rocJpegDecode, rocJpegDecode, pfn_rocjpeg_decode, handle, jpeg_stream_handle, decode_params, destination)
ROCJPEG_API_INFO_DEFINITION_V(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, ROCPROFILER_ROCJPEG_API_ID_rocJpegDecodeBatched, rocJpegDecodeBatched, pfn_rocjpeg_decode_batched, handle, jpeg_stream_handles, batch_size, decode_params, destinations)
ROCJPEG_API_INFO_DEFINITION_V(ROCPROFILER_ROCJPEG_TABLE_ID_CORE, ROCPROFILER_ROCJPEG_API_ID_rocJpegGetErrorName, rocJpegGetErrorName, pfn_rocjpeg_get_error_name, rocjpeg_status)

#else
#    error                                                                                         \
        "Do not compile this file directly. It is included by lib/rocprofiler-sdk/rocjpeg/rocjpeg.cpp"
#endif


