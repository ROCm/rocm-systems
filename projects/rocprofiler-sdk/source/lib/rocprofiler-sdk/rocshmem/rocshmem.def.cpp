// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/rocshmem/defines.hpp"
#include "lib/rocprofiler-sdk/rocshmem/rocshmem.hpp"

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocshmem.h>
#include <rocprofiler-sdk/rocshmem/table_id.h>

namespace rocprofiler
{
namespace rocshmem
{
template <>
struct rocshmem_domain_info<ROCPROFILER_ROCSHMEM_TABLE_ID_LAST>
{
    using args_type              = rocprofiler_rocshmem_api_args_t;
    using retval_type            = rocprofiler_rocshmem_api_retval_t;
    using callback_data_type     = rocprofiler_callback_tracing_rocshmem_api_data_t;
    using buffer_data_type       = rocprofiler_buffer_tracing_rocshmem_api_record_t;
    using buffered_ext_data_type = rocprofiler_buffer_tracing_rocshmem_api_ext_record_t;
};

template <>
struct rocshmem_domain_info<ROCPROFILER_ROCSHMEM_TABLE_ID_CORE>
: rocshmem_domain_info<ROCPROFILER_ROCSHMEM_TABLE_ID_LAST>
{
    using enum_type                               = rocprofiler_rocshmem_api_id_t;
    static constexpr auto callback_domain_idx     = ROCPROFILER_CALLBACK_TRACING_ROCSHMEM_API;
    static constexpr auto buffered_domain_idx     = ROCPROFILER_BUFFER_TRACING_ROCSHMEM_API;
    static constexpr auto buffered_ext_domain_idx = ROCPROFILER_BUFFER_TRACING_ROCSHMEM_API_EXT;
    static constexpr auto none                    = ROCPROFILER_ROCSHMEM_API_ID_NONE;
    static constexpr auto last                    = ROCPROFILER_ROCSHMEM_API_ID_LAST;
    static constexpr auto external_correlation_id_domain_idx =
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCSHMEM_API;
};

}  // namespace rocshmem
}  // namespace rocprofiler

#if defined(ROCPROFILER_LIB_ROCPROFILER_SDK_ROCSHMEM_ROCSHMEM_CPP_IMPL) &&                         \
    ROCPROFILER_LIB_ROCPROFILER_SDK_ROCSHMEM_ROCSHMEM_CPP_IMPL == 1

// clang-format off
ROCSHMEM_API_TABLE_LOOKUP_DEFINITION(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, rocshmem_api_func_table_t)

ROCSHMEM_API_INFO_DEFINITION_V(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, ROCPROFILER_ROCSHMEM_API_ID_barrier_all_on_stream, barrier_all_on_stream, barrier_all_on_stream_fn, stream)
ROCSHMEM_API_INFO_DEFINITION_V(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, ROCPROFILER_ROCSHMEM_API_ID_quiet_on_stream, quiet_on_stream, quiet_on_stream_fn, stream)
ROCSHMEM_API_INFO_DEFINITION_V(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, ROCPROFILER_ROCSHMEM_API_ID_sync_all_on_stream, sync_all_on_stream, sync_all_on_stream_fn, stream)
ROCSHMEM_API_INFO_DEFINITION_V(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, ROCPROFILER_ROCSHMEM_API_ID_alltoallmem_on_stream, alltoallmem_on_stream, alltoallmem_on_stream_fn, team, dest, source, size, stream)
ROCSHMEM_API_INFO_DEFINITION_V(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, ROCPROFILER_ROCSHMEM_API_ID_broadcastmem_on_stream, broadcastmem_on_stream, broadcastmem_on_stream_fn, team, dest, source, nelems, pe_root, stream)
ROCSHMEM_API_INFO_DEFINITION_V(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, ROCPROFILER_ROCSHMEM_API_ID_getmem_on_stream, getmem_on_stream, getmem_on_stream_fn, dest, source, nelems, pe, stream)
ROCSHMEM_API_INFO_DEFINITION_V(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, ROCPROFILER_ROCSHMEM_API_ID_putmem_on_stream, putmem_on_stream, putmem_on_stream_fn, dest, source, nelems, pe, stream)
ROCSHMEM_API_INFO_DEFINITION_V(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, ROCPROFILER_ROCSHMEM_API_ID_putmem_signal_on_stream, putmem_signal_on_stream, putmem_signal_on_stream_fn, dest, source, nelems, sig_addr, signal, sig_op, pe, stream)
ROCSHMEM_API_INFO_DEFINITION_V(ROCPROFILER_ROCSHMEM_TABLE_ID_CORE, ROCPROFILER_ROCSHMEM_API_ID_signal_wait_until_on_stream, signal_wait_until_on_stream, signal_wait_until_on_stream_fn, sig_addr, cmp, cmp_value, stream)
// clang-format on

#else
#    error                                                                                         \
        "Do not compile this file directly. It is included by lib/rocprofiler-sdk/rocshmem/rocshmem.cpp"
#endif
