// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/hipfile/hipfile.hpp"

#include "lib/common/abi.hpp"
#include "lib/common/defines.hpp"

#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/hipfile.h>

namespace rocprofiler
{
namespace hipfile
{
static_assert(HIPFILE_RUNTIME_API_TABLE_MAJOR_VERSION == 0,
              "Major version updated for hipFILE dispatch table");

#if HIPFILE_RUNTIME_API_TABLE_STEP_VERSION == 0
ROCP_SDK_ENFORCE_ABI_VERSIONING(::hipFileDispatchTable, 31)
#elif HIPFILE_RUNTIME_API_TABLE_STEP_VERSION == 1
ROCP_SDK_ENFORCE_ABI_VERSIONING(::hipFileDispatchTable, 34)
#else
INTERNAL_CI_ROCP_SDK_ENFORCE_ABI_VERSIONING(::hipFileDispatchTable, 0)
#endif

ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_get_op_error_string, 0)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_handle_register, 1)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_handle_deregister, 2)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_buf_register, 3)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_buf_deregister, 4)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_read, 5)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_write, 6)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_driver_open, 7)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_driver_close, 8)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_use_count, 9)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_driver_get_properties, 10)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_driver_set_poll_mode, 11)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_driver_set_max_direct_io_size, 12)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_driver_set_max_cache_size, 13)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_driver_set_max_pinned_mem_size, 14)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_batch_io_set_up, 15)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_batch_io_submit, 16)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_batch_io_get_status, 17)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_batch_io_cancel, 18)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_batch_io_destroy, 19)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_read_async, 20)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_write_async, 21)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_stream_register, 22)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_stream_deregister, 23)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_get_version, 24)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_get_parameter_size_t, 25)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_get_parameter_bool, 26)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_get_parameter_string, 27)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_set_parameter_size_t, 28)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_set_parameter_bool, 29)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_set_parameter_string, 30)

#if HIPFILE_RUNTIME_API_TABLE_STEP_VERSION >= 1
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_get_stats_l1, 31)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_get_stats_l2, 32)
ROCP_SDK_ENFORCE_ABI(::hipFileDispatchTable, pfn_hipfile_get_stats_l3, 33)
#endif

}  // namespace hipfile
}  // namespace rocprofiler
