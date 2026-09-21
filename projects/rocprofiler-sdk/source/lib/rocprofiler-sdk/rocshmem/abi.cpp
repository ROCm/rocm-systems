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

#include "lib/rocprofiler-sdk/rocshmem/rocshmem.hpp"

#include "lib/common/abi.hpp"
#include "lib/common/defines.hpp"

namespace rocprofiler
{
namespace rocshmem
{
static_assert(ROCSHMEM_API_TRACE_VERSION_MAJOR == 0,
              "Major version updated for rocSHMEM dispatch table");

ROCP_SDK_ENFORCE_ABI(::rocshmemApiFuncTable, barrier_all_on_stream_fn, 0)
ROCP_SDK_ENFORCE_ABI(::rocshmemApiFuncTable, quiet_on_stream_fn, 1)
ROCP_SDK_ENFORCE_ABI(::rocshmemApiFuncTable, sync_all_on_stream_fn, 2)
ROCP_SDK_ENFORCE_ABI(::rocshmemApiFuncTable, alltoallmem_on_stream_fn, 3)
ROCP_SDK_ENFORCE_ABI(::rocshmemApiFuncTable, broadcastmem_on_stream_fn, 4)
ROCP_SDK_ENFORCE_ABI(::rocshmemApiFuncTable, getmem_on_stream_fn, 5)
ROCP_SDK_ENFORCE_ABI(::rocshmemApiFuncTable, putmem_on_stream_fn, 6)
ROCP_SDK_ENFORCE_ABI(::rocshmemApiFuncTable, putmem_signal_on_stream_fn, 7)
ROCP_SDK_ENFORCE_ABI(::rocshmemApiFuncTable, signal_wait_until_on_stream_fn, 8)

// As rocSHMEM grows new dispatch table entries, ROCSHMEM_API_TRACE_VERSION_PATCH will be
// incremented and the matching arm here must be added (mirrors the RCCL pattern in
// rccl/abi.cpp). Until that history exists, we pin the expected size to the current
// patch level (7) and force a build break on any newer/unknown patch via
// INTERNAL_CI_ROCP_SDK_ENFORCE_ABI_VERSIONING (active under ROCPROFILER_CI=1).
#if ROCSHMEM_API_TRACE_VERSION_PATCH == 7
ROCP_SDK_ENFORCE_ABI_VERSIONING(::rocshmemApiFuncTable, 9)
#else
INTERNAL_CI_ROCP_SDK_ENFORCE_ABI_VERSIONING(::rocshmemApiFuncTable, 0)
#endif

}  // namespace rocshmem
}  // namespace rocprofiler
