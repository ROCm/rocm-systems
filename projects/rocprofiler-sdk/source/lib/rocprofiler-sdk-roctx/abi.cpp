// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/abi.hpp"
#include "lib/common/defines.hpp"

#include <rocprofiler-sdk-roctx/api_trace.h>
#include <rocprofiler-sdk-roctx/defines.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <rocprofiler-sdk-roctx/types.h>
#include <rocprofiler-sdk-roctx/version.h>

namespace rocprofiler
{
namespace roctx
{
// Reminder to update step version if updated
ROCP_SDK_ENFORCE_ABI_VERSIONING(::roctxCoreApiTable_t, 6)
ROCP_SDK_ENFORCE_ABI_VERSIONING(::roctxControlApiTable_t, 2)
ROCP_SDK_ENFORCE_ABI_VERSIONING(::roctxNameApiTable_t, 4)

// These ensure that function pointers are not re-ordered
// core
ROCP_SDK_ENFORCE_ABI(::roctxCoreApiTable_t, roctxMarkA_fn, 0);
ROCP_SDK_ENFORCE_ABI(::roctxCoreApiTable_t, roctxRangePushA_fn, 1);
ROCP_SDK_ENFORCE_ABI(::roctxCoreApiTable_t, roctxRangePop_fn, 2);
ROCP_SDK_ENFORCE_ABI(::roctxCoreApiTable_t, roctxRangeStartA_fn, 3);
ROCP_SDK_ENFORCE_ABI(::roctxCoreApiTable_t, roctxRangeStop_fn, 4);
ROCP_SDK_ENFORCE_ABI(::roctxCoreApiTable_t, roctxGetThreadId_fn, 5);
// control
ROCP_SDK_ENFORCE_ABI(::roctxControlApiTable_t, roctxProfilerPause_fn, 0);
ROCP_SDK_ENFORCE_ABI(::roctxControlApiTable_t, roctxProfilerResume_fn, 1);
// name
ROCP_SDK_ENFORCE_ABI(::roctxNameApiTable_t, roctxNameOsThread_fn, 0);
ROCP_SDK_ENFORCE_ABI(::roctxNameApiTable_t, roctxNameHsaAgent_fn, 1);
ROCP_SDK_ENFORCE_ABI(::roctxNameApiTable_t, roctxNameHipDevice_fn, 2);
ROCP_SDK_ENFORCE_ABI(::roctxNameApiTable_t, roctxNameHipStream_fn, 3);
}  // namespace roctx
}  // namespace rocprofiler
