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

#include "lib/rocprofiler-sdk/rccl/rccl.hpp"

#include "lib/common/abi.hpp"
#include "lib/common/defines.hpp"

#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/rccl.h>

namespace rocprofiler
{
namespace rccl
{
static_assert(RCCL_API_TRACE_VERSION_MAJOR == 0, "Major version updated for RCCL dispatch table");
static_assert(RCCL_API_TRACE_VERSION_PATCH == 0, "Patch version updated for RCCL dispatch table");

ROCP_SDK_ENFORCE_ABI_VERSIONING(::rcclApiFuncTable, 37)

ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclAllGather_fn, 0)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclAllReduce_fn, 1)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclAllToAll_fn, 2)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclAllToAllv_fn, 3)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclBroadcast_fn, 4)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclGather_fn, 5)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclReduce_fn, 6)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclReduceScatter_fn, 7)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclScatter_fn, 8)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclSend_fn, 9)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclRecv_fn, 10)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclRedOpCreatePreMulSum_fn, 11)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclRedOpDestroy_fn, 12)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclGroupStart_fn, 13)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclGroupEnd_fn, 14)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclGetVersion_fn, 15)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclGetUniqueId_fn, 16)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommInitRank_fn, 17)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommInitAll_fn, 18)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommInitRankConfig_fn, 19)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommFinalize_fn, 20)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommDestroy_fn, 21)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommAbort_fn, 22)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommSplit_fn, 23)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclGetErrorString_fn, 24)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclGetLastError_fn, 25)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommGetAsyncError_fn, 26)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommCount_fn, 27)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommCuDevice_fn, 28)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommUserRank_fn, 29)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclMemAlloc_fn, 30)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclMemFree_fn, 31)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, mscclLoadAlgo_fn, 32)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, mscclRunAlgo_fn, 33)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, mscclUnloadAlgo_fn, 34)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommRegister_fn, 35)
ROCP_SDK_ENFORCE_ABI(::rcclApiFuncTable, ncclCommDeregister_fn, 36)
}  // namespace rccl
}  // namespace rocprofiler
