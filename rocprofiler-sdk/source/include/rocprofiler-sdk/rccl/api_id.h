

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

#pragma once

/**
 * @brief ROCProfiler enumeration of RCCL API tracing operations
 */
typedef enum rocprofiler_rccl_api_id_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_RCCL_API_ID_NONE = -1,

    ROCPROFILER_RCCL_API_ID_ncclAllGather = 0,
    ROCPROFILER_RCCL_API_ID_ncclAllReduce,
    ROCPROFILER_RCCL_API_ID_ncclAllToAll,
    ROCPROFILER_RCCL_API_ID_ncclAllToAllv,
    ROCPROFILER_RCCL_API_ID_ncclBroadcast,
    ROCPROFILER_RCCL_API_ID_ncclGather,
    ROCPROFILER_RCCL_API_ID_ncclReduce,
    ROCPROFILER_RCCL_API_ID_ncclReduceScatter,
    ROCPROFILER_RCCL_API_ID_ncclScatter,
    ROCPROFILER_RCCL_API_ID_ncclSend,
    ROCPROFILER_RCCL_API_ID_ncclRecv,
    ROCPROFILER_RCCL_API_ID_ncclRedOpCreatePreMulSum,
    ROCPROFILER_RCCL_API_ID_ncclRedOpDestroy,
    ROCPROFILER_RCCL_API_ID_ncclGroupStart,
    ROCPROFILER_RCCL_API_ID_ncclGroupEnd,
    ROCPROFILER_RCCL_API_ID_ncclGetVersion,
    ROCPROFILER_RCCL_API_ID_ncclGetUniqueId,
    ROCPROFILER_RCCL_API_ID_ncclCommInitRank,
    ROCPROFILER_RCCL_API_ID_ncclCommInitAll,
    ROCPROFILER_RCCL_API_ID_ncclCommInitRankConfig,
    ROCPROFILER_RCCL_API_ID_ncclCommFinalize,
    ROCPROFILER_RCCL_API_ID_ncclCommDestroy,
    ROCPROFILER_RCCL_API_ID_ncclCommAbort,
    ROCPROFILER_RCCL_API_ID_ncclCommSplit,
    ROCPROFILER_RCCL_API_ID_ncclGetErrorString,
    ROCPROFILER_RCCL_API_ID_ncclGetLastError,
    ROCPROFILER_RCCL_API_ID_ncclCommGetAsyncError,
    ROCPROFILER_RCCL_API_ID_ncclCommCount,
    ROCPROFILER_RCCL_API_ID_ncclCommCuDevice,
    ROCPROFILER_RCCL_API_ID_ncclCommUserRank,
    ROCPROFILER_RCCL_API_ID_ncclMemAlloc,
    ROCPROFILER_RCCL_API_ID_ncclMemFree,
    ROCPROFILER_RCCL_API_ID_mscclLoadAlgo,
    ROCPROFILER_RCCL_API_ID_mscclRunAlgo,
    ROCPROFILER_RCCL_API_ID_mscclUnloadAlgo,
    ROCPROFILER_RCCL_API_ID_ncclCommRegister,
    ROCPROFILER_RCCL_API_ID_ncclCommDeregister,

    ROCPROFILER_RCCL_API_ID_LAST,
} rocprofiler_rccl_api_id_t;
