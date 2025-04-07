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

#include "lib/rocprofiler-sdk/rccl/defines.hpp"
#include "lib/rocprofiler-sdk/rccl/rccl.hpp"

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rccl.h>
#include <rocprofiler-sdk/rccl/table_id.h>

namespace rocprofiler
{
namespace rccl
{
template <>
struct rccl_domain_info<ROCPROFILER_RCCL_TABLE_ID_LAST>
{
    using args_type          = rocprofiler_rccl_api_args_t;
    using retval_type        = rocprofiler_rccl_api_retval_t;
    using callback_data_type = rocprofiler_callback_tracing_rccl_api_data_t;
    using buffer_data_type   = rocprofiler_buffer_tracing_rccl_api_record_t;
};

template <>
struct rccl_domain_info<ROCPROFILER_RCCL_TABLE_ID>
: rccl_domain_info<ROCPROFILER_RCCL_TABLE_ID_LAST>
{
    using enum_type                           = rocprofiler_marker_core_api_id_t;
    static constexpr auto callback_domain_idx = ROCPROFILER_CALLBACK_TRACING_RCCL_API;
    static constexpr auto buffered_domain_idx = ROCPROFILER_BUFFER_TRACING_RCCL_API;
    static constexpr auto none                = ROCPROFILER_RCCL_API_ID_NONE;
    static constexpr auto last                = ROCPROFILER_RCCL_API_ID_LAST;
    static constexpr auto external_correlation_id_domain_idx =
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_RCCL_API;
};

}  // namespace rccl
}  // namespace rocprofiler

#if defined(ROCPROFILER_LIB_ROCPROFILER_SDK_RCCL_RCCL_CPP_IMPL) &&                                 \
    ROCPROFILER_LIB_ROCPROFILER_SDK_RCCL_RCCL_CPP_IMPL == 1

// clang-format off
RCCL_API_TABLE_LOOKUP_DEFINITION(ROCPROFILER_RCCL_TABLE_ID, rccl_api_func_table_t)

RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclAllGather, ncclAllGather, ncclAllGather_fn, sendbuff, recvbuff, sendcount, datatype, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclAllReduce, ncclAllReduce, ncclAllReduce_fn, sendbuff, recvbuff, count, datatype, op, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclAllToAll, ncclAllToAll, ncclAllToAll_fn, sendbuff, recvbuff, count, datatype, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclAllToAllv, ncclAllToAllv, ncclAllToAllv_fn, sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclBroadcast, ncclBroadcast, ncclBroadcast_fn, sendbuff, recvbuff, count, datatype, root, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclGather, ncclGather, ncclGather_fn, sendbuff, recvbuff, sendcount, datatype, root, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclReduce, ncclReduce, ncclReduce_fn, sendbuff, recvbuff, count, datatype, op, root, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclReduceScatter, ncclReduceScatter, ncclReduceScatter_fn, sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclScatter, ncclScatter, ncclScatter_fn, sendbuff, recvbuff, recvcount, datatype, root, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclSend, ncclSend, ncclSend_fn, sendbuff, count, datatype, peer, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclRecv, ncclRecv, ncclRecv_fn, recvbuff, count, datatype, peer, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclRedOpCreatePreMulSum, ncclRedOpCreatePreMulSum, ncclRedOpCreatePreMulSum_fn, op, scalar, datatype, residence, comm)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclRedOpDestroy, ncclRedOpDestroy, ncclRedOpDestroy_fn, op, comm)

RCCL_API_INFO_DEFINITION_0(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclGroupStart, ncclGroupStart, ncclGroupStart_fn)
RCCL_API_INFO_DEFINITION_0(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclGroupEnd, ncclGroupEnd, ncclGroupEnd_fn)

RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclGetVersion, ncclGetVersion, ncclGetVersion_fn, version)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclGetUniqueId, ncclGetUniqueId, ncclGetUniqueId_fn, out)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommInitRank, ncclCommInitRank, ncclCommInitRank_fn, newcomm, nranks, commId, myrank)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommInitAll, ncclCommInitAll, ncclCommInitAll_fn, comms, ndev, devlist)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommInitRankConfig, ncclCommInitRankConfig, ncclCommInitRankConfig_fn, comm, nranks, commId, myrank, config)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommFinalize, ncclCommFinalize, ncclCommFinalize_fn, comm)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommDestroy, ncclCommDestroy, ncclCommDestroy_fn, comm)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommAbort, ncclCommAbort, ncclCommAbort_fn, comm)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommSplit, ncclCommSplit, ncclCommSplit_fn, comm, color, key, newcomm, config)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclGetErrorString, ncclGetErrorString, ncclGetErrorString_fn, code)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclGetLastError, ncclGetLastError, ncclGetLastError_fn, comm)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommGetAsyncError, ncclCommGetAsyncError, ncclCommGetAsyncError_fn, comm, asyncError)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommCount, ncclCommCount, ncclCommCount_fn, comm, count)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommCuDevice, ncclCommCuDevice, ncclCommCuDevice_fn, comm, devid)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommUserRank, ncclCommUserRank, ncclCommUserRank_fn, comm, rank)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclMemAlloc, ncclMemAlloc, ncclMemAlloc_fn, ptr, size)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclMemFree, ncclMemFree, ncclMemFree_fn, ptr)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_mscclLoadAlgo, mscclLoadAlgo, mscclLoadAlgo_fn, mscclAlgoFilePath, mscclAlgoHandle, rank)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_mscclRunAlgo, mscclRunAlgo, mscclRunAlgo_fn, sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls, count, dataType, root, peer, op, mscclAlgoHandle, comm, stream)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_mscclUnloadAlgo, mscclUnloadAlgo, mscclUnloadAlgo_fn, mscclAlgoHandle)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommRegister, ncclCommRegister, ncclCommRegister_fn, comm, buff, size, handle)
RCCL_API_INFO_DEFINITION_V(ROCPROFILER_RCCL_TABLE_ID, ROCPROFILER_RCCL_API_ID_ncclCommDeregister, ncclCommDeregister, ncclCommDeregister_fn, comm, handle)

#else
#    error                                                                                         \
        "Do not compile this file directly. It is included by lib/rocprofiler-sdk/rccl/rccl.cpp"
#endif
