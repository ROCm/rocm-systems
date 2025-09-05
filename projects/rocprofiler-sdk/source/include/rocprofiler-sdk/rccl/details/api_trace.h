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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#if !defined(ROCPROFILER_SDK_USE_SYSTEM_RCCL)
#    if defined __has_include
#        if __has_include(<rccl/rccl.h>)
#            define ROCPROFILER_SDK_USE_SYSTEM_RCCL 1
#        else
#            define ROCPROFILER_SDK_USE_SYSTEM_RCCL 0
#        endif
#    else
#        define ROCPROFILER_SDK_USE_SYSTEM_RCCL 0
#    endif
#endif

#if ROCPROFILER_SDK_USE_SYSTEM_RCCL > 0
#    include <rccl/rccl.h>
#else
#    include <rocprofiler-sdk/rccl/details/rccl.h>
#endif

#include <stddef.h>
#include <stdint.h>

// should only be increased if fundamental changes to dispatch table(s)
#define RCCL_API_TRACE_VERSION_MAJOR 0

// should be increased every time new members are added to existing dispatch tables
#define RCCL_API_TRACE_VERSION_PATCH 0

#if !defined(RCCL_EXTERN_C_INIT)
#    ifdef __cplusplus
#        define RCCL_EXTERN_C_INIT extern "C" {
#    else
#        define RCCL_EXTERN_C_INIT
#    endif
#endif

#if !defined(RCCL_EXTERN_C_FINI)
#    ifdef __cplusplus
#        define RCCL_EXTERN_C_FINI }
#    else
#        define RCCL_EXTERN_C_FINI
#    endif
#endif

RCCL_EXTERN_C_INIT

typedef uint64_t rccl_range_id_t;
typedef ncclResult_t (*ncclAllGather_fn_t)(const void*    sendbuff,
                                           void*          recvbuff,
                                           size_t         sendcount,
                                           ncclDataType_t datatype,
                                           ncclComm_t     comm,
                                           hipStream_t    stream);
typedef ncclResult_t (*ncclAllReduce_fn_t)(const void*      sendbuff,
                                           void*            recvbuff,
                                           size_t           count,
                                           ncclDataType_t   datatype,
                                           ncclRedOp_t      op,
                                           struct ncclComm* comm,
                                           hipStream_t      stream);
typedef ncclResult_t (*ncclAllToAll_fn_t)(const void*    sendbuff,
                                          void*          recvbuff,
                                          size_t         count,
                                          ncclDataType_t datatype,
                                          ncclComm_t     comm,
                                          hipStream_t    stream);
typedef ncclResult_t (*ncclAllToAllv_fn_t)(const void*    sendbuff,
                                           const size_t   sendcounts[],
                                           const size_t   sdispls[],
                                           void*          recvbuff,
                                           const size_t   recvcounts[],
                                           const size_t   rdispls[],
                                           ncclDataType_t datatype,
                                           ncclComm_t     comm,
                                           hipStream_t    stream);
typedef ncclResult_t (*ncclBroadcast_fn_t)(const void*    sendbuff,
                                           void*          recvbuff,
                                           size_t         count,
                                           ncclDataType_t datatype,
                                           int            root,
                                           ncclComm_t     comm,
                                           hipStream_t    stream);
typedef ncclResult_t (*ncclGather_fn_t)(const void*    sendbuff,
                                        void*          recvbuff,
                                        size_t         sendcount,
                                        ncclDataType_t datatype,
                                        int            root,
                                        ncclComm_t     comm,
                                        hipStream_t    stream);
typedef ncclResult_t (*ncclReduce_fn_t)(const void*    sendbuff,
                                        void*          recvbuff,
                                        size_t         count,
                                        ncclDataType_t datatype,
                                        ncclRedOp_t    op,
                                        int            root,
                                        ncclComm_t     comm,
                                        hipStream_t    stream);
typedef ncclResult_t (*ncclReduceScatter_fn_t)(const void*      sendbuff,
                                               void*            recvbuff,
                                               size_t           recvcount,
                                               ncclDataType_t   datatype,
                                               ncclRedOp_t      op,
                                               struct ncclComm* comm,
                                               hipStream_t      stream);
typedef ncclResult_t (*ncclScatter_fn_t)(const void*    sendbuff,
                                         void*          recvbuff,
                                         size_t         recvcount,
                                         ncclDataType_t datatype,
                                         int            root,
                                         ncclComm_t     comm,
                                         hipStream_t    stream);
typedef ncclResult_t (*ncclSend_fn_t)(const void*    sendbuff,
                                      size_t         count,
                                      ncclDataType_t datatype,
                                      int            peer,
                                      ncclComm_t     comm,
                                      hipStream_t    stream);
typedef ncclResult_t (*ncclRecv_fn_t)(void*          recvbuff,
                                      size_t         count,
                                      ncclDataType_t datatype,
                                      int            peer,
                                      ncclComm_t     comm,
                                      hipStream_t    stream);
typedef ncclResult_t (*ncclRedOpCreatePreMulSum_fn_t)(ncclRedOp_t*          op,
                                                      void*                 scalar,
                                                      ncclDataType_t        datatype,
                                                      ncclScalarResidence_t residence,
                                                      ncclComm_t            comm);
typedef ncclResult_t (*ncclRedOpDestroy_fn_t)(ncclRedOp_t op, ncclComm_t comm);
typedef ncclResult_t (*ncclGroupStart_fn_t)();
typedef ncclResult_t (*ncclGroupEnd_fn_t)();
typedef ncclResult_t (*ncclGetVersion_fn_t)(int* version);
typedef ncclResult_t (*ncclGetUniqueId_fn_t)(ncclUniqueId* out);

typedef ncclResult_t (*ncclCommInitRank_fn_t)(ncclComm_t*  newcomm,
                                              int          nranks,
                                              ncclUniqueId commId,
                                              int          myrank);

typedef ncclResult_t (*ncclCommInitAll_fn_t)(ncclComm_t* comms, int ndev, const int* devlist);

typedef ncclResult_t (*ncclCommInitRankConfig_fn_t)(ncclComm_t*   comm,
                                                    int           nranks,
                                                    ncclUniqueId  commId,
                                                    int           myrank,
                                                    ncclConfig_t* config);

typedef ncclResult_t (*ncclCommFinalize_fn_t)(ncclComm_t comm);

typedef ncclResult_t (*ncclCommDestroy_fn_t)(ncclComm_t comm);

typedef ncclResult_t (*ncclCommAbort_fn_t)(ncclComm_t comm);

typedef ncclResult_t (*ncclCommSplit_fn_t)(ncclComm_t    comm,
                                           int           color,
                                           int           key,
                                           ncclComm_t*   newcomm,
                                           ncclConfig_t* config);

typedef const char* (*ncclGetErrorString_fn_t)(ncclResult_t code);

typedef const char* (*ncclGetLastError_fn_t)(const ncclComm_t comm);

typedef ncclResult_t (*ncclCommGetAsyncError_fn_t)(ncclComm_t comm, ncclResult_t* asyncError);

typedef ncclResult_t (*ncclCommCount_fn_t)(const ncclComm_t comm, int* count);

typedef ncclResult_t (*ncclCommCuDevice_fn_t)(const ncclComm_t comm, int* devid);

typedef ncclResult_t (*ncclCommUserRank_fn_t)(const ncclComm_t comm, int* rank);

typedef ncclResult_t (*ncclMemAlloc_fn_t)(void** ptr, size_t size);

typedef ncclResult_t (*ncclMemFree_fn_t)(void* ptr);

typedef ncclResult_t (*mscclLoadAlgo_fn_t)(const char*        mscclAlgoFilePath,
                                           mscclAlgoHandle_t* mscclAlgoHandle,
                                           int                rank);

typedef ncclResult_t (*mscclRunAlgo_fn_t)(const void*       sendBuff,
                                          const size_t      sendCounts[],
                                          const size_t      sDisPls[],
                                          void*             recvBuff,
                                          const size_t      recvCounts[],
                                          const size_t      rDisPls[],
                                          size_t            count,
                                          ncclDataType_t    dataType,
                                          int               root,
                                          int               peer,
                                          ncclRedOp_t       op,
                                          mscclAlgoHandle_t mscclAlgoHandle,
                                          ncclComm_t        comm,
                                          hipStream_t       stream);

typedef ncclResult_t (*mscclUnloadAlgo_fn_t)(mscclAlgoHandle_t mscclAlgoHandle);

typedef ncclResult_t (*ncclCommRegister_fn_t)(const ncclComm_t comm,
                                              void*            buff,
                                              size_t           size,
                                              void**           handle);

typedef ncclResult_t (*ncclCommDeregister_fn_t)(const ncclComm_t comm, void* handle);

typedef struct rcclApiFuncTable
{
    uint64_t                      size;
    ncclAllGather_fn_t            ncclAllGather_fn;
    ncclAllReduce_fn_t            ncclAllReduce_fn;
    ncclAllToAll_fn_t             ncclAllToAll_fn;
    ncclAllToAllv_fn_t            ncclAllToAllv_fn;
    ncclBroadcast_fn_t            ncclBroadcast_fn;
    ncclGather_fn_t               ncclGather_fn;
    ncclReduce_fn_t               ncclReduce_fn;
    ncclReduceScatter_fn_t        ncclReduceScatter_fn;
    ncclScatter_fn_t              ncclScatter_fn;
    ncclSend_fn_t                 ncclSend_fn;
    ncclRecv_fn_t                 ncclRecv_fn;
    ncclRedOpCreatePreMulSum_fn_t ncclRedOpCreatePreMulSum_fn;
    ncclRedOpDestroy_fn_t         ncclRedOpDestroy_fn;
    ncclGroupStart_fn_t           ncclGroupStart_fn;
    ncclGroupEnd_fn_t             ncclGroupEnd_fn;
    ncclGetVersion_fn_t           ncclGetVersion_fn;
    ncclGetUniqueId_fn_t          ncclGetUniqueId_fn;
    ncclCommInitRank_fn_t         ncclCommInitRank_fn;
    ncclCommInitAll_fn_t          ncclCommInitAll_fn;
    ncclCommInitRankConfig_fn_t   ncclCommInitRankConfig_fn;
    ncclCommFinalize_fn_t         ncclCommFinalize_fn;
    ncclCommDestroy_fn_t          ncclCommDestroy_fn;
    ncclCommAbort_fn_t            ncclCommAbort_fn;
    ncclCommSplit_fn_t            ncclCommSplit_fn;
    ncclGetErrorString_fn_t       ncclGetErrorString_fn;
    ncclGetLastError_fn_t         ncclGetLastError_fn;
    ncclCommGetAsyncError_fn_t    ncclCommGetAsyncError_fn;
    ncclCommCount_fn_t            ncclCommCount_fn;
    ncclCommCuDevice_fn_t         ncclCommCuDevice_fn;
    ncclCommUserRank_fn_t         ncclCommUserRank_fn;
    ncclMemAlloc_fn_t             ncclMemAlloc_fn;
    ncclMemFree_fn_t              ncclMemFree_fn;
    mscclLoadAlgo_fn_t            mscclLoadAlgo_fn;
    mscclRunAlgo_fn_t             mscclRunAlgo_fn;
    mscclUnloadAlgo_fn_t          mscclUnloadAlgo_fn;
    ncclCommRegister_fn_t         ncclCommRegister_fn;
    ncclCommDeregister_fn_t       ncclCommDeregister_fn;

} rcclApiFuncTable;

RCCL_EXTERN_C_FINI
