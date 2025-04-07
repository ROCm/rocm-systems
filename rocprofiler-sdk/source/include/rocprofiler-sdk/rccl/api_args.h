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

#include <rocprofiler-sdk/defines.h>

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

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

// Empty struct has a size of 0 in C but size of 1 in C++.
// This struct is added to the union members which represent
// functions with no arguments to ensure ABI compatibility
typedef struct rocprofiler_rccl_api_no_args
{
    char empty;
} rocprofiler_rccl_api_no_args;

typedef union rocprofiler_rccl_api_retval_t
{
    int32_t     ncclResult_t_retval;
    const char* const_charp_retval;
} rocprofiler_rccl_api_retval_t;

typedef union rocprofiler_rccl_api_args_t
{
    struct
    {
        const void*    sendbuff;
        void*          recvbuff;
        size_t         sendcount;
        ncclDataType_t datatype;
        ncclComm_t     comm;
        hipStream_t    stream;
    } ncclAllGather;
    struct
    {
        const void*      sendbuff;
        void*            recvbuff;
        size_t           count;
        ncclDataType_t   datatype;
        ncclRedOp_t      op;
        struct ncclComm* comm;
        hipStream_t      stream;
    } ncclAllReduce;
    struct
    {
        const void*    sendbuff;
        void*          recvbuff;
        size_t         count;
        ncclDataType_t datatype;
        ncclComm_t     comm;
        hipStream_t    stream;
    } ncclAllToAll;
    struct
    {
        const void*    sendbuff;
        const size_t*  sendcounts;
        const size_t*  sdispls;
        void*          recvbuff;
        const size_t*  recvcounts;
        const size_t*  rdispls;
        ncclDataType_t datatype;
        ncclComm_t     comm;
        hipStream_t    stream;
    } ncclAllToAllv;
    struct
    {
        const void*    sendbuff;
        void*          recvbuff;
        size_t         count;
        ncclDataType_t datatype;
        int            root;
        ncclComm_t     comm;
        hipStream_t    stream;
    } ncclBroadcast;
    struct
    {
        const void*    sendbuff;
        void*          recvbuff;
        size_t         sendcount;
        ncclDataType_t datatype;
        int            root;
        ncclComm_t     comm;
        hipStream_t    stream;
    } ncclGather;
    struct
    {
        const void*    sendbuff;
        void*          recvbuff;
        size_t         count;
        ncclDataType_t datatype;
        ncclRedOp_t    op;
        int            root;
        ncclComm_t     comm;
        hipStream_t    stream;
    } ncclReduce;
    struct
    {
        const void*      sendbuff;
        void*            recvbuff;
        size_t           recvcount;
        ncclDataType_t   datatype;
        ncclRedOp_t      op;
        struct ncclComm* comm;
        hipStream_t      stream;
    } ncclReduceScatter;
    struct
    {
        const void*    sendbuff;
        void*          recvbuff;
        size_t         recvcount;
        ncclDataType_t datatype;
        int            root;
        ncclComm_t     comm;
        hipStream_t    stream;
    } ncclScatter;
    struct
    {
        const void*    sendbuff;
        size_t         count;
        ncclDataType_t datatype;
        int            peer;
        ncclComm_t     comm;
        hipStream_t    stream;
    } ncclSend;
    struct
    {
        void*          recvbuff;
        size_t         count;
        ncclDataType_t datatype;
        int            peer;
        ncclComm_t     comm;
        hipStream_t    stream;
    } ncclRecv;
    struct
    {
        ncclRedOp_t*          op;
        void*                 scalar;
        ncclDataType_t        datatype;
        ncclScalarResidence_t residence;
        ncclComm_t            comm;
    } ncclRedOpCreatePreMulSum;
    struct
    {
        ncclRedOp_t op;
        ncclComm_t  comm;
    } ncclRedOpDestroy;
    struct
    {
        rocprofiler_rccl_api_no_args no_args;
    } ncclGroupStart;
    struct
    {
        rocprofiler_rccl_api_no_args no_args;
    } ncclGroupEnd;
    struct
    {
        int* version;
    } ncclGetVersion;
    struct
    {
        ncclUniqueId* out;
    } ncclGetUniqueId;
    struct
    {
        ncclComm_t*  newcomm;
        int          nranks;
        ncclUniqueId commId;
        int          myrank;
    } ncclCommInitRank;
    struct
    {
        ncclComm_t* comms;
        int         ndev;
        const int*  devlist;
    } ncclCommInitAll;
    struct
    {
        ncclComm_t*   comm;
        int           nranks;
        ncclUniqueId  commId;
        int           myrank;
        ncclConfig_t* config;
    } ncclCommInitRankConfig;
    struct
    {
        ncclComm_t comm;
    } ncclCommFinalize;
    struct
    {
        ncclComm_t comm;
    } ncclCommDestroy;
    struct
    {
        ncclComm_t comm;
    } ncclCommAbort;
    struct
    {
        ncclComm_t    comm;
        int           color;
        int           key;
        ncclComm_t*   newcomm;
        ncclConfig_t* config;
    } ncclCommSplit;
    struct
    {
        ncclResult_t code;
    } ncclGetErrorString;
    struct
    {
        ncclComm_t comm;
    } ncclGetLastError;
    struct
    {
        ncclComm_t    comm;
        ncclResult_t* asyncError;
    } ncclCommGetAsyncError;
    struct
    {
        ncclComm_t comm;
        int*       count;
    } ncclCommCount;
    struct
    {
        ncclComm_t comm;
        int*       devid;
    } ncclCommCuDevice;
    struct
    {
        ncclComm_t comm;
        int*       rank;
    } ncclCommUserRank;
    struct
    {
        void** ptr;
        size_t size;
    } ncclMemAlloc;
    struct
    {
        void* ptr;
    } ncclMemFree;
    struct
    {
        const char*        mscclAlgoFilePath;
        mscclAlgoHandle_t* mscclAlgoHandle;
        int                rank;
    } mscclLoadAlgo;
    struct
    {
        const void*       sendBuff;
        const size_t*     sendCounts;
        const size_t*     sDisPls;
        void*             recvBuff;
        const size_t*     recvCounts;
        const size_t*     rDisPls;
        size_t            count;
        ncclDataType_t    dataType;
        int               root;
        int               peer;
        ncclRedOp_t       op;
        mscclAlgoHandle_t mscclAlgoHandle;
        ncclComm_t        comm;
        hipStream_t       stream;
    } mscclRunAlgo;
    struct
    {
        mscclAlgoHandle_t mscclAlgoHandle;
    } mscclUnloadAlgo;
    struct
    {
        ncclComm_t comm;
        void*      buff;
        size_t     size;
        void**     handle;
    } ncclCommRegister;
    struct
    {
        ncclComm_t comm;
        void*      handle;
    } ncclCommDeregister;

} rocprofiler_rccl_api_args_t;

ROCPROFILER_EXTERN_C_FINI
