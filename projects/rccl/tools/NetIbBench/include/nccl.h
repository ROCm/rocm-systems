/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Minimal nccl.h shim for NetIbBench tools.
//
// Provides only the type definitions that the RCCL net plugin headers
// require (ncclResult_t, ncclRedOp_t, ncclDataType_t) without pulling
// in hip/hip_runtime.h.  This file is picked up instead of the real
// nccl.h because the local include directory appears first in -I order.

#ifndef NCCL_H_
#define NCCL_H_

typedef enum {
    ncclSuccess            = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError        = 2,
    ncclInternalError      = 3,
    ncclInvalidArgument    = 4,
    ncclInvalidUsage       = 5,
    ncclRemoteError        = 6,
    ncclInProgress         = 7,
    ncclNumResults         = 8
} ncclResult_t;

typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;

typedef enum {
    ncclSum      = 0,
    ncclProd     = 1,
    ncclMax      = 2,
    ncclMin      = 3,
    ncclAvg      = 4,
    ncclNumOps   = 5,
    ncclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(ncclRedOp_dummy_t))
} ncclRedOp_t;

typedef enum {
    ncclInt8       = 0, ncclChar   = 0,
    ncclUint8      = 1,
    ncclInt32      = 2, ncclInt    = 2,
    ncclUint32     = 3,
    ncclInt64      = 4,
    ncclUint64     = 5,
    ncclFloat16    = 6, ncclHalf   = 6,
    ncclFloat32    = 7, ncclFloat  = 7,
    ncclFloat64    = 8, ncclDouble = 8,
    ncclBfloat16   = 9,
    ncclFloat8e4m3 = 10,
    ncclFloat8e5m2 = 11,
    ncclNumTypes   = 12
} ncclDataType_t;


static const char* ncclResultStr(ncclResult_t r) {
    switch (r) {
        case ncclSuccess:            return "ncclSuccess";
        case ncclUnhandledCudaError: return "ncclUnhandledCudaError";
        case ncclSystemError:        return "ncclSystemError";
        case ncclInternalError:      return "ncclInternalError";
        case ncclInvalidArgument:    return "ncclInvalidArgument";
        case ncclInvalidUsage:       return "ncclInvalidUsage";
        case ncclRemoteError:        return "ncclRemoteError";
        default:                     return "unknown";
    }
}

#define CHECK_NCCL(call) do {                                        \
    ncclResult_t _r = (call);                                        \
    if (_r != ncclSuccess) {                                         \
        fprintf(stderr, "[rank %d] %s:%d  %s  returned %s (%d)\n",  \
                rank, __FILE__, __LINE__, #call,                     \
                ncclResultStr(_r), (int)_r);                         \
        MPI_Abort(MPI_COMM_WORLD, 1);                                \
    }                                                                \
} while (0)

#endif // NCCL_H_
