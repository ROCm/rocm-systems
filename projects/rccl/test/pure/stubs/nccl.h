// Minimal nccl.h stub for CPU-only RCCL unit tests.
// Provides result codes and basic types without pulling in HIP.
#pragma once

#include <cstdint>
#include <cstddef>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef enum {
    ncclSuccess                 =  0,
    ncclUnhandledCudaError      =  1,
    ncclSystemError             =  2,
    ncclInternalError           =  3,
    ncclInvalidArgument         =  4,
    ncclInvalidUsage            =  5,
    ncclRemoteError             =  6,
    ncclInProgress              =  7,
    ncclTimeout                 =  8,
    ncclNumResults              =  9
} ncclResult_t;

typedef struct ncclComm* ncclComm_t;

// Public API declarations used by TimeoutTests
const char* ncclGetErrorString(ncclResult_t code);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);

// Data types (needed by CollCommon.h and other headers)
typedef enum {
    ncclInt8       = 0, ncclChar       = 0,
    ncclUint8      = 1,
    ncclInt32      = 2, ncclInt        = 2,
    ncclUint32     = 3,
    ncclInt64      = 4,
    ncclUint64     = 5,
    ncclFloat16    = 6, ncclHalf       = 6,
    ncclFloat32    = 7, ncclFloat      = 7,
    ncclFloat64    = 8, ncclDouble     = 8,
    ncclBfloat16   = 9,
    ncclFp8E4M3    = 10,
    ncclFp8E5M2    = 11,
    ncclNumTypes   = 12
} ncclDataType_t;

typedef enum {
    ncclSum        = 0,
    ncclProd       = 1,
    ncclMax        = 2,
    ncclMin        = 3,
    ncclAvg        = 4,
    ncclNumOps     = 5,
    ncclMaxRedOp   = 0x7fff
} ncclRedOp_t;
