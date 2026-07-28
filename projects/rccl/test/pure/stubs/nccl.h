// Minimal nccl.h stub for CPU-only RCCL unit tests.
#pragma once

#include <cstdint>
#include <cstddef>
#include <climits>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define NCCL_VERSION_CODE 23004
#define NCCL_API_MAGIC 0xcafebeef
#define NCCL_CONFIG_UNDEF_INT INT_MIN
#define NCCL_CONFIG_UNDEF_PTR NULL
#define NCCL_NET_HANDLE_MAXSIZE 256
#define NCCL_UUID_NBYTES 16

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
struct ncclWindow_vidmem;
typedef struct ncclWindow_vidmem* ncclWindow_t;

extern "C" {
const char* ncclGetErrorString(ncclResult_t code);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
}

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

typedef enum {
  ncclStatGpuMemSuspend      = 0,
  ncclStatGpuMemSuspended    = 1,
  ncclStatGpuMemPersist      = 2,
  ncclStatGpuMemTotal        = 3
} ncclCommMemStat_t;

struct ncclUniqueId { char internal[128]; };

struct ncclConfig_v21701 {
  size_t size; unsigned int magic; unsigned int version;
  int blocking; int cgaClusterSize; int minCTAs; int maxCTAs;
  const char* netName; int splitShare; int trafficClass;
  const char* commName; int collnetEnable; int CTAPolicy;
  int shrinkShare; int nvlsCTAs; int nChannelsPerNetPeer;
  int nvlinkCentricSched;
};
typedef struct ncclConfig_v21701 ncclConfig_t;

#define NCCL_NUM_FUNCTIONS 5
