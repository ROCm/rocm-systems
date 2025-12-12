/*************************************************************************
 * Stub nccl.h for standalone amdsmi_wrap testing
 * Provides minimal NCCL type definitions required by amdsmi_wrap.h
 ************************************************************************/

#ifndef NCCL_H_
#define NCCL_H_

#include <stdint.h>

// ncclResult_t enum (matches RCCL definition)
typedef enum {
    ncclSuccess                 =  0,
    ncclUnhandledCudaError      =  1,
    ncclSystemError             =  2,
    ncclInternalError           =  3,
    ncclInvalidArgument         =  4,
    ncclInvalidUsage            =  5,
    ncclRemoteError             =  6,
    ncclInProgress              =  7,
    ncclNumResults              =  8
} ncclResult_t;

#endif // NCCL_H_
