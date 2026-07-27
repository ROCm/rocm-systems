// Stub implementations for RCCL symbols needed by CPU-only test code.
// These replace the real implementations in librccl.so.

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "nccl.h"
#include "nccl_common.h"
#include "debug.h"
#include "comm.h"

int ncclDebugLevel = NCCL_LOG_WARN;
uint64_t ncclDebugMask = NCCL_ALL;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags,
                  const char* filefunc, int line, const char* fmt, ...) {
    if (level > ncclDebugLevel) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

// From init.cc line 4208
const char* ncclGetErrorString(ncclResult_t code) {
    switch (code) {
    case ncclSuccess:            return "no error";
    case ncclUnhandledCudaError: return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError:        return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError:      return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument:    return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage:       return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError:        return "remote process exited or there was a network error";
    case ncclInProgress:         return "NCCL operation in progress";
    case ncclTimeout:            return "timeout";
    default:                     return "unknown result code";
    }
}

// From init.cc line 3399
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
    if (nextState < 0 || nextState >= ncclNumResults || comm == nullptr) {
        return ncclInvalidArgument;
    }
    __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
    return ncclSuccess;
}

// Simplified from init.cc line 4244 — no proxyState/ginState checks needed
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
    if (comm == nullptr || asyncError == nullptr) return ncclInvalidArgument;
    if (comm->startMagic != NCCL_MAGIC || comm->endMagic != NCCL_MAGIC) {
        return ncclInvalidArgument;
    }
    *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
    return ncclSuccess;
}
