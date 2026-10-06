// Stubs for init.cc API functions that need real RCCL types (ncclResult_t enum,
// ncclComm struct layout). Exercised by TimeoutTests and used by mem_manager.cc.

#include "nccl.h"
#include "comm.h"

extern "C" {
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

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
    if (!comm || !asyncError) return ncclInvalidArgument;
    *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
    return ncclSuccess;
}
} // extern "C"

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
    if (nextState < 0 || nextState >= ncclNumResults || comm == nullptr)
        return ncclInvalidArgument;
    __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
    return ncclSuccess;
}

struct ncclAsyncJob;
ncclResult_t ncclMgmtTaskEnqueue(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*),
                                 void (*)(void*), struct ncclComm*) {
    return ncclSuccess;
}
