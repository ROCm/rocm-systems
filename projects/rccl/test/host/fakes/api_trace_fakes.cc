/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The NCCL_API dispatch symbols src/misc/api_trace.cc emits. That layer is not
// linked into the host-only microtests, so each entry point is defined here and
// forwards to the _impl the unit under test provides. No state, so no reset.

#include <cstring>

#include "nccl.h"

extern ncclResult_t ncclCommGetAsyncError_impl(ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  return ncclCommGetAsyncError_impl(comm, asyncError);
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
  if (id) std::memset(id, 0, sizeof(*id));
  return ncclSuccess;
}
