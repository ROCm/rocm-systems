/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The register_stubs.cc entries that are controllable seams, not hard aborts.

#ifndef RCCL_TEST_HOST_REGISTER_STUBS_H_
#define RCCL_TEST_HOST_REGISTER_STUBS_H_

#include <cstddef>
#include <functional>

#include "comm.h"
#include "nccl.h"

struct ncclConnector;

using ncclCommCallbackQueue =
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>;

using ncclRegisterCollBuffersFn =
  std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, void**, void**,
                             ncclCommCallbackQueue*, bool*)>;

using ncclRegisterP2pIpcBufferFn =
  std::function<ncclResult_t(struct ncclComm*, void*, size_t, int, int*, void**,
                             ncclCommCallbackQueue*)>;

using ncclRegisterP2pNetBufferFn =
  std::function<ncclResult_t(struct ncclComm*, void*, size_t, struct ncclConnector*, int*, void**,
                             ncclCommCallbackQueue*)>;

extern ncclRegisterCollBuffersFn g_ncclRegisterCollBuffers;
extern ncclRegisterCollBuffersFn g_ncclRegisterCollNvlsBuffers;
extern ncclRegisterP2pIpcBufferFn g_ncclRegisterP2pIpcBuffer;
extern ncclRegisterP2pNetBufferFn g_ncclRegisterP2pNetBuffer;

void ResetRegisterStubs();

#endif  // RCCL_TEST_HOST_REGISTER_STUBS_H_
