/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The one register_stubs.cc entry that is a controllable seam, not a hard abort.

#ifndef RCCL_TEST_HOST_REGISTER_STUBS_H_
#define RCCL_TEST_HOST_REGISTER_STUBS_H_

#include <functional>

#include "comm.h"
#include "nccl.h"

using ncclRegisterCollBuffersFn =
  std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, void**, void**,
                             struct ncclIntruQueue<struct ncclCommCallback,
                                                   &ncclCommCallback::next>*,
                             bool*)>;

extern ncclRegisterCollBuffersFn g_ncclRegisterCollBuffers;

void ResetRegisterStubs();

#endif  // RCCL_TEST_HOST_REGISTER_STUBS_H_
