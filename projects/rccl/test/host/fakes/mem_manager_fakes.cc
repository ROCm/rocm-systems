/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/mem_manager.cc fakes. See mem_manager_fakes.h.

#include "mem_manager_fakes.h"

ncclResult_t g_ncclMemManagerInitResult = ncclSuccess;
int g_ncclMemManagerInitCalls = 0;

extern "C" ncclResult_t ncclMemManagerInit(struct ncclComm*) {
  g_ncclMemManagerInitCalls++;
  return g_ncclMemManagerInitResult;
}

void ResetMemManagerFakes() {
  g_ncclMemManagerInitResult = ncclSuccess;
  g_ncclMemManagerInitCalls = 0;
}
