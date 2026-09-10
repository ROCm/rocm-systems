/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/mem_manager.cc.

#ifndef RCCL_TEST_HOST_MEM_MANAGER_FAKES_H_
#define RCCL_TEST_HOST_MEM_MANAGER_FAKES_H_

#include "nccl.h"

struct ncclComm;

extern ncclResult_t g_ncclMemManagerInitResult;
// The fake writes nothing to comm->memManager, so the call count is the only proof init.cc:806 ran.
extern int g_ncclMemManagerInitCalls;

void ResetMemManagerFakes();

#endif  // RCCL_TEST_HOST_MEM_MANAGER_FAKES_H_
