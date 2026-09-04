/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams bootstrap_stubs.cc OWNS, declared once so a signature change is a compile error rather than a link mismatch.

#ifndef RCCL_TEST_HOST_BOOTSTRAP_STUBS_H_
#define RCCL_TEST_HOST_BOOTSTRAP_STUBS_H_

#include <cstdint>
#include <functional>

#include "nccl.h"

struct ncclBootstrapHandle;
struct ncclComm;

// src/bootstrap.cc. Fail-loud by default: only a test that scripts one may reach it.
extern std::function<ncclResult_t(int /*nHandles*/, void* /*handle*/, struct ncclComm* /*comm*/,
                                  struct ncclComm* /*parent*/)>
    g_bootstrapInit;

extern std::function<ncclResult_t(uint64_t /*commHash*/, struct ncclComm* /*comm*/, struct ncclComm* /*parent*/,
                                  int /*color*/, int /*key*/, int* /*parentRanks*/)>
    g_bootstrapSplit;

extern std::function<ncclResult_t(struct ncclBootstrapHandle* /*handle*/, bool /*idFromEnv*/)> g_bootstrapCreateRoot;

void ResetBootstrapStubs();

#endif  // RCCL_TEST_HOST_BOOTSTRAP_STUBS_H_
