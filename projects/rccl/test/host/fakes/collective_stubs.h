/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The one hookable seam in collective_stubs.cc; everything else there stays a
// hard ::abort() floor. See collective_stubs.cc's own file comment for why.

#ifndef RCCL_TEST_HOST_COLLECTIVE_STUBS_H_
#define RCCL_TEST_HOST_COLLECTIVE_STUBS_H_

#include <functional>

#include "nccl.h"

struct ncclArgsInfo;

extern std::function<ncclResult_t(struct ncclArgsInfo*)> g_ncclArgsGlobalCheck;

void ResetCollectiveStubs();

#endif  // RCCL_TEST_HOST_COLLECTIVE_STUBS_H_
