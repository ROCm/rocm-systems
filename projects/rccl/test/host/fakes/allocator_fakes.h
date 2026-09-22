/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the symbols defined by src/allocator.cc.

#ifndef RCCL_TEST_HOST_ALLOCATOR_FAKES_H_
#define RCCL_TEST_HOST_ALLOCATOR_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclShadowPool;

extern std::function<ncclResult_t(struct ncclShadowPool*, void*, void**)> g_shadowPoolToHost;

void ResetAllocatorFakes();

#endif  // RCCL_TEST_HOST_ALLOCATOR_FAKES_H_
