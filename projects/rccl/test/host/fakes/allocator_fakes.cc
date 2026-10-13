/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "allocator_fakes.h"

#include "allocator.h"
#include "comm.h"
#include "fail_loud.h"

static ncclResult_t DefaultShadowPoolToHost(struct ncclShadowPool*, void*, void**) {
  FailLoudUnfaked("allocator_fakes", "ncclShadowPoolToHost");
}
std::function<ncclResult_t(struct ncclShadowPool*, void*, void**)> g_shadowPoolToHost = DefaultShadowPoolToHost;

ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, void* devObj, void** outHostObj) {
  return g_shadowPoolToHost(pool, devObj, outHostObj);
}

void ResetAllocatorFakes() { g_shadowPoolToHost = DefaultShadowPoolToHost; }
