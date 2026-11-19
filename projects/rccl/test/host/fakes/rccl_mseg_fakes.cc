/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See rccl_mseg_fakes.h.

#include "rccl_mseg_fakes.h"

#include "rccl_common.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_rmaMaxInflightRequests, rcclRmaProxyMaxInflightRequests);
ASSERT_HOOK_MATCHES_PROD(g_hostSegmentFdLookup, rcclHostSegmentFdLookup);
#undef ASSERT_HOOK_MATCHES_PROD

std::function<uint32_t(bool, uint32_t)> g_rmaMaxInflightRequests = [](bool, uint32_t n) { return n; };
uint32_t rcclRmaProxyMaxInflightRequests(bool multiSeg, uint32_t maxRequests) {
  return g_rmaMaxInflightRequests(multiSeg, maxRequests);
}

std::function<int(uintptr_t, size_t, uint64_t*)> g_hostSegmentFdLookup = [](uintptr_t, size_t, uint64_t*) {
  return -1;
};
int rcclHostSegmentFdLookup(uintptr_t base, size_t len, uint64_t* offset) {
  return g_hostSegmentFdLookup(base, len, offset);
}
