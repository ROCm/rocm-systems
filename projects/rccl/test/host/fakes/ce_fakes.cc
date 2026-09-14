/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See ce_fakes.h.

#include "ce_fakes.h"

#include "comm.h"
#include "nccl.h"
#include "signature-drift.h"
#include "sym_kernels.h"  // ncclSymRegType_t

ASSERT_HOOK_MATCHES_PROD(g_ceAvailable, ncclCeAvailable);
ASSERT_HOOK_MATCHES_PROD(g_ceScratchAvailable, ncclCeScratchAvailable);
ASSERT_HOOK_MATCHES_PROD(g_ceLocalReduceBlocks, ncclCeLocalReduceBlocks);
#undef ASSERT_HOOK_MATCHES_PROD

bool g_ceImplemented = false;
bool g_ceAvailableValue = false;
bool g_ceScratchAvailableValue = false;
bool g_hierCeAvailable = false;

static bool DefaultCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_ceAvailableValue;
}
std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t)> g_ceAvailable =
    DefaultCeAvailable;

static bool DefaultCeScratchAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_ceScratchAvailableValue;
}
std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t)> g_ceScratchAvailable =
    DefaultCeScratchAvailable;

static int DefaultCeLocalReduceBlocks(ncclDataType_t, size_t) { return 1; }
std::function<int(ncclDataType_t, size_t)> g_ceLocalReduceBlocks = DefaultCeLocalReduceBlocks;

bool ncclCeImplemented(ncclFunc_t, int, ncclDataType_t) { return g_ceImplemented; }
bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t func, int op, ncclDataType_t type,
                     ncclSymRegType_t regType) {
  return g_ceAvailable(comm, func, op, type, regType);
}
bool ncclCeScratchAvailable(struct ncclComm* comm, ncclFunc_t func, int op, ncclDataType_t type,
                            ncclSymRegType_t regType) {
  return g_ceScratchAvailable(comm, func, op, type, regType);
}
int ncclCeLocalReduceBlocks(ncclDataType_t type, size_t count) { return g_ceLocalReduceBlocks(type, count); }
bool ncclHierCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_hierCeAvailable;
}

void ResetCeFakes() {
  g_ceImplemented = false;
  g_ceAvailableValue = false;
  g_ceScratchAvailableValue = false;
  g_hierCeAvailable = false;
  g_ceAvailable = DefaultCeAvailable;
  g_ceScratchAvailable = DefaultCeScratchAvailable;
  g_ceLocalReduceBlocks = DefaultCeLocalReduceBlocks;
}
