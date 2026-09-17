/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See ce_fakes.h.

#include "fail_loud.h"
#include "ce_fakes.h"

#include <cstdlib>

#include "ce_coll.h"
#include "comm.h"
#include "nccl.h"
#include "signature-drift.h"
#include "sym_kernels.h"  // ncclSymRegType_t

ASSERT_HOOK_MATCHES_PROD(g_ceAvailable, ncclCeAvailable);
ASSERT_HOOK_MATCHES_PROD(g_ceScratchAvailable, ncclCeScratchAvailable);
ASSERT_HOOK_MATCHES_PROD(g_ceLocalReduceBlocks, ncclCeLocalReduceBlocks);
ASSERT_HOOK_MATCHES_PROD(g_ceInitBatchOpsParams, ncclCeInitBatchOpsParams);
ASSERT_HOOK_MATCHES_PROD(g_ceLaunchBatchOps, ncclCeLaunchBatchOps);
#undef ASSERT_HOOK_MATCHES_PROD

bool g_ceImplemented = false;
bool g_ceAvailableValue = false;
bool g_ceScratchAvailableValue = false;
bool g_hierCeAvailable = false;

static bool DefaultCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t,
                               struct ncclDevrWindow*, struct ncclDevrWindow*) {
  return g_ceAvailableValue;
}
std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t,
                   struct ncclDevrWindow*, struct ncclDevrWindow*)>
    g_ceAvailable = DefaultCeAvailable;

static bool DefaultCeScratchAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_ceScratchAvailableValue;
}
std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t)> g_ceScratchAvailable =
    DefaultCeScratchAvailable;

static int DefaultCeLocalReduceBlocks(ncclDataType_t, size_t) { return 1; }
std::function<int(ncclDataType_t, size_t)> g_ceLocalReduceBlocks = DefaultCeLocalReduceBlocks;

bool ncclCeImplemented(ncclFunc_t, int, ncclDataType_t) { return g_ceImplemented; }
bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t func, int op, ncclDataType_t type,
                     ncclSymRegType_t regType, struct ncclDevrWindow* sendWin,
                     struct ncclDevrWindow* recvWin) {
  return g_ceAvailable(comm, func, op, type, regType, sendWin, recvWin);
}
bool ncclCeScratchAvailable(struct ncclComm* comm, ncclFunc_t func, int op, ncclDataType_t type,
                            ncclSymRegType_t regType) {
  return g_ceScratchAvailable(comm, func, op, type, regType);
}
int ncclCeLocalReduceBlocks(ncclDataType_t type, size_t count) { return g_ceLocalReduceBlocks(type, count); }
bool ncclHierCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t,
                         struct ncclDevrWindow*, struct ncclDevrWindow*) {
  return g_hierCeAvailable;
}

// Allocates the op arrays, as src/ce_coll.cc does. A unit that builds a batch
// writes straight into params->srcs[numOps] and friends, so a fake that leaves
// them null faults at the first op rather than reporting anything useful.
static ncclResult_t DefaultCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int capacity) {
  if (params == nullptr) return ncclInvalidArgument;
  *params = {};
  const size_t n = capacity > 0 ? static_cast<size_t>(capacity) : 1;
  params->srcs = static_cast<void**>(calloc(n, sizeof(void*)));
  params->dsts = static_cast<void**>(calloc(n, sizeof(void*)));
  params->sizes = static_cast<size_t*>(calloc(n, sizeof(size_t)));
  if (!params->srcs || !params->dsts || !params->sizes) return ncclSystemError;
  return ncclSuccess;
}
// Fail-loud: submitting a batch is the observable act of these units, so a
// fixture that reaches this without having said what happens to the batch is
// not testing anything -- silently succeeding would hide the call entirely.
static ncclResult_t DefaultCeLaunchBatchOps(struct ncclComm*, struct ncclCeBatchOpsParams*,
                                            hipStream_t, struct ncclCeCollArgs*) {
  FailLoudUnfaked("ce_fakes", "ncclCeLaunchBatchOps");
}

std::function<ncclResult_t(struct ncclCeBatchOpsParams*, int)>
    g_ceInitBatchOpsParams = DefaultCeInitBatchOpsParams;
std::function<ncclResult_t(struct ncclComm*, struct ncclCeBatchOpsParams*, hipStream_t,
                           struct ncclCeCollArgs*)>
    g_ceLaunchBatchOps = DefaultCeLaunchBatchOps;

ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int capacity) {
  return g_ceInitBatchOpsParams(params, capacity);
}
ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params,
                                  hipStream_t stream, struct ncclCeCollArgs* profilerArgs) {
  return g_ceLaunchBatchOps(comm, params, stream, profilerArgs);
}
// Paired with Init above; nothing asserts on the free, so no seam.
void ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params) {
  if (params == nullptr) return;
  free(params->srcs);
  free(params->dsts);
  free(params->sizes);
  params->srcs = nullptr;
  params->dsts = nullptr;
  params->sizes = nullptr;
  params->numOps = 0;
}

void ResetCeFakes() {
  g_ceInitBatchOpsParams = DefaultCeInitBatchOpsParams;
  g_ceLaunchBatchOps     = DefaultCeLaunchBatchOps;
  g_ceImplemented = false;
  g_ceAvailableValue = false;
  g_ceScratchAvailableValue = false;
  g_hierCeAvailable = false;
  g_ceAvailable = DefaultCeAvailable;
  g_ceScratchAvailable = DefaultCeScratchAvailable;
  g_ceLocalReduceBlocks = DefaultCeLocalReduceBlocks;
}
