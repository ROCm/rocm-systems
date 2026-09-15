/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See ce_fakes.h.

#include "ce_fakes.h"

#include <cstdlib>

#include "ce_coll.h"

#include "nccl.h"
#include "comm.h"
#include "sym_kernels.h"  // ncclSymRegType_t

bool g_ceImplemented = false;
bool g_ceAvailable = false;
bool g_ceScratchAvailable = false;
bool g_hierCeAvailable = false;

bool ncclCeImplemented(ncclFunc_t, int, ncclDataType_t) { return g_ceImplemented; }
bool ncclCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_ceAvailable;
}
bool ncclCeScratchAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_ceScratchAvailable;
}
bool ncclHierCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
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
static ncclResult_t DefaultCeLaunchBatchOps(struct ncclComm*, struct ncclCeBatchOpsParams*,
                                            hipStream_t, struct ncclCeCollArgs*) {
  return ncclSuccess;
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
  g_ceAvailable = false;
  g_ceScratchAvailable = false;
  g_hierCeAvailable = false;
}
