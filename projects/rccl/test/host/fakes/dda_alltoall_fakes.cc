/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See dda_alltoall_fakes.h. Host doubles for the DDA AllToAll dispatch surface
// (src/algorithms/dda/alltoall/*.cu, which is GPU code and not linked here) plus
// rcclDdaEnabled (src/rccl_wrap.cc), so ncclAlltoAll_impl's backend-selection
// branches are reachable and observable without a GPU.

#include "dda_alltoall_fakes.h"

#include "hip/hip_runtime.h"  // hipStream_t
#include "nccl.h"

// Match the production signatures (src/algorithms/dda/alltoall/dda_alltoall.h).
// The production headers spell the stream parameter cudaStream_t, which hipify
// rewrites to hipStream_t; this host TU is not hipified, so it uses hipStream_t
// directly -- the same effective type, so the symbols still match at link.
struct ncclComm;

bool g_ddaEnabled = false;
bool g_ddaIpcEligible = false;
bool g_ddaFabricEligible = false;
bool g_ddaFabricLLEligible = false;
bool g_ddaFabricLL128Eligible = false;

bool g_ddaIpcDispatched = false;
bool g_ddaFabricDispatched = false;
bool g_ddaFabricLLDispatched = false;
bool g_ddaFabricLL128Dispatched = false;

bool DdaAlltoAllDispatched() {
  return g_ddaIpcDispatched || g_ddaFabricDispatched || g_ddaFabricLLDispatched || g_ddaFabricLL128Dispatched;
}

void ResetDdaAlltoAllFakes() {
  g_ddaEnabled = false;
  g_ddaIpcEligible = false;
  g_ddaFabricEligible = false;
  g_ddaFabricLLEligible = false;
  g_ddaFabricLL128Eligible = false;
  g_ddaIpcDispatched = false;
  g_ddaFabricDispatched = false;
  g_ddaFabricLLDispatched = false;
  g_ddaFabricLL128Dispatched = false;
}

// rcclDdaEnabled(comm, totalBytes, gfx942Default, gfx950Default, gfx1250Default)
bool rcclDdaEnabled(const ncclComm*, size_t, size_t, size_t, size_t) { return g_ddaEnabled; }

// --- IPC path (the gfx942 branch the ticket exercises) ---
bool ncclAllToAllDdaIpcEligible(ncclComm*, const void*, void*, size_t, ncclDataType_t) { return g_ddaIpcEligible; }
ncclResult_t ncclAllToAllDdaIpc(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t) {
  g_ddaIpcDispatched = true;
  return ncclSuccess;
}

// --- Fabric/VMM path (gfx1250) ---
bool ncclAllToAllDdaFabricEligible(ncclComm*, const void*, void*, size_t, ncclDataType_t) {
  return g_ddaFabricEligible;
}
ncclResult_t ncclAllToAllDdaFabric(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t) {
  g_ddaFabricDispatched = true;
  return ncclSuccess;
}

// --- Fabric LL fast lane ---
bool ncclAllToAllDdaFabricLLEligible(ncclComm*, const void*, void*, size_t, ncclDataType_t) {
  return g_ddaFabricLLEligible;
}
ncclResult_t ncclAllToAllDdaFabricLL(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t) {
  g_ddaFabricLLDispatched = true;
  return ncclSuccess;
}

// --- Fabric LL128 fast lane ---
bool ncclAllToAllDdaFabricLL128Eligible(ncclComm*, const void*, void*, size_t, ncclDataType_t) {
  return g_ddaFabricLL128Eligible;
}
ncclResult_t ncclAllToAllDdaFabricLL128(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t) {
  g_ddaFabricLL128Dispatched = true;
  return ncclSuccess;
}
