/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See dev_runtime_fakes.h. ncclDevrFindWindow / ncclDevrIsOneLsaTeam remain in
// nccl_stubs.cc as part of that fail-loud floor; move them here when one of them
// needs a seam.

#include "dev_runtime_fakes.h"

struct ncclDevrWindow;

bool g_devrWindowIsMultiSegment = false;
bool g_devrWindowHasSysmemSegment = false;

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow*) { return g_devrWindowIsMultiSegment; }
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow*) { return g_devrWindowHasSysmemSegment; }

static ncclResult_t DefaultDevrInitOnce(struct ncclComm*) { return ncclSuccess; }

// Identity by default: the world rank is the LSA rank unless a test says otherwise.
static ncclResult_t DefaultDevrWorldToLsaRank(struct ncclComm*, int peerWorldRank,
                                              int* peerLsaRank) {
  if (peerLsaRank) *peerLsaRank = peerWorldRank;
  return ncclSuccess;
}

// Null by default. A test that cares what address the unit computed installs a
// hook returning its own storage.
static ncclResult_t DefaultDevrGetLsaRankPtr(struct ncclComm*, struct ncclDevrWindow*, size_t,
                                             int, void** outPtr) {
  if (outPtr) *outPtr = nullptr;
  return ncclSuccess;
}

std::function<ncclResult_t(struct ncclComm*)> g_devrInitOnce = DefaultDevrInitOnce;
std::function<ncclResult_t(struct ncclComm*, int, int*)>
    g_devrWorldToLsaRank = DefaultDevrWorldToLsaRank;
std::function<ncclResult_t(struct ncclComm*, struct ncclDevrWindow*, size_t, int, void**)>
    g_devrGetLsaRankPtr = DefaultDevrGetLsaRankPtr;

ncclResult_t ncclDevrInitOnce(struct ncclComm* comm) { return g_devrInitOnce(comm); }
ncclResult_t ncclDevrWorldToLsaRank(struct ncclComm* comm, int peerWorldRank, int* peerLsaRank) {
  return g_devrWorldToLsaRank(comm, peerWorldRank, peerLsaRank);
}
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost,
                                   size_t offset, int lsaRank, void** outPtr) {
  return g_devrGetLsaRankPtr(comm, winHost, offset, lsaRank, outPtr);
}

void ResetDevRuntimeFakes() {
  g_devrInitOnce       = DefaultDevrInitOnce;
  g_devrWorldToLsaRank = DefaultDevrWorldToLsaRank;
  g_devrGetLsaRankPtr  = DefaultDevrGetLsaRankPtr;
  g_devrWindowIsMultiSegment = false;
  g_devrWindowHasSysmemSegment = false;
}
