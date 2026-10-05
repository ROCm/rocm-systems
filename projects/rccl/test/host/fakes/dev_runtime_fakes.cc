/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See dev_runtime_fakes.h. ncclDevrFindWindow is controllable here; only
// ncclDevrIsOneLsaTeam remains in nccl_stubs.cc as a fail-loud floor.

#include "dev_runtime_fakes.h"

#include "dev_runtime.h"
#include "fail_loud.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_devrFindWindow, ncclDevrFindWindow);
ASSERT_HOOK_MATCHES_PROD(g_devrWindowIsMultiSegment, ncclDevrWindowIsMultiSegment);
ASSERT_HOOK_MATCHES_PROD(g_devrWindowHasSysmemSegment, ncclDevrWindowHasSysmemSegment);
ASSERT_HOOK_MATCHES_PROD(g_devrInitOnce, ncclDevrInitOnce);
ASSERT_HOOK_MATCHES_PROD(g_devrWindowRegisterInGroup, ncclDevrWindowRegisterInGroup);
#undef ASSERT_HOOK_MATCHES_PROD

static ncclResult_t DefaultDevrFindWindow(struct ncclComm*, void const*, struct ncclDevrWindow** window) {
  if (window) *window = nullptr;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, void const*, struct ncclDevrWindow**)> g_devrFindWindow =
    DefaultDevrFindWindow;
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* ptr, struct ncclDevrWindow** window) {
  return g_devrFindWindow(comm, ptr, window);
}

bool g_devrWindowIsMultiSegmentValue = false;
bool g_devrWindowHasSysmemSegmentValue = false;
static bool DefaultDevrWindowIsMultiSegment(struct ncclDevrWindow*) { return g_devrWindowIsMultiSegmentValue; }
static bool DefaultDevrWindowHasSysmemSegment(struct ncclDevrWindow*) { return g_devrWindowHasSysmemSegmentValue; }
std::function<bool(struct ncclDevrWindow*)> g_devrWindowIsMultiSegment =
    DefaultDevrWindowIsMultiSegment;
std::function<bool(struct ncclDevrWindow*)> g_devrWindowHasSysmemSegment =
    DefaultDevrWindowHasSysmemSegment;

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* window) {
  return g_devrWindowIsMultiSegment(window);
}
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* window) {
  return g_devrWindowHasSysmemSegment(window);
}

static ncclResult_t DefaultDevrInitOnce(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_devrInitOnce = DefaultDevrInitOnce;
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm) { return g_devrInitOnce(comm); }

static ncclResult_t DefaultDevrWindowRegisterInGroup(struct ncclComm*, void*, size_t, int, ncclWindow_t*) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclDevrWindowRegisterInGroup");
}
std::function<ncclResult_t(struct ncclComm*, void*, size_t, int, ncclWindow_t*)> g_devrWindowRegisterInGroup =
    DefaultDevrWindowRegisterInGroup;
ncclResult_t ncclDevrWindowRegisterInGroup(struct ncclComm* comm, void* ptr, size_t size, int winFlags,
                                           ncclWindow_t* outWinDev) {
  return g_devrWindowRegisterInGroup(comm, ptr, size, winFlags, outWinDev);
}

// Floor for ce_coll.cc's staging-setup failure path, which deregisters the window it registered.
ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclCommWindowDeregister");
}

// Floors for the LSA addressing ce_coll.cc's copy paths reach.
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm*, struct ncclDevrWindow*, size_t, int, void**) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclDevrGetLsaRankPtr");
}
ncclResult_t ncclDevrWorldToLsaRank(struct ncclComm*, int, int*) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclDevrWorldToLsaRank");
}
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm*, struct ncclDevrWindow*, size_t, struct ncclTeam, void**) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclDevrGetLsaTeamPtrMC");
}

void ResetDevRuntimeFakes() {
  g_devrFindWindow = DefaultDevrFindWindow;
  g_devrWindowIsMultiSegmentValue = false;
  g_devrWindowHasSysmemSegmentValue = false;
  g_devrWindowIsMultiSegment = DefaultDevrWindowIsMultiSegment;
  g_devrWindowHasSysmemSegment = DefaultDevrWindowHasSysmemSegment;
  g_devrInitOnce = DefaultDevrInitOnce;
  g_devrWindowRegisterInGroup = DefaultDevrWindowRegisterInGroup;
}
