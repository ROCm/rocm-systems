/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See dev_runtime_fakes.h. ncclDevrFindWindow is controllable here; ncclDevrIsOneLsaTeam
// stays in nccl_stubs.cc because rccl-UnitTestsMicroSymKernels links that file without this one.

#include "dev_runtime_fakes.h"

#include "dev_runtime.h"
#include "dev_runtime_internal.h"
#include "fail_loud.h"
#include "signature-drift.h"

#include <algorithm>

ASSERT_HOOK_MATCHES_PROD(g_devrFindWindow, ncclDevrFindWindow);
ASSERT_HOOK_MATCHES_PROD(g_devrWindowIsMultiSegment, ncclDevrWindowIsMultiSegment);
ASSERT_HOOK_MATCHES_PROD(g_devrWindowHasSysmemSegment, ncclDevrWindowHasSysmemSegment);
ASSERT_HOOK_MATCHES_PROD(g_devrInitOnce, ncclDevrInitOnce);
ASSERT_HOOK_MATCHES_PROD(g_devrGetLsaRankPtr, ncclDevrGetLsaRankPtr);
ASSERT_HOOK_MATCHES_PROD(g_devrWindowRegisterInGroup, ncclDevrWindowRegisterInGroup);
ASSERT_HOOK_MATCHES_PROD(g_devrNcclCommWindowDeregister, ncclCommWindowDeregister);
ASSERT_HOOK_MATCHES_PROD(g_computeLsaSize, computeLsaSize);
ASSERT_HOOK_MATCHES_PROD(g_devrWinRegEnabled, ncclDevrWinRegEnabled);
ASSERT_HOOK_MATCHES_PROD(g_symTeamObtain, symTeamObtain);
ASSERT_HOOK_MATCHES_PROD(g_findCommAndHostWindowFromDeviceWindow, findCommAndHostWindowFromDeviceWindow);
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

// Same arithmetic as dev_runtime.cc, so ce_coll.cc's receive-window bound sees production sizes.
size_t ncclDevrWindowPeerSafeSize(struct ncclDevrWindow const* win) {
  if (win == nullptr) return 0;
  const size_t allocMin = win->memory != nullptr ? win->memory->lsaMinSize : win->size;
  const size_t memOffset = win->memory == nullptr ? 0 : win->bigOffset - win->memory->bigOffset;
  if (allocMin <= memOffset) return 0;
  return std::min(win->size, allocMin - memOffset);
}

static ncclResult_t DefaultDevrInitOnce(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_devrInitOnce = DefaultDevrInitOnce;
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm) { return g_devrInitOnce(comm); }

static ncclResult_t DefaultDevrGetLsaRankPtr(struct ncclComm*, struct ncclDevrWindow*, size_t, int, void**) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclDevrGetLsaRankPtr");
}
std::function<ncclResult_t(struct ncclComm*, struct ncclDevrWindow*, size_t, int, void**)> g_devrGetLsaRankPtr =
    DefaultDevrGetLsaRankPtr;
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank,
                                   void** outPtr) {
  return g_devrGetLsaRankPtr(comm, winHost, offset, lsaRank, outPtr);
}

static ncclResult_t DefaultDevrWindowRegisterInGroup(struct ncclComm*, void*, size_t, int, ncclWindow_t*) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclDevrWindowRegisterInGroup");
}
std::function<ncclResult_t(struct ncclComm*, void*, size_t, int, ncclWindow_t*)> g_devrWindowRegisterInGroup =
    DefaultDevrWindowRegisterInGroup;
ncclResult_t ncclDevrWindowRegisterInGroup(struct ncclComm* comm, void* ptr, size_t size, int winFlags,
                                           ncclWindow_t* outWinDev) {
  return g_devrWindowRegisterInGroup(comm, ptr, size, winFlags, outWinDev);
}

static ncclResult_t DefaultCommWindowDeregister(ncclComm_t, ncclWindow_t) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclCommWindowDeregister");
}
std::function<ncclResult_t(ncclComm_t, ncclWindow_t)> g_devrNcclCommWindowDeregister = DefaultCommWindowDeregister;
ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  return g_devrNcclCommWindowDeregister(comm, win);
}

static int DefaultComputeLsaSize(struct ncclComm*) { FailLoudUnfaked("dev_runtime_fakes", "computeLsaSize"); }
std::function<int(struct ncclComm*)> g_computeLsaSize = DefaultComputeLsaSize;
int computeLsaSize(struct ncclComm* comm) { return g_computeLsaSize(comm); }

static bool DefaultDevrWinRegEnabled(int, ncclDevrRegisterCapability) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclDevrWinRegEnabled");
}
std::function<bool(int, ncclDevrRegisterCapability)> g_devrWinRegEnabled = DefaultDevrWinRegEnabled;
bool ncclDevrWinRegEnabled(int winFlags, ncclDevrRegisterCapability capability) {
  return g_devrWinRegEnabled(winFlags, capability);
}

static ncclResult_t DefaultSymTeamObtain(struct ncclComm*, struct ncclTeam, bool, bool, bool, bool,
                                         struct ncclDevrTeam**, bool*) {
  FailLoudUnfaked("dev_runtime_fakes", "symTeamObtain");
}
std::function<ncclResult_t(struct ncclComm*, struct ncclTeam, bool, bool, bool, bool, struct ncclDevrTeam**, bool*)>
    g_symTeamObtain = DefaultSymTeamObtain;
ncclResult_t symTeamObtain(struct ncclComm* comm, struct ncclTeam team, bool multimem, bool counted, bool wantsLeUc,
                           bool wantsLeMc, struct ncclDevrTeam** outTeam, bool* needBarrier) {
  return g_symTeamObtain(comm, team, multimem, counted, wantsLeUc, wantsLeMc, outTeam, needBarrier);
}

static ncclResult_t DefaultFindCommAndHostWindow(ncclWindow_t, ncclComm_t*, struct ncclDevrWindow**) {
  FailLoudUnfaked("dev_runtime_fakes", "findCommAndHostWindowFromDeviceWindow");
}
std::function<ncclResult_t(ncclWindow_t, ncclComm_t*, struct ncclDevrWindow**)>
    g_findCommAndHostWindowFromDeviceWindow = DefaultFindCommAndHostWindow;
ncclResult_t findCommAndHostWindowFromDeviceWindow(ncclWindow_t devWindow, ncclComm_t* foundComm,
                                                   struct ncclDevrWindow** hostWindow) {
  return g_findCommAndHostWindowFromDeviceWindow(devWindow, foundComm, hostWindow);
}
ncclComm_t g_foundComm = nullptr;
struct ncclDevrWindow* g_foundWinHost = nullptr;
static ncclResult_t ReturnFoundCommAndWindow(ncclWindow_t, ncclComm_t* foundComm, struct ncclDevrWindow** hostWindow) {
  *foundComm = g_foundComm;
  *hostWindow = g_foundWinHost;
  return ncclSuccess;
}
void UseFoundCommAndWindow() { g_findCommAndHostWindowFromDeviceWindow = ReturnFoundCommAndWindow; }

// One LSA team: world rank is the LSA rank. A split team has no fake mapping.
ncclResult_t ncclDevrWorldToLsaRank(struct ncclComm*, int worldRank, int* lsaRank) {
  if (g_devrIsOneLsaTeamValue) {
    if (lsaRank) *lsaRank = worldRank;
    return ncclSuccess;
  }
  FailLoudUnfaked("dev_runtime_fakes", "ncclDevrWorldToLsaRank");
}
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm*, struct ncclDevrWindow*, size_t, struct ncclTeam, void**) {
  FailLoudUnfaked("dev_runtime_fakes", "ncclDevrGetLsaTeamPtrMC");
}

void ResetDevRuntimeFakes() {
  g_devrFindWindow = DefaultDevrFindWindow;
  g_devrWindowIsMultiSegmentValue = false;
  g_devrWindowHasSysmemSegmentValue = false;
  g_devrIsOneLsaTeamValue = true;
  g_devrWindowIsMultiSegment = DefaultDevrWindowIsMultiSegment;
  g_devrWindowHasSysmemSegment = DefaultDevrWindowHasSysmemSegment;
  g_devrInitOnce = DefaultDevrInitOnce;
  g_devrGetLsaRankPtr = DefaultDevrGetLsaRankPtr;
  g_devrWindowRegisterInGroup = DefaultDevrWindowRegisterInGroup;
  g_devrNcclCommWindowDeregister = DefaultCommWindowDeregister;
  g_computeLsaSize = DefaultComputeLsaSize;
  g_devrWinRegEnabled = DefaultDevrWinRegEnabled;
  g_symTeamObtain = DefaultSymTeamObtain;
  g_findCommAndHostWindowFromDeviceWindow = DefaultFindCommAndHostWindow;
  g_foundComm = nullptr;
  g_foundWinHost = nullptr;
}
