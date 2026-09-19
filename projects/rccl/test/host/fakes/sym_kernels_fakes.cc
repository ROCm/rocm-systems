/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See sym_kernels_fakes.h.

#include "sym_kernels_fakes.h"

#include "comm.h"  // also declares ncclDevrWindow, so no forward declaration here
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_getSymRegType, ncclGetSymRegType);
ASSERT_HOOK_MATCHES_PROD(g_symkInitOnce, ncclSymkInitOnce);
ASSERT_HOOK_MATCHES_PROD(g_symkAvailable, ncclSymkAvailable);
ASSERT_HOOK_MATCHES_PROD(g_symkKernelIdIsLL, rcclSymkKernelIdIsLL);
ASSERT_HOOK_MATCHES_PROD(g_symkLLKernelMask, ncclSymkLLKernelMask);
ASSERT_HOOK_MATCHES_PROD(g_symkDynamicSmemKernelMask, ncclSymkDynamicSmemKernelMask);
ASSERT_HOOK_MATCHES_PROD(g_symkGetKernelIndex, ncclSymkGetKernelIndex);
ASSERT_HOOK_MATCHES_PROD(g_symkKernelIdToString, ncclSymkKernelIdToString);
ASSERT_HOOK_MATCHES_PROD(g_symkMakeDevWork, ncclSymkMakeDevWork);
#undef ASSERT_HOOK_MATCHES_PROD

ncclSymRegType_t g_symRegType = ncclSymSendNonregRecvNonreg;
ncclResult_t g_getSymRegTypeResult = ncclSuccess;
int g_getSymRegTypeCalls = 0;

static ncclResult_t DefaultGetSymRegType(struct ncclDevrWindow*, struct ncclDevrWindow*, ncclSymRegType_t* out) {
  ++g_getSymRegTypeCalls;
  if (out) *out = g_symRegType;
  return g_getSymRegTypeResult;
}
std::function<ncclResult_t(struct ncclDevrWindow*, struct ncclDevrWindow*, ncclSymRegType_t*)> g_getSymRegType =
    DefaultGetSymRegType;
ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin,
                               ncclSymRegType_t* out) {
  return g_getSymRegType(sendWin, recvWin, out);
}

static ncclResult_t DefaultSymkInitOnce(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_symkInitOnce = DefaultSymkInitOnce;
ncclResult_t ncclSymkInitOnce(struct ncclComm* comm) { return g_symkInitOnce(comm); }

static bool DefaultSymkAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return false; }
std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t)> g_symkAvailable = DefaultSymkAvailable;
bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int op, ncclDataType_t type, size_t count) {
  return g_symkAvailable(comm, coll, op, type, count);
}

static bool DefaultSymkKernelIdIsLL(int) { return false; }
std::function<bool(int)> g_symkKernelIdIsLL = DefaultSymkKernelIdIsLL;
bool rcclSymkKernelIdIsLL(int kernelId) { return g_symkKernelIdIsLL(kernelId); }

static int DefaultSymkLLKernelMask() { return 0; }
std::function<int()> g_symkLLKernelMask = DefaultSymkLLKernelMask;
int ncclSymkLLKernelMask() { return g_symkLLKernelMask(); }

static int DefaultSymkDynamicSmemKernelMask() { return 0; }
std::function<int()> g_symkDynamicSmemKernelMask = DefaultSymkDynamicSmemKernelMask;
int ncclSymkDynamicSmemKernelMask() { return g_symkDynamicSmemKernelMask(); }

// Index 0 always: the kernel-table arrays below are sized ncclSymkKernelId_Count but only index 0 is populated.
static int DefaultSymkGetKernelIndex(ncclSymkKernelId, int, ncclDataType_t) { return 0; }
std::function<int(ncclSymkKernelId, int, ncclDataType_t)> g_symkGetKernelIndex = DefaultSymkGetKernelIndex;
int ncclSymkGetKernelIndex(ncclSymkKernelId kernelId, int red, ncclDataType_t ty) {
  return g_symkGetKernelIndex(kernelId, red, ty);
}

static const char* DefaultSymkKernelIdToString(int) { return "fake-sym-kernel"; }
std::function<const char*(int)> g_symkKernelIdToString = DefaultSymkKernelIdToString;
const char* ncclSymkKernelIdToString(int kernelId) { return g_symkKernelIdToString(kernelId); }

static ncclResult_t DefaultSymkMakeDevWork(struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*)> g_symkMakeDevWork =
    DefaultSymkMakeDevWork;
ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork) {
  return g_symkMakeDevWork(comm, task, outDevWork);
}

// Sized to fit every enum value g_symkGetKernelIndex could return (not a 1-element placeholder), so a test
// hooking it to a nonzero index still reads in bounds; real GENERATE_SYM_KERNELS builds size these differently.
void* ncclSymkKernelList[ncclSymkKernelId_Count] = {nullptr};
void* ncclSymkKernelListProfile[ncclSymkKernelId_Count] = {nullptr};
int ncclSymkKernelMaxDynamicSmem[ncclSymkKernelId_Count] = {0};

void ResetSymKernelsFakes() {
  g_symRegType = ncclSymSendNonregRecvNonreg;
  g_getSymRegTypeResult = ncclSuccess;
  g_getSymRegTypeCalls = 0;
  g_getSymRegType = DefaultGetSymRegType;
  g_symkInitOnce = DefaultSymkInitOnce;
  g_symkAvailable = DefaultSymkAvailable;
  g_symkKernelIdIsLL = DefaultSymkKernelIdIsLL;
  g_symkLLKernelMask = DefaultSymkLLKernelMask;
  g_symkDynamicSmemKernelMask = DefaultSymkDynamicSmemKernelMask;
  g_symkGetKernelIndex = DefaultSymkGetKernelIndex;
  g_symkKernelIdToString = DefaultSymkKernelIdToString;
  g_symkMakeDevWork = DefaultSymkMakeDevWork;
  for (int i = 0; i < ncclSymkKernelId_Count; i++) {
    ncclSymkKernelList[i] = nullptr;  // raw globals, not std::function seams: reset here to avoid cross-test leaks
    ncclSymkKernelListProfile[i] = nullptr;
    ncclSymkKernelMaxDynamicSmem[i] = 0;
  }
}
