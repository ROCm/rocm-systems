/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See sym_kernels_index_fakes.h.

#include "sym_kernels_index_fakes.h"

#include "comm.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_symkLLKernelMask, ncclSymkLLKernelMask);
ASSERT_HOOK_MATCHES_PROD(g_symkDynamicSmemKernelMask, ncclSymkDynamicSmemKernelMask);
ASSERT_HOOK_MATCHES_PROD(g_symkGetKernelIndex, ncclSymkGetKernelIndex);
ASSERT_HOOK_MATCHES_PROD(g_symkKernelIdToString, ncclSymkKernelIdToString);
ASSERT_HOOK_MATCHES_PROD(g_symkMakeDevWork, ncclSymkMakeDevWork);
#undef ASSERT_HOOK_MATCHES_PROD

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

void ResetSymKernelsIndexFakes() {
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
