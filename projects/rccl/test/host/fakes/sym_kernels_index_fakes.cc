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
ASSERT_HOOK_MATCHES_PROD(g_symkTmaKernelMask, ncclSymkTmaKernelMask);
ASSERT_HOOK_MATCHES_PROD(g_symkGetKernelIndex, ncclSymkGetKernelIndex);
ASSERT_HOOK_MATCHES_PROD(g_symkKernelIdToString, ncclSymkKernelIdToString);
ASSERT_HOOK_MATCHES_PROD(g_symkMakeDevWork, ncclSymkMakeDevWork);
ASSERT_HOOK_MATCHES_PROD(g_genkKernelMask, ncclGenkKernelMask);
ASSERT_HOOK_MATCHES_PROD(g_genkGetKernelIndex, ncclGenkGetKernelIndex);
ASSERT_HOOK_MATCHES_PROD(g_genkInitStart, ncclGenkInitStart);
ASSERT_HOOK_MATCHES_PROD(g_genkInitEnd, ncclGenkInitEnd);
#undef ASSERT_HOOK_MATCHES_PROD

static uint64_t DefaultSymkLLKernelMask() { return 0; }
std::function<uint64_t()> g_symkLLKernelMask = DefaultSymkLLKernelMask;
ncclSymkKernelMask ncclSymkLLKernelMask() { return g_symkLLKernelMask(); }

static uint64_t DefaultSymkDynamicSmemKernelMask() { return 0; }
std::function<uint64_t()> g_symkDynamicSmemKernelMask = DefaultSymkDynamicSmemKernelMask;
ncclSymkKernelMask ncclSymkDynamicSmemKernelMask() { return g_symkDynamicSmemKernelMask(); }

static uint64_t DefaultSymkTmaKernelMask() { return 0; }
std::function<uint64_t()> g_symkTmaKernelMask = DefaultSymkTmaKernelMask;
ncclSymkKernelMask ncclSymkTmaKernelMask() { return g_symkTmaKernelMask(); }

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

// NCCL 2.32.6 general (Genk) kernels. AMD builds no Genk kernels: sym_kernels.cc's fallback reports an empty
// mask and index -1, and genk_runtime.cc's init seams default to success here.
static uint64_t DefaultGenkKernelMask() { return 0; }
std::function<uint64_t()> g_genkKernelMask = DefaultGenkKernelMask;
ncclSymkKernelMask ncclGenkKernelMask() { return g_genkKernelMask(); }

static int DefaultGenkGetKernelIndex(ncclSymkKernelId, int, ncclDataType_t) { return -1; }
std::function<int(ncclSymkKernelId, int, ncclDataType_t)> g_genkGetKernelIndex = DefaultGenkGetKernelIndex;
int ncclGenkGetKernelIndex(ncclSymkKernelId kernelId, int red, ncclDataType_t ty) {
  return g_genkGetKernelIndex(kernelId, red, ty);
}

static ncclResult_t DefaultGenkInitStart(struct ncclComm*, bool*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*, bool*)> g_genkInitStart = DefaultGenkInitStart;
ncclResult_t ncclGenkInitStart(struct ncclComm* comm, bool* needGenkDevComm) {
  return g_genkInitStart(comm, needGenkDevComm);
}

static ncclResult_t DefaultGenkInitEnd(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_genkInitEnd = DefaultGenkInitEnd;
ncclResult_t ncclGenkInitEnd(struct ncclComm* comm) { return g_genkInitEnd(comm); }

void* ncclGenkKernelList[ncclSymkKernelId_Count] = {nullptr};
void* ncclGenkKernelListProfile[ncclSymkKernelId_Count] = {nullptr};
int ncclGenkKernelMaxDynamicSmem[ncclSymkKernelId_Count] = {0};

void ResetSymKernelsIndexFakes() {
  g_symkLLKernelMask = DefaultSymkLLKernelMask;
  g_symkDynamicSmemKernelMask = DefaultSymkDynamicSmemKernelMask;
  g_symkTmaKernelMask = DefaultSymkTmaKernelMask;
  g_symkGetKernelIndex = DefaultSymkGetKernelIndex;
  g_symkKernelIdToString = DefaultSymkKernelIdToString;
  g_symkMakeDevWork = DefaultSymkMakeDevWork;
  g_genkKernelMask = DefaultGenkKernelMask;
  g_genkGetKernelIndex = DefaultGenkGetKernelIndex;
  g_genkInitStart = DefaultGenkInitStart;
  g_genkInitEnd = DefaultGenkInitEnd;
  for (int i = 0; i < ncclSymkKernelId_Count; i++) {
    ncclGenkKernelList[i] = nullptr;
    ncclGenkKernelListProfile[i] = nullptr;
    ncclGenkKernelMaxDynamicSmem[i] = 0;
    ncclSymkKernelList[i] = nullptr;  // raw globals, not std::function seams: reset here to avoid cross-test leaks
    ncclSymkKernelListProfile[i] = nullptr;
    ncclSymkKernelMaxDynamicSmem[i] = 0;
  }
}
