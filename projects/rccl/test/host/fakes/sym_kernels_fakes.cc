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
ASSERT_HOOK_MATCHES_PROD(g_symkKernelIdIsTma, rcclSymkKernelIdIsTma);
ASSERT_HOOK_MATCHES_PROD(g_symkFinalize, ncclSymkFinalize);
ASSERT_HOOK_MATCHES_PROD(g_symkHierAvailable, ncclSymkHierAvailable);
ASSERT_HOOK_MATCHES_PROD(g_symkHierGroups, ncclSymkHierGroups);
ASSERT_HOOK_MATCHES_PROD(g_symkRsHierLaunchChunkSize, ncclSymkRsHierLaunchChunkSize);
ASSERT_HOOK_MATCHES_PROD(g_symkAlltoAllLLForced, ncclSymkAlltoAllLLForced);
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

static bool DefaultSymkKernelIdIsTma(int) { return false; }
std::function<bool(int)> g_symkKernelIdIsTma = DefaultSymkKernelIdIsTma;
bool rcclSymkKernelIdIsTma(int kernelId) { return g_symkKernelIdIsTma(kernelId); }

static ncclResult_t DefaultSymkFinalize(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_symkFinalize = DefaultSymkFinalize;
ncclResult_t ncclSymkFinalize(struct ncclComm* comm) { return g_symkFinalize(comm); }

static bool DefaultSymkHierAvailable(struct ncclComm*) { return false; }
std::function<bool(struct ncclComm*)> g_symkHierAvailable = DefaultSymkHierAvailable;
bool ncclSymkHierAvailable(struct ncclComm* comm) { return g_symkHierAvailable(comm); }

static int DefaultSymkHierGroups(struct ncclComm*) { return 0; }
std::function<int(struct ncclComm*)> g_symkHierGroups = DefaultSymkHierGroups;
int ncclSymkHierGroups(struct ncclComm* comm) { return g_symkHierGroups(comm); }

static uint32_t DefaultSymkRsHierLaunchChunkSize(struct ncclComm* comm, size_t) {
  return comm->symkState.kcomm.rsHierChunkSize;
}
std::function<uint32_t(struct ncclComm*, size_t)> g_symkRsHierLaunchChunkSize = DefaultSymkRsHierLaunchChunkSize;
uint32_t ncclSymkRsHierLaunchChunkSize(struct ncclComm* comm, size_t blockBytes) {
  return g_symkRsHierLaunchChunkSize(comm, blockBytes);
}

int ncclSymkAgHierMaxBlocksLimit() { return ncclSymkAgHierMaxBlocks; }
size_t ncclSymkAgHierMaxChunkBytesLimit() { return ncclSymkAgHierMaxChunkBytes; }

// NCCL 2.32.6: whether the tuner forced the AlltoAll LL symmetric kernel. Default false, as for a comm whose
// tuning context forces nothing.
static bool DefaultSymkAlltoAllLLForced(struct ncclComm*) { return false; }
std::function<bool(struct ncclComm*)> g_symkAlltoAllLLForced = DefaultSymkAlltoAllLLForced;
bool ncclSymkAlltoAllLLForced(struct ncclComm* comm) { return g_symkAlltoAllLLForced(comm); }

// [RCCL] tuning.cc now calls ncclSymkMask() once per ncclTuningCompute() invocation.
// Return all bits set so no sym kernel entry is pre-filtered in host tests.
// The last flag matches src/sym_kernels.cc; host tests do not model unregistered GIN windows.
ncclSymkKernelMask ncclSymkMask(struct ncclComm*, ncclFunc_t, int, ncclDataType_t,
                                size_t, bool, bool) { return ~ncclSymkKernelMask{0}; }

void ResetSymKernelsFakes() {
  g_symRegType = ncclSymSendNonregRecvNonreg;
  g_getSymRegTypeResult = ncclSuccess;
  g_getSymRegTypeCalls = 0;
  g_getSymRegType = DefaultGetSymRegType;
  g_symkInitOnce = DefaultSymkInitOnce;
  g_symkAvailable = DefaultSymkAvailable;
  g_symkKernelIdIsLL = DefaultSymkKernelIdIsLL;
  g_symkKernelIdIsTma = DefaultSymkKernelIdIsTma;
  g_symkFinalize = DefaultSymkFinalize;
  g_symkHierAvailable = DefaultSymkHierAvailable;
  g_symkHierGroups = DefaultSymkHierGroups;
  g_symkRsHierLaunchChunkSize = DefaultSymkRsHierLaunchChunkSize;
  g_symkAlltoAllLLForced = DefaultSymkAlltoAllLLForced;
}
