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
ASSERT_HOOK_MATCHES_PROD(g_symkPickKernel, ncclSymkPickKernel);
ASSERT_HOOK_MATCHES_PROD(g_symkKernelIdIsLL, rcclSymkKernelIdIsLL);
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

static ncclResult_t DefaultSymkPickKernel(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, size_t, int,
                                          ncclSymRegType_t, float* time, ncclSymkKernelId* kernelId,
                                          int* maxChannels, int* nWarps, bool* forced) {
  *time = 0.0f;
  *kernelId = ncclSymkKernelId_Count;
  *maxChannels = 0;
  *nWarps = 0;
  *forced = false;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, size_t, int,
                           ncclSymRegType_t, float*, ncclSymkKernelId*, int*, int*, bool*)>
    g_symkPickKernel = DefaultSymkPickKernel;
ncclResult_t ncclSymkPickKernel(struct ncclComm* comm, ncclFunc_t coll, int op, ncclDataType_t type, size_t count,
                                size_t count2, int n, ncclSymRegType_t regType, float* time,
                                ncclSymkKernelId* kernelId, int* maxChannels, int* nWarps, bool* forced) {
  return g_symkPickKernel(comm, coll, op, type, count, count2, n, regType, time, kernelId, maxChannels, nWarps,
                          forced);
}

static bool DefaultSymkKernelIdIsLL(int) { return false; }
std::function<bool(int)> g_symkKernelIdIsLL = DefaultSymkKernelIdIsLL;
bool rcclSymkKernelIdIsLL(int kernelId) { return g_symkKernelIdIsLL(kernelId); }

void ResetSymKernelsFakes() {
  g_symRegType = ncclSymSendNonregRecvNonreg;
  g_getSymRegTypeResult = ncclSuccess;
  g_getSymRegTypeCalls = 0;
  g_getSymRegType = DefaultGetSymRegType;
  g_symkInitOnce = DefaultSymkInitOnce;
  g_symkAvailable = DefaultSymkAvailable;
  g_symkPickKernel = DefaultSymkPickKernel;
  g_symkKernelIdIsLL = DefaultSymkKernelIdIsLL;
}
