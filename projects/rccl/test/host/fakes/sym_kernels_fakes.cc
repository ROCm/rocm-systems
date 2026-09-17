/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See sym_kernels_fakes.h.

#include "sym_kernels_fakes.h"

#include "comm.h"  // also declares ncclDevrWindow, so no forward declaration here

ncclSymRegType_t g_symRegType = ncclSymSendNonregRecvNonreg;
ncclResult_t g_getSymRegTypeResult = ncclSuccess;
int g_getSymRegTypeCalls = 0;

ncclResult_t ncclGetSymRegType(struct ncclDevrWindow*, struct ncclDevrWindow*,
                               ncclSymRegType_t* out) {
  ++g_getSymRegTypeCalls;
  if (out) *out = g_symRegType;
  return g_getSymRegTypeResult;
}

void ResetSymKernelsFakes() {
  g_symRegType = ncclSymSendNonregRecvNonreg;
  g_getSymRegTypeResult = ncclSuccess;
  g_getSymRegTypeCalls = 0;
}
