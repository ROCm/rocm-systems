/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See ce_fakes.h.

#include "ce_fakes.h"

#include "nccl.h"
#include "comm.h"
#include "sym_kernels.h"  // ncclSymRegType_t

bool g_ceImplemented = false;
bool g_ceAvailable = false;
bool g_ceScratchAvailable = false;
bool g_hierCeAvailable = false;

bool ncclCeImplemented(ncclFunc_t, int, ncclDataType_t) { return g_ceImplemented; }
bool ncclCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_ceAvailable;
}
bool ncclCeScratchAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_ceScratchAvailable;
}
bool ncclHierCeAvailable(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t) {
  return g_hierCeAvailable;
}

void ResetCeFakes() {
  g_ceImplemented = false;
  g_ceAvailable = false;
  g_ceScratchAvailable = false;
  g_hierCeAvailable = false;
}
