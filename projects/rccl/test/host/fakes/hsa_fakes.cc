/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Link floor for HSA runtime diagnostics used by host-only microtests.

#include <hsa/hsa.h>

extern "C" hsa_status_t hsa_status_string(hsa_status_t, const char** status) {
  if (status) *status = "micro-hsa-status";
  return HSA_STATUS_SUCCESS;
}
