/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Link floor for HSA runtime diagnostics used by host-only microtests.
// rma_proxy.cc reaches hsa_status_string directly through HSACHECK rather than
// through a dynamically loaded pfn. Debug/-O0 retains that file-static error
// path even when no test executes it, while host microtest binaries deliberately
// link neither ROCr nor librccl. Keep this owner-neutral runtime stub shared
// instead of defining it in each test TU.

#include <hsa/hsa.h>

extern "C" hsa_status_t hsa_status_string(hsa_status_t, const char** status) {
  if (status) *status = "micro-hsa-status";
  return HSA_STATUS_SUCCESS;
}
