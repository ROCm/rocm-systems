/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/rocmwrap.cc fakes. See rocmwrap_fakes.h.

#include "rocmwrap_fakes.h"

#include "nccl.h"
#include "nccl_fakes.h"  // g_loadParam
#include "rocmwrap.h"

ncclResult_t rocmLibraryInit(void) { return ncclSuccess; }

// dmaBufSupported gate: NULL -> unsupported.
PFN_hsa_amd_portable_export_dmabuf pfn_hsa_amd_portable_export_dmabuf = nullptr;

int64_t rcclParamForceEnableDMABUF() { return g_loadParam("RCCL_FORCE_ENABLE_DMABUF", 0); }

// hsa-runtime64, not rocmwrap.cc: the entry rocmwrap.h's HSACHECK reports failures through.
hsa_status_t hsa_status_string(hsa_status_t, const char** status_string) {
  *status_string = "fake HSA error";
  return HSA_STATUS_SUCCESS;
}

void ResetRocmWrapFakes() { pfn_hsa_amd_portable_export_dmabuf = nullptr; }
