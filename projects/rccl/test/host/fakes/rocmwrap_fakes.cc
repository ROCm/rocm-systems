/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/rocmwrap.cc fakes. See rocmwrap_fakes.h.

#include "rocmwrap_fakes.h"

#include "nccl.h"
#include "nccl_fakes.h"
#include "rocmwrap.h"

ncclResult_t rocmLibraryInit(void) { return ncclSuccess; }

// dmaBufSupported gate: NULL -> unsupported.
PFN_hsa_amd_portable_export_dmabuf pfn_hsa_amd_portable_export_dmabuf = nullptr;

#ifndef RCCL_ROCMWRAP_FAKES_OMIT_ncclParamDmaBufEnable
int64_t ncclParamDmaBufEnable() { return g_loadParam("DMABUF_ENABLE", 1); }
#endif
int64_t rcclParamForceEnableDMABUF() { return g_loadParam("RCCL_FORCE_ENABLE_DMABUF", 0); }

void ResetRocmWrapFakes() { pfn_hsa_amd_portable_export_dmabuf = nullptr; }
