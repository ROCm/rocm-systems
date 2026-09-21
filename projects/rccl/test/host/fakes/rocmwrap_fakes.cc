/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/rocmwrap.cc fakes. See rocmwrap_fakes.h.

#include "rocmwrap_fakes.h"

#include "nccl.h"
#include "rocmwrap.h"

ncclResult_t rocmLibraryInit(void) { return ncclSuccess; }

// dmaBufSupported gate: NULL -> unsupported.
PFN_hsa_amd_portable_export_dmabuf pfn_hsa_amd_portable_export_dmabuf = nullptr;

void ResetRocmWrapFakes() { pfn_hsa_amd_portable_export_dmabuf = nullptr; }
