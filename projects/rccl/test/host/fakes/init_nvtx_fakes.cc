/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/init_nvtx.cc. No state, so no reset.

#include <cstdint>

#include "nccl.h"
#include "nccl_fakes.h"  // g_loadParam, for the NCCL_PARAM default this stands in for

void initNvtxRegisteredEnums() {}

// Referenced by init.cc but not declared inside it, so the redirected NCCL_PARAM does not cover it.
int64_t ncclParamNvtxDisable() { return g_loadParam("NVTX_DISABLE", 0); }  // init_nvtx.cc:16
