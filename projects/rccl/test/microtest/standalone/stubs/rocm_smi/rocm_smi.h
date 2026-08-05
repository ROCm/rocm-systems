/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Standalone (no-ROCm) stub for <rocm_smi/rocm_smi.h>.
//
// init.cc (with USE_AMDSMI unset) includes rocm_smi_wrap.h, which includes this
// system header only for the RSMI_IO_LINK_TYPE enum used in one wrapper
// declaration (rocm_smi_getLinkInfo). rocm_smi_wrap.cc is NOT compiled into the
// microtest, and init.cc's rocm_smi_* calls live in Tier-E paths that are out of
// scope here, so a minimal type definition is all that's needed to compile.

#pragma once

#include <cstdint>

typedef enum {
  RSMI_IOLINK_TYPE_UNDEFINED = 0,
  RSMI_IOLINK_TYPE_PCIEXPRESS = 1,
  RSMI_IOLINK_TYPE_XGMI = 2,
} RSMI_IO_LINK_TYPE;
