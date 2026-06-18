// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <amd_smi/amdsmi.h>

// AMD-SMI >= 26.3 exposes the AMD NIC (AINIC) APIs, enabled only when the user
// opted in via ROCPROFSYS_USE_AINIC. Owned by the backend layer so AMD SMI
// feature detection has a single source of truth; consumers (e.g. core/amd_smi)
// include this header rather than re-deriving the version check.
#if(AMDSMI_LIB_VERSION_MAJOR > 26 ||                                                     \
    (AMDSMI_LIB_VERSION_MAJOR == 26 && AMDSMI_LIB_VERSION_MINOR > 2)) &&                 \
    ROCPROFSYS_USE_AINIC > 0
#    define AINIC_SUPPORTED 1
#endif
