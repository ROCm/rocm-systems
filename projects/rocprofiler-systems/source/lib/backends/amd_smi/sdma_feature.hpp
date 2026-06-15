// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <amd_smi/amdsmi.h>

// AMD-SMI >= 26.3 exposes the SDMA process-list API. Defined in a single
// shared header so every TU that mentions AMD_SMI_SDMA_SUPPORTED resolves it
// identically — required because the macro gates layout-affecting MOCK_METHODs
// in mock_backend.hpp.
#if AMDSMI_LIB_VERSION_MAJOR > 26 ||                                                     \
    (AMDSMI_LIB_VERSION_MAJOR == 26 && AMDSMI_LIB_VERSION_MINOR > 2)
#    define AMD_SMI_SDMA_SUPPORTED 1
#endif
