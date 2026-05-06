// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/sdma_feature.hpp"
#include "core/timemory.hpp"

#include <amd_smi/amdsmi.h>

// AMD-SMI >= 26.3 also exposes NIC APIs (gated independently by ROCPROFSYS_USE_AINIC).
// The SDMA feature flag lives in sdma_feature.hpp so that mock_driver.hpp can include
// it directly and avoid the ODR fragility from include-order dependence.
#if AMDSMI_LIB_VERSION_MAJOR > 26 ||                                                     \
    (AMDSMI_LIB_VERSION_MAJOR == 26 && AMDSMI_LIB_VERSION_MINOR > 2)
#    if ROCPROFSYS_USE_AINIC > 0
#        define AINIC_SUPPORTED 1
#    endif
#endif

namespace rocprofsys
{
namespace amd_smi
{
void
config_settings(const std::shared_ptr<settings>&);
}  // namespace amd_smi
}  // namespace rocprofsys
