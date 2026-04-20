// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/timemory.hpp"

#include <amd_smi/amdsmi.h>

// AMD-SMI >= 26.3 supports NIC APIs and SDMA usage
#if AMDSMI_LIB_VERSION_MAJOR > 26 ||                                                     \
    (AMDSMI_LIB_VERSION_MAJOR == 26 && AMDSMI_LIB_VERSION_MINOR > 2)
#    if ROCPROFSYS_USE_AINIC > 0
#        define AINIC_SUPPORTED 1
#    endif
#    define AMD_SMI_SDMA_SUPPORTED 1
#endif

namespace rocprofsys
{
namespace amd_smi
{
void
config_settings(const std::shared_ptr<settings>&);
}  // namespace amd_smi
}  // namespace rocprofsys
