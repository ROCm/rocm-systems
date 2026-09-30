// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/timemory.hpp"

// AMD SMI feature flags are owned by the backend layer (single source of truth):
//   AMD_SMI_SDMA_SUPPORTED -> backends/amd_smi/sdma_feature.hpp
//   AINIC_SUPPORTED        -> backends/amd_smi/ainic_feature.hpp

namespace rocprofsys::amd_smi
{
void
config_settings(const std::shared_ptr<settings>&);
}  // namespace rocprofsys::amd_smi
