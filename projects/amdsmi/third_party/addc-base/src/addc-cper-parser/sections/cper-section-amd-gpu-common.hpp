// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace addc::cper::sections
{

/// Field layout of the AMD GPU section bodies. MI300 and MI450 use the same
/// outer format; PLDM_BNDL and the GPU Error Info structure differ. The
/// section dispatcher selects the layout from the section kind.
enum class AmdGpuLayout
{
    Mi300,
    Mi450,
};

/// PLDM_BNDL (body offset 12).
[[nodiscard]] nlohmann::json parse_amd_gpu_pldm_bundle(uint32_t raw,
                                                       AmdGpuLayout layout);

/// 64-byte GPU Error Info structure at `base`. Returns null when the
/// structure does not fit in `body`.
[[nodiscard]] nlohmann::json parse_amd_gpu_error_info(
    std::span<const uint8_t> body, std::size_t base, AmdGpuLayout layout);

} // namespace addc::cper::sections
