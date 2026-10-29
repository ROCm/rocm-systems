// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"
#include "cper-section-amd-gpu-common.hpp"

#include <nlohmann/json.hpp>

namespace addc::cper::sections
{

/// AMD GPU crashdump sections. MI300 (RAS gen 2) and MI450 (RAS gen 3) use
/// the same body walk; `layout` selects the generation-specific fields.
/// GUID: 32ac0c78-2623-48f6-b0d0-7365725fd6ae (shared with EPYC; the
/// descriptor revision selects the format).
[[nodiscard]] bool matches_amd_mi300_crashdump(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] bool matches_amd_mi450_crashdump(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;

[[nodiscard]] nlohmann::json parse_amd_gpu_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    AmdGpuLayout layout);
[[nodiscard]] nlohmann::json parse_amd_gpu_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    AmdGpuLayout layout, ParseContext& context);

} // namespace addc::cper::sections
