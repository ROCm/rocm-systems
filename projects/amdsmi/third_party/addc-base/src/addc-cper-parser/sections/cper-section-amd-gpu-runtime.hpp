// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"
#include "cper-section-amd-gpu-common.hpp"

#include <nlohmann/json.hpp>

namespace addc::cper::sections
{

/// Parses AMD GPU Non-Standard (runtime) error sections. MI300 and MI450 use
/// this section type; `layout` selects the generation-specific fields.
/// GUID: 32ac0c78-2623-48f6-81a2-ac691780551d
/// Match: GUID only. This GUID is unique to GPU non-standard runtime sections.
/// Note: revision check is omitted because some CPER generators populate
/// program_id with a value other than 0x2 (Instinct).
[[nodiscard]] bool matches_amd_gpu_runtime(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_amd_gpu_runtime(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    AmdGpuLayout layout);
[[nodiscard]] nlohmann::json parse_amd_gpu_runtime(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    AmdGpuLayout layout, ParseContext& context);

} // namespace addc::cper::sections
