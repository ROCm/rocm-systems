// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

#include <nlohmann/json.hpp>

namespace addc::cper::sections
{

/// Parses AMD MI450 Scale Up (UALoE) sections.
/// GUID: 5eee25c9-83a0-234d-82bb-7b7fc140f21f
[[nodiscard]] bool matches_amd_mi450_scaleup(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_amd_mi450_scaleup(
    std::span<const uint8_t> body, const SectionDescriptor& desc);
[[nodiscard]] nlohmann::json parse_amd_mi450_scaleup(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    ParseContext& context);

} // namespace addc::cper::sections
