// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

namespace addc::cper::sections
{

/// Parses IA32/x64 processor CPER sections (GUID
/// dc3ea0b0-a144-4797-b95b-53fa242b6e1d) into the normalized JSON IR used by
/// the analyzer pipeline.
[[nodiscard]] bool matches_ia32x64(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_ia32x64(
    std::span<const uint8_t> body, const SectionDescriptor& desc);
[[nodiscard]] nlohmann::json parse_ia32x64(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    ParseContext& context);

} // namespace addc::cper::sections
