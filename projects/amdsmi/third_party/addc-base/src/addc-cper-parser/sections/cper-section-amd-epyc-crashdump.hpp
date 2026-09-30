// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

namespace addc::cper::sections
{

/// Independently parses AMD EPYC Crashdump CPER sections from the AMD
/// specification and emits a libcper-compatible JSON representation.
///
/// Matches GUID 32ac0c78-2623-48f6-b0d0-7365725fd6ae for EPYC and embedded
/// program generations, using the revision major byte to discriminate from the
/// AMD GPU non-standard crashdump parser which shares the same GUID for MI300
/// GPU sections (program_rev == 2, ras_gen == 2).
[[nodiscard]] bool matches_amd_epyc_crashdump(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_amd_epyc_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& desc);
[[nodiscard]] nlohmann::json parse_amd_epyc_crashdump(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    ParseContext& context);

} // namespace addc::cper::sections
