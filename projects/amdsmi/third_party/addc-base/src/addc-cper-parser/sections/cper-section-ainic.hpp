// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

namespace addc::cper::sections
{

// --- AINIC Platform Context --------------------------------------------------
/// Parses AINIC Platform Context CPER sections
/// GUID: {3a8f1d2e-7c4b-5e96-af01-23b456c78d9e}
[[nodiscard]] bool matches_ainic_platform_context(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_ainic_platform_context(
    std::span<const uint8_t> body, const SectionDescriptor& desc);

// --- AINIC Interface ---------------------------------------------------------
/// Parses AINIC Interface Error CPER sections
/// GUID: {a0b1c2d3-e4f5-6789-abcd-ef0123456789}
[[nodiscard]] bool matches_ainic_interface(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_ainic_interface(
    std::span<const uint8_t> body, const SectionDescriptor& desc);

// --- AINIC Poison ------------------------------------------------------------
/// Parses AINIC Poison Error CPER sections
/// GUID: {b1c2d3e4-f506-789a-bcde-f01234567890}
[[nodiscard]] bool matches_ainic_poison(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_ainic_poison(
    std::span<const uint8_t> body, const SectionDescriptor& desc);

// --- AINIC ECC ---------------------------------------------------------------
/// Parses AINIC ECC Error CPER sections
/// GUID: {c2d3e4f5-0617-89ab-cdef-012345678901}
[[nodiscard]] bool matches_ainic_ecc(
    const RecordHeader& header, const SectionDescriptor& desc) noexcept;
[[nodiscard]] nlohmann::json parse_ainic_ecc(
    std::span<const uint8_t> body, const SectionDescriptor& desc);

} // namespace addc::cper::sections
