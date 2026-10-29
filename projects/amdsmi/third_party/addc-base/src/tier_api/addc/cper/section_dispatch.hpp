// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/structural.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <span>
#include <string_view>

namespace addc::cper
{

struct SectionClassification
{
    SectionKind kind = SectionKind::Opaque;
    std::string_view json_key = "Unknown";
    std::string_view type_name = "Unknown";
    std::string_view description;
};

/// One explicit, precedence-aware classifier replaces parser registration,
/// numeric priorities, and first-match-wins behavior.
[[nodiscard]] SectionClassification classify_section(
    const RecordHeader& header, const SectionDescriptor& descriptor) noexcept;

[[nodiscard]] SectionClassification describe_section(SectionKind kind) noexcept;

/// True when an AMD GPU runtime section descriptor revision selects the MI450
/// layout (RAS generation 3, Instinct program).
[[nodiscard]] bool is_mi450_gpu_runtime_revision(
    uint8_t revision_major) noexcept;
[[nodiscard]] SectionKind section_kind_from_type_name(
    std::string_view type_name) noexcept;

/// Directly invoke the parser function selected by classify_section().
[[nodiscard]] nlohmann::json parse_section_json(
    SectionKind kind, std::span<const uint8_t> body,
    const SectionDescriptor& descriptor, ParseContext& context);

} // namespace addc::cper
