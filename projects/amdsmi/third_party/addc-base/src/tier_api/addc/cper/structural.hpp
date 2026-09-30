// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"
#include "addc/diagnostics.hpp"
#include "addc/pipeline/section_data.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace addc::cper
{

enum class SectionKind
{
    Opaque,
    AinicPlatformContext,
    AinicInterface,
    AinicPoison,
    AinicEcc,
    AmdEpycCrashdump,
    AmdEpycMtl,
    AmdMi450Mtl,
    AmdEpycCdd,
    AmdEpycPmic,
    AmdMi300Crashdump,
    AmdMi450Crashdump,
    AmdMi300Runtime,
    AmdMi450Runtime,
    Ia32X64,
    Pcie,
};

struct SourceRange
{
    std::size_t offset{};
    std::size_t length{};

    constexpr bool operator==(const SourceRange&) const noexcept = default;
};

/// Lossless fallback for unknown, excluded, or malformed section bodies.
struct OpaqueSection
{
    std::string guid;
    uint8_t revision_major{};
    uint8_t revision_minor{};
    SourceRange source_range;
    std::vector<uint8_t> raw_data;
    nlohmann::json canonical_descriptor;
    nlohmann::json canonical_section;
};

using SectionPayload = std::variant<
    OpaqueSection, pipeline::AinicPlatformContextSection,
    pipeline::AmdEpycCrashdumpSection, pipeline::Ia32x64Section,
    pipeline::PcieSection, pipeline::AmdGpuCrashdumpSection,
    pipeline::AmdGpuMi450CrashdumpSection,
    pipeline::AmdGpuMi450RuntimeSection, pipeline::AmdGpuRuntimeSection,
    pipeline::AmdEpycMtlSection, pipeline::AmdEpycCddSection,
    pipeline::AmdEpycPmicSection>;

struct StructuralSection
{
    SectionKind kind = SectionKind::Opaque;
    SourceRange source_range;
    std::vector<uint8_t> raw_data;
    nlohmann::json canonical_descriptor;
    nlohmann::json canonical_section;
    SectionPayload payload;
};

struct StructuralRecord
{
    nlohmann::json canonical_header;
    std::vector<StructuralSection> sections;
    std::vector<ParseRepair> repairs;
    Diagnostics boundary_diagnostics;
};

enum class TypedRecordError
{
    InvalidRoot,
    MissingHeader,
    MissingSectionArrays,
    SectionCountMismatch,
    InvalidDescriptor,
    InvalidSourceRange,
};

struct TypedRecordFailure
{
    TypedRecordError code = TypedRecordError::InvalidRoot;
    std::optional<std::size_t> section_index;
};

/// Convert one canonical section to its finite typed alternative. Evidence is
/// passed explicitly so opaque fallback never invents provenance from JSON.
[[nodiscard]] SectionPayload convert_section(
    SectionKind kind, const nlohmann::json& canonical_descriptor,
    const nlohmann::json& canonical_section, SourceRange source,
    std::span<const uint8_t> original_bytes,
    std::string* error_out = nullptr);

/// Convert the canonical parser JSON exactly once while retaining original
/// bytes and every canonical descriptor/section field at the opaque boundary.
[[nodiscard]] std::optional<StructuralRecord> to_typed_record(
    const nlohmann::json& canonical_parser_json,
    std::span<const uint8_t> original_bytes,
    const Diagnostics& boundary_diagnostics = {},
    TypedRecordFailure* failure_out = nullptr);

} // namespace addc::cper
