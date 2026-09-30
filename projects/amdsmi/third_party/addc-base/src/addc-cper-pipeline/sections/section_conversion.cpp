// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

#include "addc/cper/structural.hpp"


namespace addc::pipeline
{

std::optional<AmdEpycCrashdumpSection> parseEpycCrashdumpSection(
    const nlohmann::json&, const nlohmann::json&, std::string*);
std::optional<AmdGpuCrashdumpSection> parseGpuCrashdumpSection(
    const nlohmann::json&, const nlohmann::json&);
std::optional<AmdGpuMi450CrashdumpSection> parseGpuMi450CrashdumpSection(
    const nlohmann::json&, const nlohmann::json&);
std::optional<AmdGpuMi450RuntimeSection> parseGpuMi450RuntimeSection(
    const nlohmann::json&, const nlohmann::json&);
std::optional<AmdGpuRuntimeSection> parseGpuRuntimeSection(
    const nlohmann::json&, const nlohmann::json&);
std::optional<Ia32x64Section> parseIa32x64Section(
    const nlohmann::json&, const nlohmann::json&, std::string*);
std::optional<PcieSection> parsePcieSection(const nlohmann::json&,
                                              const nlohmann::json&);
std::optional<AmdEpycMtlSection> parseMtlSection(const nlohmann::json&,
                                                   const nlohmann::json&);
std::optional<AmdEpycMtlSection> parseMi450MtlSection(
    const nlohmann::json&, const nlohmann::json&);
std::optional<AmdEpycCddSection> parseCddSection(const nlohmann::json&,
                                                   const nlohmann::json&);
std::optional<AmdEpycPmicSection> parsePmicSection(const nlohmann::json&,
                                                     const nlohmann::json&);
std::optional<AinicPlatformContextSection> parseAinicSection(
    const nlohmann::json&, const nlohmann::json&);

SectionDescriptor parse_descriptor(const nlohmann::json& descriptor)
{
    using namespace detail;
    SectionDescriptor value;
    value.guid = extract_guid(descriptor);
    value.flags = parse_descriptor_flags(descriptor);
    value.fru = json_str(descriptor, "fruText").value_or(std::string{});
    value.fru_id =
        json_str(descriptor, "fruID").value_or(std::string{kZeroGuid});
    value.platform = detect_platform_from_descriptor(descriptor, "");
    value.platform_name = std::string{platform_display_name(value.platform)};
    value.addc_version = addc_version_string(descriptor);
    if (const nlohmann::json* revision = json_obj(descriptor, "revision"))
    {
        if (const auto major = json_u32(*revision, "major"))
        {
            value.revision_major = static_cast<uint8_t>(*major);
        }
        if (const auto minor = json_u32(*revision, "minor"))
        {
            value.revision_minor = static_cast<uint8_t>(*minor);
        }
    }
    if (const nlohmann::json* severity = json_obj(descriptor, "severity"))
    {
        value.section_severity =
            json_str(*severity, "name").value_or(std::string{});
    }
    return value;
}

} // namespace addc::pipeline

namespace addc::cper
{

namespace
{

OpaqueSection makeOpaque(const nlohmann::json& descriptor,
                         const nlohmann::json& section, SourceRange source,
                         std::span<const uint8_t> original_bytes)
{
    const auto parsed = pipeline::parse_descriptor(descriptor);
    return OpaqueSection{
        .guid = parsed.guid,
        .revision_major = parsed.revision_major,
        .revision_minor = parsed.revision_minor,
        .source_range = source,
        .raw_data = std::vector<uint8_t>{original_bytes.begin(),
                                         original_bytes.end()},
        .canonical_descriptor = descriptor,
        .canonical_section = section,
    };
}

template <typename T>
SectionPayload typedOrOpaque(std::optional<T> value,
                             const nlohmann::json& descriptor,
                             const nlohmann::json& section,
                             SourceRange source,
                             std::span<const uint8_t> original_bytes)
{
    if (value)
    {
        return SectionPayload{std::move(*value)};
    }
    return SectionPayload{
        makeOpaque(descriptor, section, source, original_bytes)};
}

} // namespace

SectionPayload convert_section(SectionKind kind,
                               const nlohmann::json& descriptor,
                               const nlohmann::json& section,
                               SourceRange source,
                               std::span<const uint8_t> original_bytes,
                               std::string* error_out)
{
    using namespace pipeline;
    switch (kind)
    {
    case SectionKind::AmdEpycCrashdump:
        return typedOrOpaque(parseEpycCrashdumpSection(descriptor, section,
                                                       error_out),
                             descriptor, section, source, original_bytes);
    case SectionKind::AmdMi300Crashdump:
        return typedOrOpaque(parseGpuCrashdumpSection(descriptor, section),
                             descriptor, section, source, original_bytes);
    case SectionKind::AmdMi450Crashdump:
        return typedOrOpaque(
            parseGpuMi450CrashdumpSection(descriptor, section), descriptor,
            section, source, original_bytes);
    case SectionKind::AmdMi300Runtime:
        return typedOrOpaque(parseGpuRuntimeSection(descriptor, section),
                             descriptor, section, source, original_bytes);
    case SectionKind::AmdMi450Runtime:
        return typedOrOpaque(
            parseGpuMi450RuntimeSection(descriptor, section), descriptor,
            section, source, original_bytes);
    case SectionKind::AmdMi450Mtl:
        return typedOrOpaque(parseMi450MtlSection(descriptor, section),
                             descriptor, section, source, original_bytes);
    case SectionKind::AmdEpycMtl:
        return typedOrOpaque(parseMtlSection(descriptor, section), descriptor,
                             section, source, original_bytes);
    case SectionKind::AmdEpycCdd:
        return typedOrOpaque(parseCddSection(descriptor, section), descriptor,
                             section, source, original_bytes);
    case SectionKind::AmdEpycPmic:
        return typedOrOpaque(parsePmicSection(descriptor, section), descriptor,
                             section, source, original_bytes);
    case SectionKind::Ia32X64:
        return typedOrOpaque(
            parseIa32x64Section(descriptor, section, error_out), descriptor,
            section, source, original_bytes);
    case SectionKind::Pcie:
        return typedOrOpaque(parsePcieSection(descriptor, section), descriptor,
                             section, source, original_bytes);
    case SectionKind::AinicPlatformContext:
        return typedOrOpaque(parseAinicSection(descriptor, section), descriptor,
                             section, source, original_bytes);
    case SectionKind::Opaque:
    case SectionKind::AinicInterface:
    case SectionKind::AinicPoison:
    case SectionKind::AinicEcc:
        break;
    }
    return SectionPayload{
        makeOpaque(descriptor, section, source, original_bytes)};
}

} // namespace addc::cper
