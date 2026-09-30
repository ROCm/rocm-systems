// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/cper/section_dispatch.hpp"

#include "sections/cper-section-ainic.hpp"
#if defined(ADDC_HAS_FIRERANGE) || defined(ADDC_HAS_GENOA) ||                 \
    defined(ADDC_HAS_TURIN) || defined(ADDC_HAS_VENICE)
#define ADDC_PARSER_HAS_CPU
#endif
#if defined(ADDC_HAS_MI300A) || defined(ADDC_HAS_MI300C) ||                  \
    defined(ADDC_HAS_MI300X) || defined(ADDC_HAS_MI325X) ||                 \
    defined(ADDC_HAS_MI350)
#define ADDC_PARSER_HAS_MI300
#endif
#if defined(ADDC_PARSER_HAS_CPU) || defined(ADDC_PARSER_HAS_MI300)
#include "sections/cper-section-amd-epyc-crashdump.hpp"
#endif
#if defined(ADDC_PARSER_HAS_CPU) || defined(ADDC_HAS_MI450)
#include "sections/cper-section-amd-epyc-mtl.hpp"
#endif
#ifdef ADDC_PARSER_HAS_CPU
#include "sections/cper-section-amd-epyc-cdd.hpp"
#include "sections/cper-section-amd-epyc-pmic.hpp"
#endif
#ifdef ADDC_PARSER_HAS_MI300
#include "sections/cper-section-amd-mi300-crashdump.hpp"
#endif
#if defined(ADDC_PARSER_HAS_MI300) || defined(ADDC_HAS_MI450)
#include "sections/cper-section-amd-mi300-runtime.hpp"
#endif
#ifdef ADDC_HAS_MI450
#include "sections/cper-section-amd-mi450-crashdump.hpp"
#include "sections/cper-section-amd-mi450-mtl.hpp"
#endif
#include "sections/cper-section-ia32x64.hpp"
#include "sections/cper-section-pcie.hpp"

namespace addc::cper
{

SectionClassification describe_section(SectionKind kind) noexcept
{
    switch (kind)
    {
        case SectionKind::AmdMi450Mtl:
            return {kind, "AmdMi450MTL", "AMD MI450 MPx Trace Log Section",
                    "An MI450 MPx Trace Log occurred"};
        case SectionKind::AinicPlatformContext:
            return {kind, "AinicPlatformContext", "AINIC Platform Context",
                    "An AINIC Platform Context Error occurred"};
        case SectionKind::AinicInterface:
            return {kind, "AinicInterface", "AINIC Interface",
                    "An AINIC Interface Error occurred"};
        case SectionKind::AinicPoison:
            return {kind, "AinicPoison", "AINIC Poison",
                    "An AINIC Poison Error occurred"};
        case SectionKind::AinicEcc:
            return {kind, "AinicEcc", "AINIC ECC",
                    "An AINIC ECC Error occurred"};
        case SectionKind::AmdEpycCrashdump:
            return {kind, "AmdEpyc", "AMD EPYC Crashdump",
                    "AMD EPYC Crashdump Occurred"};
        case SectionKind::AmdEpycMtl:
            return {kind, "AmdEpycMTL", "AMD EPYC MPx Trace Log",
                    "An MPx Trace Log occurred"};
        case SectionKind::AmdEpycCdd:
            return {kind, "AmdEpycCDD", "AMD EPYC Core Debug Dump",
                    "A Core Debug Dump occurred"};
        case SectionKind::AmdEpycPmic:
            return {kind, "AmdEpycPMIC", "AMD EPYC PMIC Fault",
                    "A PMIC fault occurred"};
        case SectionKind::AmdMi300Crashdump:
            return {kind, "AmdMi300Crashdump",
                    "AMD MI300 Crashdump Section", {}};
        case SectionKind::AmdMi450Crashdump:
            return {kind, "AmdMi450Crashdump",
                    "AMD MI450 Crashdump Section", {}};
        case SectionKind::AmdMi300Runtime:
            return {kind, "AmdMi300Runtime",
                    "AMD MI300 Runtime Error Section", {}};
        case SectionKind::AmdMi450Runtime:
            // Keep the established JSON body key for compatibility while
            // reporting the correct platform in descriptor metadata.
            return {kind, "AmdMi300Runtime",
                    "AMD MI450 Runtime Error Section", {}};
        case SectionKind::Ia32X64:
            return {kind, "Ia32x64Processor", "Ia32X64Section",
                    "An IA32/x64 Processor Error occurred"};
        case SectionKind::Pcie:
            return {kind, "PCIe", "PCIe SECTION",
                    "A PCIe Error occurred"};
        case SectionKind::Opaque:
            return {};
    }
    return {};
}

SectionKind section_kind_from_type_name(std::string_view type_name) noexcept
{
    static constexpr SectionKind kinds[] = {
        SectionKind::AinicPlatformContext,
        SectionKind::AinicInterface,
        SectionKind::AinicPoison,
        SectionKind::AinicEcc,
#if defined(ADDC_PARSER_HAS_CPU) || defined(ADDC_PARSER_HAS_MI300)
        SectionKind::AmdEpycCrashdump,
#endif
#if defined(ADDC_PARSER_HAS_CPU) || defined(ADDC_HAS_MI450)
        SectionKind::AmdEpycMtl,
#endif
#ifdef ADDC_HAS_MI450
        SectionKind::AmdMi450Mtl,
#endif
#ifdef ADDC_PARSER_HAS_CPU
        SectionKind::AmdEpycCdd,
        SectionKind::AmdEpycPmic,
#endif
#ifdef ADDC_PARSER_HAS_MI300
        SectionKind::AmdMi300Crashdump,
#endif
#if defined(ADDC_PARSER_HAS_MI300) || defined(ADDC_HAS_MI450)
        SectionKind::AmdMi300Runtime,
#endif
#ifdef ADDC_HAS_MI450
        SectionKind::AmdMi450Runtime,
        SectionKind::AmdMi450Crashdump,
#endif
        SectionKind::Ia32X64,
        SectionKind::Pcie,
    };
    for (const auto kind : kinds)
    {
        if (describe_section(kind).type_name == type_name)
        {
            return kind;
        }
    }
    return SectionKind::Opaque;
}

SectionClassification classify_section(
    const RecordHeader& header, const SectionDescriptor& descriptor) noexcept
{
    using namespace sections;
    // This compatibility discriminator intentionally precedes the broad EPYC
    // MTL rule; the condition is visible instead of encoded as priority 4.
#ifdef ADDC_HAS_MI450
    if (matches_amd_mi450_mtl(header, descriptor))
        return describe_section(SectionKind::AmdMi450Mtl);
#endif
    if (matches_ainic_platform_context(header, descriptor))
        return describe_section(SectionKind::AinicPlatformContext);
    if (matches_ainic_interface(header, descriptor))
        return describe_section(SectionKind::AinicInterface);
    if (matches_ainic_poison(header, descriptor))
        return describe_section(SectionKind::AinicPoison);
    if (matches_ainic_ecc(header, descriptor))
        return describe_section(SectionKind::AinicEcc);
#if defined(ADDC_PARSER_HAS_CPU) || defined(ADDC_PARSER_HAS_MI300)
    if (matches_amd_epyc_crashdump(header, descriptor))
        return describe_section(SectionKind::AmdEpycCrashdump);
#endif
#if defined(ADDC_PARSER_HAS_CPU) || defined(ADDC_HAS_MI450)
    if (matches_amd_epyc_mtl(header, descriptor))
        return describe_section(SectionKind::AmdEpycMtl);
#endif
#ifdef ADDC_PARSER_HAS_CPU
    if (matches_amd_epyc_cdd(header, descriptor))
        return describe_section(SectionKind::AmdEpycCdd);
    if (matches_amd_epyc_pmic(header, descriptor))
        return describe_section(SectionKind::AmdEpycPmic);
#endif
#ifdef ADDC_PARSER_HAS_MI300
    if (matches_amd_mi300_crashdump(header, descriptor))
        return describe_section(SectionKind::AmdMi300Crashdump);
#endif
#ifdef ADDC_HAS_MI450
    if (matches_amd_mi450_crashdump(header, descriptor))
        return describe_section(SectionKind::AmdMi450Crashdump);
#endif
    if (matches_ia32x64(header, descriptor))
        return describe_section(SectionKind::Ia32X64);
    if (matches_pcie(header, descriptor))
        return describe_section(SectionKind::Pcie);
#if defined(ADDC_PARSER_HAS_MI300) || defined(ADDC_HAS_MI450)
    if (matches_amd_mi300_runtime(header, descriptor))
    {
#ifdef ADDC_HAS_MI450
        if ((descriptor.revision_major & 0xFU) == 2U &&
            ((descriptor.revision_major >> 4U) & 0xFU) == 3U)
        {
            return describe_section(SectionKind::AmdMi450Runtime);
        }
#endif
        return describe_section(SectionKind::AmdMi300Runtime);
    }
#endif
    return {};
}

nlohmann::json parse_section_json(SectionKind kind,
                                  std::span<const uint8_t> body,
                                  const SectionDescriptor& descriptor,
                                  ParseContext& context)
{
    using namespace sections;
    switch (kind)
    {
        case SectionKind::AinicPlatformContext:
            return parse_ainic_platform_context(body, descriptor);
        case SectionKind::AinicInterface:
            return parse_ainic_interface(body, descriptor);
        case SectionKind::AinicPoison:
            return parse_ainic_poison(body, descriptor);
        case SectionKind::AinicEcc:
            return parse_ainic_ecc(body, descriptor);
        case SectionKind::AmdEpycCrashdump:
#if defined(ADDC_PARSER_HAS_CPU) || defined(ADDC_PARSER_HAS_MI300)
            return parse_amd_epyc_crashdump(body, descriptor, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdEpycMtl:
#if defined(ADDC_PARSER_HAS_CPU) || defined(ADDC_HAS_MI450)
            return parse_amd_epyc_mtl(body, descriptor, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdMi450Mtl:
#ifdef ADDC_HAS_MI450
            return parse_amd_mi450_mtl(body, descriptor, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdEpycCdd:
#ifdef ADDC_PARSER_HAS_CPU
            return parse_amd_epyc_cdd(body, descriptor, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdEpycPmic:
#ifdef ADDC_PARSER_HAS_CPU
            return parse_amd_epyc_pmic(body, descriptor);
#else
            return nullptr;
#endif
        case SectionKind::AmdMi300Crashdump:
#ifdef ADDC_PARSER_HAS_MI300
            return parse_amd_mi300_crashdump(body, descriptor, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdMi450Crashdump:
#ifdef ADDC_HAS_MI450
            return parse_amd_mi450_crashdump(body, descriptor, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdMi300Runtime:
        case SectionKind::AmdMi450Runtime:
#if defined(ADDC_PARSER_HAS_MI300) || defined(ADDC_HAS_MI450)
            return parse_amd_mi300_runtime(body, descriptor, context);
#else
            return nullptr;
#endif
        case SectionKind::Ia32X64:
            return parse_ia32x64(body, descriptor, context);
        case SectionKind::Pcie:
            return parse_pcie(body, descriptor, context);
        case SectionKind::Opaque:
            return nullptr;
    }
    return nullptr;
}

} // namespace addc::cper
