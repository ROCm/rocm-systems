// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/cper/section_dispatch.hpp"

#include "addc/cper/amd_section_ids.hpp"

#if defined(ADDC_HAS_EMBEDDED_2005) || defined(ADDC_HAS_GENOA) ||                 \
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
#if defined(ADDC_PARSER_HAS_MI300) || defined(ADDC_HAS_MI450)
#include "sections/cper-section-amd-gpu-crashdump.hpp"
#include "sections/cper-section-amd-gpu-runtime.hpp"
#endif
#ifdef ADDC_HAS_MI450
#include "sections/cper-section-amd-mi450-mtl.hpp"
#include "sections/cper-section-amd-mi450-scaleup.hpp"
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
        case SectionKind::AmdMi450ScaleUp:
            return {kind, "AmdMi450ScaleUp", "AMD MI450 Scale Up Section",
                    "An MI450 Scale Up (UALoE) event occurred"};
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

bool is_mi450_gpu_runtime_revision(uint8_t revision_major) noexcept
{
    return program_gen(revision_major) == 2U && ras_gen(revision_major) == 3U;
}

SectionKind section_kind_from_type_name(std::string_view type_name) noexcept
{
    static constexpr SectionKind kinds[] = {
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
        SectionKind::AmdMi450Runtime,
#endif
#ifdef ADDC_HAS_MI450
        SectionKind::AmdMi450Crashdump,
        SectionKind::AmdMi450ScaleUp,
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
    if (matches_amd_mi450_scaleup(header, descriptor))
        return describe_section(SectionKind::AmdMi450ScaleUp);
#endif
    if (matches_ia32x64(header, descriptor))
        return describe_section(SectionKind::Ia32X64);
    if (matches_pcie(header, descriptor))
        return describe_section(SectionKind::Pcie);
#if defined(ADDC_PARSER_HAS_MI300) || defined(ADDC_HAS_MI450)
    if (matches_amd_gpu_runtime(header, descriptor))
    {
        // Both runtime layouts are built with the runtime parser, so the
        // revision alone selects the kind in every product build.
        return describe_section(
            is_mi450_gpu_runtime_revision(descriptor.revision_major)
                ? SectionKind::AmdMi450Runtime
                : SectionKind::AmdMi300Runtime);
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
            return parse_amd_gpu_crashdump(body, descriptor,
                                           AmdGpuLayout::Mi300, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdMi450Crashdump:
#ifdef ADDC_HAS_MI450
            return parse_amd_gpu_crashdump(body, descriptor,
                                           AmdGpuLayout::Mi450, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdMi300Runtime:
#if defined(ADDC_PARSER_HAS_MI300) || defined(ADDC_HAS_MI450)
            return parse_amd_gpu_runtime(body, descriptor,
                                         AmdGpuLayout::Mi300, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdMi450Runtime:
#if defined(ADDC_PARSER_HAS_MI300) || defined(ADDC_HAS_MI450)
            return parse_amd_gpu_runtime(body, descriptor,
                                         AmdGpuLayout::Mi450, context);
#else
            return nullptr;
#endif
        case SectionKind::AmdMi450ScaleUp:
#ifdef ADDC_HAS_MI450
            return parse_amd_mi450_scaleup(body, descriptor, context);
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
