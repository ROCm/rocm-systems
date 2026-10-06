// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/cdd_decoder.hpp"

#include "addc/common.hpp"
#include "addc/detail/format.hpp"
#include "addc/pipeline/section_data.hpp"

namespace addc::pipeline
{

std::vector<PipelineEvent> decode_cdd(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const AmdEpycCddSection& s, const DecodeContext& context,
    [[maybe_unused]] std::string* error_out)
{

    PipelineEvent ev;
    ev.metadata["section_index"] = context.section_index;
    ev.metadata["context_structure_index"] = 0;
    ev.metadata["context_type"] = "CDD";
    ev.metadata["event_class"] = "core_debug_dump";
    ev.metadata["event_index"] = 0;
    ev.metadata["timestamp"] = context.timestamp;
    ev.metadata["addc_version"] = {{"name", s.platform_name},
                                   {"value", s.addc_version}};

    nlohmann::ordered_json cpu_platform_j;
    cpu_platform_j["apic_id"] = addc::format("0x{:016x}", s.apic_id);
    nlohmann::ordered_json cpuid_platform_j;
    cpuid_platform_j["eax"] = addc::format("0x{:016x}", s.cpuid_eax);
    cpuid_platform_j["ebx"] = addc::format("0x{:016x}", s.cpuid_ebx);
    cpuid_platform_j["ecx"] = addc::format("0x{:016x}", s.cpuid_ecx);
    cpuid_platform_j["edx"] = addc::format("0x{:016x}", s.cpuid_edx);
    cpu_platform_j["cpuid"] = std::move(cpuid_platform_j);
    cpu_platform_j["ucode"] = nullptr;
    cpu_platform_j["ppin"] = nullptr;

    ev.platform["fru"] = s.fru;
    ev.platform["fru_id"] = s.fru_id;
    ev.platform["cpu"] = std::move(cpu_platform_j);
    ev.platform["gpu"] = nullptr;

    // event_report: structured fields matching the common schema.
    // Base does not have enough information to report error_location.
    ev.event_report["error_type"] = "x86 Exception";
    ev.event_report["error_severity"] = descriptor.section_severity.empty()
                                            ? "Fatal"
                                            : descriptor.section_severity;
    ev.event_report["error_description"] =
        "Core Debug Dump captured on fatal x86 exception";
    // validation: from valid_bits so consumers can gate on field presence.
    ev.validation["apic_id_valid"] = bool(s.valid_bits & 0x1U);
    ev.validation["cpuid_valid"] = bool((s.valid_bits >> 1U) & 0x1U);
    ev.validation["corrupted"] = false;

    ev.data_array = nullptr;
    ev.analysis["afid"] = addc::afid_list(addc::kAfidSentinel);

    return {std::move(ev)};
}

} // namespace addc::pipeline
