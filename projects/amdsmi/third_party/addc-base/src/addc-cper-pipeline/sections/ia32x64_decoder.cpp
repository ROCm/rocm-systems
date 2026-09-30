// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/ia32x64_decoder.hpp"

#include "addc/detail/format.hpp"
#include "addc/mca/decoder.hpp"
#include "addc/mca/registers.hpp"
#include "addc/pipeline/section_data.hpp"

namespace addc::pipeline
{

namespace
{

[[nodiscard]] std::string formatHex64(uint64_t v)
{
    return addc::format("0x{:016x}", v);
}

[[nodiscard]] std::string_view registerContextTypeName(uint16_t t) noexcept
{
    switch (t)
    {
        case 1U:
            return "MSR Registers";
        default:
            return "Unknown";
    }
}

[[nodiscard]] nlohmann::ordered_json makeMetadata(
    const Ia32x64Section& s, const DecodeContext& context,
    std::size_t entry_index, uint16_t register_context_type,
    std::size_t event_index)
{
    nlohmann::ordered_json m;
    m["section_index"] = context.section_index;
    m["context_structure_index"] = entry_index;
    m["context_type"] = registerContextTypeName(register_context_type);
    m["event_class"] = "mca";
    m["event_index"] = event_index;
    m["timestamp"] = context.timestamp;
    m["addc_version"] = {{"name", s.platform_name}, {"value", s.addc_version}};
    return m;
}

[[nodiscard]] nlohmann::ordered_json makePlatform(const Ia32x64Section& s)
{
    nlohmann::ordered_json p;

    nlohmann::ordered_json cpu = nlohmann::ordered_json::object();
    cpu["apic_id"] = s.apic_id_valid
                         ? nlohmann::ordered_json(formatHex64(s.apic_id))
                         : nlohmann::ordered_json(nullptr);
    if (s.cpuid_valid)
    {
        cpu["cpuid"] = nlohmann::ordered_json{
            {"eax", formatHex64(s.cpuid.eax)},
            {"ebx", formatHex64(s.cpuid.ebx)},
            {"ecx", formatHex64(s.cpuid.ecx)},
            {"edx", formatHex64(s.cpuid.edx)},
        };
    }
    else
    {
        cpu["cpuid"] = nullptr;
    }
    cpu["ucode"] = nullptr;
    cpu["ppin"] = nullptr;

    p["fru"] = s.fru;
    p["fru_id"] = s.fru_id;
    p["cpu"] = std::move(cpu);
    p["gpu"] = nullptr;

    return p;
}

} // namespace

std::vector<PipelineEvent> decode_ia32x64(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const Ia32x64Section& s, const DecodeContext& context,
    std::string* error_out)
{

    if (context.decode_mca == nullptr)
    {
        if (error_out != nullptr)
        {
            *error_out = "Ia32x64Decoder: no MCA registry in DecodeContext";
        }
        return {};
    }

    std::vector<PipelineEvent> events;

    for (std::size_t ei = 0U; ei < s.entries.size(); ++ei)
    {
        const Ia32x64Entry& entry = s.entries[ei];

        if (entry.error_type_name != "MS Check Error")
        {
            continue;
        }

        if (!entry.context.bank.has_value())
        {
            continue;
        }

        const McaBankDump& bank = *entry.context.bank;
        addc::mca::McaRegisters regs{bank.status, bank.ipid,  bank.synd,
                                     bank.addr,   bank.misc0, bank.misc1};
        if (!regs.status.val)
        {
            continue;
        }

        // Shared bank -> event assembly (event_report / validation / ppr).
        const auto me =
            context.decode_mca(s.platform, {regs, bank.transaddr});
        if (!me)
        {
            continue;
        }

        PipelineEvent ev;

        ev.metadata = makeMetadata(
            s, context, ei, entry.context.register_context_type, events.size());
        ev.platform = makePlatform(s);
        ev.event_report = me->event_report;
        ev.data_array = to_data_array(bank);
        ev.validation = me->validation;

        // The section report owns placement of the decoded AFID analysis.
        ev.analysis["afid"] = me->decoded.afid;

        ev.ppr = me->ppr;

        events.push_back(std::move(ev));
    }

    return events;
}

} // namespace addc::pipeline
