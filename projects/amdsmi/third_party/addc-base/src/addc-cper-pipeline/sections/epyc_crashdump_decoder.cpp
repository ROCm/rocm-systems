// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/epyc_crashdump_decoder.hpp"

#include "addc/decoder/dbglog/decoder.hpp"
#include "addc/decoder/outbound/msg.hpp"
#include "addc/decoder/wdt/decoder.hpp"
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

[[nodiscard]] std::string_view contextTypeName(uint16_t t) noexcept
{
    switch (t)
    {
        case 1U:
            return "Crashdump";
        case 2U:
            return "Crashdump on MCA initiated Shutdown";
        case 3U:
            return "Break Event";
        default:
            return "Unknown";
    }
}

[[nodiscard]] nlohmann::ordered_json makeMetadata(
    const AmdEpycCrashdumpSection& s, const DecodeContext& context,
    std::size_t ci, uint16_t context_type, std::string_view event_class,
    std::size_t event_index, std::string_view section_severity)
{
    nlohmann::ordered_json m;
    m["section_index"] = context.section_index;
    m["context_structure_index"] = ci;
    m["context_type"] = contextTypeName(context_type);
    m["event_class"] = event_class;
    m["event_index"] = event_index;
    m["timestamp"] = context.timestamp;
    if (!section_severity.empty())
    {
        m["severity"] = section_severity;
    }
    m["addc_version"] = {{"name", s.platform_name}, {"value", s.addc_version}};
    return m;
}

[[nodiscard]] nlohmann::ordered_json makePlatform(
    const AmdEpycCrashdumpSection& s, const EpycProcessorContext& pctx)
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
    cpu["ucode"] = addc::format("0x{:08x}", pctx.ucode);
    cpu["ppin"] = formatHex64(pctx.ppin);

    p["fru"] = s.fru;
    p["fru_id"] = s.fru_id;
    p["cpu"] = std::move(cpu);
    p["gpu"] = nullptr;

    return p;
}

} // namespace

std::vector<PipelineEvent> decode_epyc_crashdump(
    const SectionDescriptor& descriptor,
    const AmdEpycCrashdumpSection& s, const DecodeContext& context,
    std::string* error_out)
{

    if (context.decode_mca == nullptr)
    {
        if (error_out != nullptr)
        {
            *error_out =
                "EpycCrashdumpDecoder: no MCA registry in DecodeContext";
        }
        return {};
    }

    std::vector<PipelineEvent> events;

    for (std::size_t ci = 0U; ci < s.contexts.size(); ++ci)
    {
        const EpycProcessorContext& pctx = s.contexts[ci];

        // -- MCA banks -------------------------------------------------------
        if (pctx.crashdump.has_value())
        {
            for (const McaBankDump& bank : pctx.crashdump->banks)
            {
                addc::mca::McaRegisters regs{bank.status, bank.ipid,
                                             bank.synd,   bank.addr,
                                             bank.misc0,  bank.misc1};
                if (!regs.status.val)
                {
                    continue;
                }

                const auto decoded = context.decode_mca(
                    s.platform, addc::mca::BankCapture{regs, bank.transaddr});
                if (!decoded)
                {
                    continue;
                }

                PipelineEvent ev;

                ev.metadata = makeMetadata(
                    s, context, ci, pctx.context_type, "mca", events.size(),
                    "");
                ev.platform = makePlatform(s, pctx);
                ev.event_report = decoded->event_report;
                if (ev.event_report.is_object())
                {
                    const std::string severity =
                        ev.event_report.value("error_severity", "");
                    if (!severity.empty())
                    {
                        ev.metadata["severity"] = severity;
                    }
                }
                ev.data_array = to_data_array(bank);
                ev.validation = decoded->validation;
                ev.analysis["afid"] = decoded->decoded.afid;
                ev.ppr = decoded->ppr;

                events.push_back(std::move(ev));
            }
        }

        // -- WDT -------------------------------------------------------------
        if (pctx.df_wdt_addr.has_value())
        {
            const auto& wdt_data = *pctx.df_wdt_addr;
            bool all_zero = true;
            for (uint8_t b : wdt_data)
            {
                if (b != 0U)
                {
                    all_zero = false;
                    break;
                }
            }

            if (!all_zero)
            {
                if (context.decode_wdt != nullptr)
                {
                    addc::decoder::DecodedWdt decoded = context.decode_wdt(
                        s.platform, wdt_data,
                        s.cpuid_valid ? s.cpuid.eax : 0U);

                    PipelineEvent ev;

                    ev.metadata =
                        makeMetadata(
                            s, context, ci, pctx.context_type, "wdt_dump",
                            events.size(), descriptor.section_severity);
                    ev.platform = makePlatform(s, pctx);
                    ev.event_report = decoded.event_report;
                    ev.data_array = decoded.data_array;

                    ev.validation["struct"] = true;
                    ev.validation["bank"] = false;
                    ev.validation["corrupted"] = false;
                    ev.validation["possible_non_mca_fatal"] = false;
                    ev.validation["mtl_raw_data"] = false;

                    ev.analysis["afid"] = decoded.afid;

                    events.push_back(std::move(ev));
                }
            }
        }

        // -- DbgLog ----------------------------------------------------------
        if (!pctx.dbg_logs.empty())
        {
            if (context.decode_dbglog != nullptr)
            {
                std::vector<addc::decoder::DecodedDbgLog> decoded_blocks =
                    context.decode_dbglog(s.platform, pctx.dbg_logs);

                for (const addc::decoder::DecodedDbgLog& blk : decoded_blocks)
                {
                    PipelineEvent ev;

                    ev.metadata =
                        makeMetadata(
                            s, context, ci, pctx.context_type, "dbglog_dump",
                            events.size(), descriptor.section_severity);
                    ev.platform = makePlatform(s, pctx);

                    ev.event_report["name"] =
                        blk.key_name.empty()
                            ? nlohmann::ordered_json(nullptr)
                            : nlohmann::ordered_json(blk.key_name);
                    ev.event_report["id"] = blk.id;
                    if (!blk.event_report_extra.is_null() &&
                        blk.event_report_extra.is_object())
                    {
                        for (auto& [k, v] : blk.event_report_extra.items())
                        {
                            ev.event_report[k] = v;
                        }
                    }

                    if (blk.possible_non_mca_fatal &&
                        !descriptor.section_severity.empty())
                    {
                        // For non-MCA fatal dbglog events, use section descriptor
                        // severity as the source of truth.
                        ev.event_report["severity"] =
                            descriptor.section_severity;
                        ev.event_report["error_severity"] =
                            descriptor.section_severity;
                    }

                    ev.data_array = blk.data_array;

                    ev.validation["struct"] = true;
                    ev.validation["bank"] = false;
                    ev.validation["corrupted"] =
                        blk.corrupted ? nlohmann::ordered_json(true)
                                      : nlohmann::ordered_json(nullptr);
                    ev.validation["possible_non_mca_fatal"] =
                        blk.possible_non_mca_fatal
                            ? nlohmann::ordered_json(true)
                            : nlohmann::ordered_json(nullptr);
                    ev.validation["mtl_raw_data"] = false;

                    ev.analysis["afid"] = blk.afid;

                    events.push_back(std::move(ev));
                }
            }
        }

        // -- Outbound message ------------------------------------------------
        if (pctx.outbound_msg.has_value())
        {
            addc::decoder::DecodedOutbound decoded =
                addc::decoder::decode_outbound(*pctx.outbound_msg);

            PipelineEvent ev;

            ev.metadata = makeMetadata(s, context, ci, pctx.context_type,
                                       "outbound_message", events.size(),
                                       descriptor.section_severity);
            ev.platform = makePlatform(s, pctx);
            ev.event_report = addc::decoder::to_json(decoded);
            ev.data_array = decoded.data_array;

            const bool outbound_decoded =
                !decoded.code_type_name.empty() || !decoded.sub_type_name.empty() ||
                !decoded.description.empty() || !decoded.runtime_error_bits.empty();

            ev.validation["struct"] = false;
            ev.validation["decoded"] = outbound_decoded;
            ev.validation["corrupted"] =
                decoded.corrupted ? nlohmann::ordered_json(true)
                                  : nlohmann::ordered_json(nullptr);
            ev.validation["mtl_raw_data"] = false;

            ev.analysis["afid"] = decoded.afid;

            events.push_back(std::move(ev));
        }
    }

    return events;
}

} // namespace addc::pipeline
