// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/gpu_mi450_crashdump_decoder.hpp"

#include "addc/boot/decoder.hpp"
#include "addc/boot_mi450/decoder.hpp"
#include "addc/common.hpp"
#include "addc/decoder/dbglog/decoder.hpp"
#include "addc/decoder/wdt/decoder.hpp"
#include "addc/detail/format.hpp"
#include "addc/mca/decoder.hpp"
#include "addc/mca/registers.hpp"
#include "addc/pipeline/section_data.hpp"
#include "gpu_decode_helpers.hpp"

#include <stdexcept>

namespace addc::pipeline
{

namespace
{

[[nodiscard]] std::string mi450ContextTypeName(uint16_t context_type)
{
    switch (context_type)
    {
        case 1U:
            return "MI450 Crashdump";
        case 3U:
            return "MI450 Break Event";
        case 5U:
            return "MI450 DBG Log";
        case 9U:
            return "MI450 Boot Status";
        default:
            return addc::format("MI450 Context Type {}", context_type);
    }
}

[[nodiscard]] nlohmann::ordered_json makeMetadata(
    const AmdGpuMi450CrashdumpSection& s, const DecodeContext& context,
    std::size_t ci, uint16_t context_type, std::string_view event_class,
    std::size_t event_index)
{
    nlohmann::ordered_json m;
    m["section_index"] = context.section_index;
    m["context_structure_index"] = ci;
    m["context_type"] = mi450ContextTypeName(context_type);
    m["event_class"] = event_class;
    m["event_index"] = event_index;
    m["timestamp"] = context.timestamp;
    m["addc_version"] = {{"name", s.platform_name}, {"value", s.addc_version}};
    return m;
}

[[nodiscard]] nlohmann::ordered_json makePlatform(
    const AmdGpuMi450CrashdumpSection& s)
{
    nlohmann::ordered_json p;
    p["fru"] = s.fru;
    p["fru_id"] = s.fru_id;
    p["cpu"] = nullptr;

    nlohmann::ordered_json gpu_obj = nlohmann::ordered_json::object();
    gpu_obj["serial_number"] = nullptr;

    gpu_obj["pcie_device"] = gpu::make_gpu_pcie_device(s);
    if (!s.error_info_structures.empty())
    {
        const auto& ei = s.error_info_structures[0];
        gpu_obj["fru_manufacturer_part_number"] =
            ei.fru_mpn_valid && !ei.fru_manufacturer_part_number.empty()
                ? nlohmann::ordered_json(ei.fru_manufacturer_part_number)
                : nlohmann::ordered_json(nullptr);
        const auto redfish = gpu::make_redfish_event_log(
            ei.redfish_event_log_id_valid, ei.redfish_event_log_id);
        gpu_obj["redfish_event_log_id"] = redfish["id"];
        gpu_obj["redfish_event_log_timestamp"] = redfish["timestamp"];
        gpu_obj["check_info"] = gpu::make_mi450_check_info(ei);
    }
    else
    {
        gpu_obj["fru_manufacturer_part_number"] = nullptr;
        gpu_obj["redfish_event_log_id"] = nullptr;
        gpu_obj["redfish_event_log_timestamp"] = nullptr;
        gpu_obj["check_info"] = nullptr;
    }

    p["gpu"] = std::move(gpu_obj);
    return p;
}

[[nodiscard]] nlohmann::ordered_json makeFirmware(
    const AmdGpuMi450CrashdumpSection& s)
{
    return gpu::make_mi450_firmware_json(s);
}

// Boot Status (type 9) and Break Event (type 3) contexts both carry one
// 8-byte MP0_OutBndMsg value. Base keeps the raw value; Extended decodes the
// message fields. Only Boot Status has a Base AFID.
void emitMi450OutBndMsgEvent(const AmdGpuMi450CrashdumpSection& s,
                             const DecodeContext& context,
                             std::size_t context_index, uint16_t context_type,
                             std::string_view event_class,
                             std::string_view data_key, uint32_t msg_lo,
                             uint32_t msg_hi, bool base_afid,
                             std::vector<PipelineEvent>& events)
{
    PipelineEvent ev;
    ev.metadata = makeMetadata(s, context, context_index, context_type,
                               event_class, events.size());
    ev.platform = makePlatform(s);
    ev.firmware = makeFirmware(s);

    ev.event_report = nullptr;

    const uint64_t msg = (static_cast<uint64_t>(msg_hi) << 32U) |
                         static_cast<uint64_t>(msg_lo);
    ev.data_array[std::string{data_key}] = addc::format("0x{:016x}", msg);

    ev.validation["struct"] = true;
    ev.validation["bank"] = false;
    ev.validation["corrupted"] = nullptr;
    ev.validation["possible_non_mca_fatal"] = nullptr;
    ev.validation["mtl_raw_data"] = false;

    ev.analysis["afid"] = base_afid
                              ? nlohmann::ordered_json::array(
                                    {addc::boot_mi450::decode_afid(msg)})
                              : nlohmann::ordered_json(
                                    addc::afid_list(addc::kAfidSentinel));

    events.push_back(std::move(ev));
}

void emitMi450McaEvent(
    const AmdGpuMi450CrashdumpSection& s, const DecodeContext& context,
    std::size_t context_index, uint16_t context_type, const McaBankDump& bank,
    std::vector<PipelineEvent>& events)
{
    addc::mca::McaRegisters regs{bank.status, bank.ipid,  bank.synd,
                                 bank.addr,   bank.misc0, bank.misc1};
    if (!regs.status.val)
    {
        return;
    }

    const auto decoded = context.decode_mca(
        s.platform, addc::mca::BankCapture{regs, bank.transaddr});
    if (!decoded)
    {
        return;
    }

    PipelineEvent ev;
    ev.metadata = makeMetadata(s, context, context_index, context_type, "mca",
                               events.size());
    ev.platform = makePlatform(s);
    ev.event_report = decoded->event_report;
    ev.data_array = to_data_array(bank);
    ev.firmware = makeFirmware(s);
    ev.validation = decoded->validation;
    ev.analysis["afid"] = decoded->decoded.afid;
    ev.ppr = decoded->ppr;
    events.push_back(std::move(ev));
}

} // namespace

std::vector<PipelineEvent> decode_gpu_mi450_crashdump(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const AmdGpuMi450CrashdumpSection& s, const DecodeContext& context,
    std::string* error_out)
{

    if (context.decode_mca == nullptr)
    {
        if (error_out != nullptr)
        {
            *error_out =
                "GpuMi450CrashdumpDecoder: no MCA operation in DecodeContext";
        }
        return {};
    }

    std::vector<PipelineEvent> events;

    for (std::size_t ci = 0U; ci < s.contexts.size(); ++ci)
    {
        const GpuMi450Context& mc = s.contexts[ci];

        if (mc.crashdump.has_value())
        {
            for (const McaBankDump& bank : mc.crashdump->banks)
            {
                emitMi450McaEvent(s, context, ci, mc.context_type, bank,
                                  events);
            }
        }

        if (mc.df_wdt_addr.has_value())
        {
            const auto& wdt_data = *mc.df_wdt_addr;
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
                    addc::decoder::DecodedWdt wdt_decoded =
                        context.decode_wdt(s.platform, wdt_data, 0U);

                    PipelineEvent ev;
                    ev.metadata = makeMetadata(s, context, ci, mc.context_type,
                                               "wdt_dump", events.size());
                    ev.platform = makePlatform(s);
                    ev.event_report = wdt_decoded.event_report;
                    ev.data_array = wdt_decoded.data_array;
                    ev.firmware = makeFirmware(s);

                    ev.validation["struct"] = true;
                    ev.validation["bank"] = false;
                    ev.validation["corrupted"] = false;
                    ev.validation["possible_non_mca_fatal"] = false;
                    ev.validation["mtl_raw_data"] = false;

                    ev.analysis["afid"] = wdt_decoded.afid;

                    events.push_back(std::move(ev));
                }
            }
        }

        if (!mc.dbg_logs.empty())
        {
            if (context.decode_dbglog != nullptr)
            {
                std::vector<addc::decoder::DecodedDbgLog> decoded_blocks =
                    context.decode_dbglog(s.platform, mc.dbg_logs);

                for (const addc::decoder::DecodedDbgLog& blk : decoded_blocks)
                {
                    if (blk.mca_bank.has_value())
                    {
                        emitMi450McaEvent(s, context, ci, mc.context_type,
                                          *blk.mca_bank, events);
                        continue;
                    }

                    PipelineEvent ev;
                    ev.metadata = makeMetadata(s, context, ci, mc.context_type,
                                               "dbglog_dump", events.size());
                    ev.platform = makePlatform(s);
                    ev.firmware = makeFirmware(s);

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

        if (mc.boot_status.has_value())
        {
            emitMi450OutBndMsgEvent(s, context, ci, mc.context_type,
                                    "boot_status_eam", "boot_msg",
                                    mc.boot_status->boot_msg_lo,
                                    mc.boot_status->boot_msg_hi,
                                    /*base_afid=*/true, events);
        }

        if (mc.break_event.has_value())
        {
            emitMi450OutBndMsgEvent(s, context, ci, mc.context_type,
                                    "break_event_eam", "break_event_msg",
                                    mc.break_event->msg_lo,
                                    mc.break_event->msg_hi,
                                    /*base_afid=*/false, events);
        }
    }

    return events;
}

} // namespace addc::pipeline
