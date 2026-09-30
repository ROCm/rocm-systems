// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

#include <cstring>

namespace addc::pipeline
{

using namespace detail;

std::optional<AmdGpuMi450CrashdumpSection> parseGpuMi450CrashdumpSection(
    const nlohmann::json& descriptor, const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "AmdMi450Crashdump");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AmdGpuMi450CrashdumpSection sec;
    fill_common_metadata(sec, descriptor, "mi450");
    sec.descriptor_flags = parse_descriptor_flags(descriptor);

    if (const auto* vb = json_obj(*body, "validBits"))
    {
        sec.fw_id_valid = json_bool(*vb, "fwIdValid").value_or(false);
        sec.pcie_devid_valid =
            json_bool(*vb, "pcieDevidValid").value_or(false);
        sec.pldm_bundle_valid =
            json_bool(*vb, "pldmBndlValid").value_or(false);
        if (const auto cnt = json_u32(*vb, "gpuContextInfoCount"))
        {
            sec.gpu_context_info_count = static_cast<uint8_t>(*cnt);
        }
    }

    sec.pcie_dev_id = json_str(*body, "pcieDevId").value_or(std::string{});
    sec.fw_id = json_str(*body, "fwId").value_or(std::string{});

    if (const auto* pb = json_obj(*body, "pldmBundle"))
    {
        sec.pldm_bundle = parse_pldm_bundle(*pb);
    }

    if (const nlohmann::json* eis_arr =
            json_arr(*body, "gpuErrorInfoStructures"))
    {
        for (const auto& eis : *eis_arr)
        {
            GpuErrorInfoStructure info;
            info.fru_mpn_valid =
                json_bool(eis, "fruManufacturerPartNumberValid")
                    .value_or(false);
            info.redfish_event_log_id_valid =
                json_bool(eis, "redfishEventLogIdValid").value_or(false);
            if (info.fru_mpn_valid)
            {
                info.fru_manufacturer_part_number =
                    json_str(eis, "fruManufacturerPartNumber")
                        .value_or(std::string{});
            }
            if (info.redfish_event_log_id_valid)
            {
                info.redfish_event_log_id =
                    json_str(eis, "redfishEventLogId").value_or(std::string{});
            }
            sec.error_info_structures.push_back(std::move(info));
        }
    }

    const nlohmann::json* ctxs = json_arr(*body, "gpuContextStructures");
    if (ctxs == nullptr)
    {
        return sec;
    }

    sec.contexts.reserve(ctxs->size());

    for (const auto& ctx : *ctxs)
    {
        GpuMi450Context mc;

        const auto blob_opt = json_bytes(ctx, "data");

        if (blob_opt && blob_opt->size() >= kContextHeaderSize)
        {
            const auto& blob = *blob_opt;
            const std::span<const uint8_t> sp{blob};

            mc.context_type = read_u16_le(sp, 0U);
            mc.array_size = read_u16_le(sp, 2U);

            if (mc.context_type == 1U)
            {
                // Type 1 (Crashdump): MCA banks + DF/WDT + DBG_LOG
                if (blob.size() > kContextHeaderSize)
                {
                    CrashdumpData cd;
                    cd.banks.reserve(CrashdumpData::kNumBanks);
                    for (std::size_t b = 0U; b < CrashdumpData::kNumBanks; ++b)
                    {
                        cd.banks.push_back(parse_mca_bank(
                            sp, kContextHeaderSize +
                                    b * CrashdumpData::kBankStride));
                    }
                    mc.crashdump = std::move(cd);
                }

                constexpr std::size_t kWdtOffset = 16400U;
                constexpr std::size_t kWdtSize = 128U;
                if (blob.size() >= kWdtOffset + kWdtSize)
                {
                    std::array<uint8_t, kWdtSize> wdt{};
                    std::copy_n(blob.data() + kWdtOffset, kWdtSize, wdt.data());
                    mc.df_wdt_addr = wdt;
                }

                constexpr std::size_t kDbgLogOffset = 16912U;
                if (blob.size() > kDbgLogOffset + 4U)
                {
                    const std::size_t payload_size =
                        blob.size() - kDbgLogOffset;
                    std::size_t off = 0U;
                    while (off + 4U <= payload_size)
                    {
                        const std::size_t base = kDbgLogOffset + off;
                        const uint16_t hdr = read_u16_be(sp, base);
                        const uint16_t bsz = read_u16_le(sp, base + 2U);
                        if (hdr == 0x7ADAU && bsz == 0xBAADU)
                        {
                            break;
                        }
                        const uint8_t block_id =
                            static_cast<uint8_t>(hdr >> 8U);
                        const uint8_t num_instances =
                            static_cast<uint8_t>(hdr & 0xFFU);

                        const uint8_t eff_num =
                            (block_id == 40U && num_instances == 0U)
                                ? 1U
                                : num_instances;
                        const uint16_t eff_bsz =
                            (block_id == 40U && bsz == 0U) ? 256U : bsz;

                        const std::size_t frame_bytes =
                            4U + static_cast<std::size_t>(eff_num) * eff_bsz;
                        if (frame_bytes == 0U ||
                            off + frame_bytes > payload_size)
                        {
                            break;
                        }
                        DbgLogGroup grp;
                        grp.block_id = block_id;
                        grp.block_size = eff_bsz;
                        grp.instances.reserve(eff_num);
                        for (uint8_t ni = 0U; ni < eff_num; ++ni)
                        {
                            const std::size_t inst_off =
                                base + 4U +
                                static_cast<std::size_t>(ni) * eff_bsz;
                            if (inst_off + eff_bsz > blob.size())
                            {
                                break;
                            }
                            grp.instances.push_back(DbgLogInstance{
                                .data =
                                    std::vector<uint8_t>{
                                        blob.begin() +
                                            static_cast<std::ptrdiff_t>(
                                                inst_off),
                                        blob.begin() +
                                            static_cast<std::ptrdiff_t>(
                                                inst_off + eff_bsz)},
                            });
                        }
                        if (!grp.instances.empty())
                        {
                            mc.dbg_logs.push_back(std::move(grp));
                        }
                        off += frame_bytes;
                    }
                }
            }
            else if (mc.context_type == 5U)
            {
                // Type 5 (DBG Log): compact frame at blob offset 16.
                // Header layout: block_id[7:0], instance_count[15:8],
                // per-instance byte length[31:16].
                constexpr std::size_t kDbgLogOffset = kContextHeaderSize;
                if (blob.size() >= kDbgLogOffset + 4U)
                {
                    const uint32_t header = read_u32_le(sp, kDbgLogOffset);
                    const uint8_t block_id =
                        static_cast<uint8_t>(header & 0xFFU);
                    const uint8_t instance_count =
                        static_cast<uint8_t>((header >> 8U) & 0xFFU);
                    const uint16_t block_size =
                        static_cast<uint16_t>(header >> 16U);
                    const std::size_t payload_size = std::min<std::size_t>(
                        blob.size() - kDbgLogOffset,
                        static_cast<std::size_t>(mc.array_size));
                    const std::size_t frame_bytes =
                        4U +
                        static_cast<std::size_t>(instance_count) * block_size;

                    if (instance_count != 0U && block_size != 0U &&
                        frame_bytes <= payload_size)
                    {
                        DbgLogGroup group;
                        group.block_id = block_id;
                        group.block_size = block_size;
                        group.instances.reserve(instance_count);

                        for (uint8_t instance = 0U; instance < instance_count;
                             ++instance)
                        {
                            const std::size_t instance_offset =
                                kDbgLogOffset + 4U +
                                static_cast<std::size_t>(instance) * block_size;
                            if (instance_offset + block_size > blob.size())
                            {
                                break;
                            }
                            group.instances.push_back(DbgLogInstance{
                                .data =
                                    std::vector<uint8_t>{
                                        blob.begin() +
                                            static_cast<std::ptrdiff_t>(
                                                instance_offset),
                                        blob.begin() +
                                            static_cast<std::ptrdiff_t>(
                                                instance_offset + block_size)},
                            });
                        }

                        if (!group.instances.empty())
                        {
                            mc.dbg_logs.push_back(std::move(group));
                        }
                    }
                }
            }
            else if (mc.context_type == 9U)
            {
                // Type 9 (Boot Status): 8-byte boot status buffer at blob
                // offset 16. Layout: EAM0_BootMsg[3:0] (4B) + EAM0_BootMsg[7:4]
                // (4B). No WDT or DBG_LOG.
                constexpr std::size_t kBootStatusOffset = kContextHeaderSize;
                constexpr std::size_t kBootStatusSize = 8U;
                if (blob.size() >= kBootStatusOffset + kBootStatusSize)
                {
                    mc.boot_status = GpuMi450BootStatus{
                        .boot_msg_lo = read_u32_le(sp, kBootStatusOffset),
                        .boot_msg_hi = read_u32_le(sp, kBootStatusOffset + 4U),
                    };
                }
            }
        }

        sec.contexts.push_back(std::move(mc));
    }

    return sec;
}

} // namespace addc::pipeline
