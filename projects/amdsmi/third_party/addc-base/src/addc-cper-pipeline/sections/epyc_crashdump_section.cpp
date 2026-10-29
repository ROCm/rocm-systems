// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<AmdEpycCrashdumpSection> parseEpycCrashdumpSection(
    const nlohmann::json& descriptor, const nlohmann::json& section,
    [[maybe_unused]] std::string* error_out)
{
    const nlohmann::json* body = json_obj(section, "AmdEpyc");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AmdEpycCrashdumpSection sec;
    fill_common_metadata(sec, descriptor, "venice");
    sec.descriptor_flags = parse_descriptor_flags(descriptor);

    sec.apic_id_valid = json_bool(*body, "apicIdValid").value_or(false);
    sec.apic_id = json_u64(*body, "apicId").value_or(0U);
    sec.cpuid_valid = json_bool(*body, "cpuidValid").value_or(false);
    if (sec.cpuid_valid)
    {
        if (const nlohmann::json* cpuid = json_obj(*body, "cpuidInfo"))
        {
            sec.cpuid = parse_cpuid(*cpuid);
        }
    }

    const nlohmann::json* contexts = json_arr(*body, "processorContextInfo");
    if (contexts == nullptr)
    {
        return sec;
    }

    sec.contexts.reserve(contexts->size());

    for (const auto& ctx : *contexts)
    {
        EpycProcessorContext ectx;

        const auto blob_opt = json_bytes(ctx, "data");

        if (blob_opt && blob_opt->size() >= kContextHeaderSize)
        {
            const auto& blob = *blob_opt;
            const std::span<const uint8_t> sp{blob};

            ectx.context_type = read_u16_le(sp, 0U);
            ectx.array_size = read_u16_le(sp, 2U);
            ectx.ucode = read_u32_le(sp, 4U);
            ectx.ppin = read_u64_le(sp, 8U);

            if (ectx.context_type == 1U || ectx.context_type == 2U)
            {
                // MCA banks: blob offset 16, up to 32 Ã— 512B.
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
                    ectx.crashdump = std::move(cd);
                }
            }
            else if (ectx.context_type == 3U)
            {
                // Break event: 8 Ã— u32-LE outbound message registers at blob
                // offset 16.
                constexpr std::size_t kOutboundOffset = kContextHeaderSize;
                constexpr std::size_t kOutboundSize = 8U * sizeof(uint32_t);
                if (blob.size() >= kOutboundOffset + kOutboundSize)
                {
                    std::array<uint32_t, 8> regs{};
                    for (std::size_t r = 0U; r < 8U; ++r)
                    {
                        regs[r] = read_u32_le(sp, kOutboundOffset +
                                                      r * sizeof(uint32_t));
                    }
                    ectx.outbound_msg = regs;
                }
            }

            // DfOrigWdtAddr: blob offset 16400, 128B (all context types).
            constexpr std::size_t kWdtOffset = 16400U;
            constexpr std::size_t kWdtSize = 128U;
            if (blob.size() >= kWdtOffset + kWdtSize)
            {
                std::array<uint8_t, kWdtSize> wdt{};
                std::copy_n(blob.data() + kWdtOffset, kWdtSize, wdt.data());
                ectx.df_wdt_addr = wdt;
            }

            // DBG_LOG frames start at blob offset 16912 (all context types).
            // Uncore Trace DRB (block 25) reports no instances in its header
            // but is padded to 2048 records when the CPER is generated.
            constexpr std::size_t kDbgLogOffset = 16912U;
            constexpr uint16_t kUncoreTraceDrbInstances = 2048U;
            for (auto& grp : parse_dbg_log_frames(sp, kDbgLogOffset,
                                                  kUncoreTraceDrbInstances))
            {
                ectx.dbg_logs.push_back(std::move(grp));
            }
        }

        sec.contexts.push_back(std::move(ectx));
    }

    return sec;
}

} // namespace addc::pipeline
