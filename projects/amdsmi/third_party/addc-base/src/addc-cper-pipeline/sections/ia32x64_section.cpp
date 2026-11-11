// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<Ia32x64Section> parseIa32x64Section(
    const nlohmann::json& descriptor, const nlohmann::json& section,
    std::string* error_out)
{
    (void)error_out;
    const nlohmann::json* body = json_obj(section, "Ia32x64Processor");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    Ia32x64Section sec;
    fill_common_metadata(sec, descriptor, "venice");
    sec.descriptor_flags = parse_descriptor_flags(descriptor);

    sec.apic_id_valid = json_bool(*body, "apicIdValid").value_or(false);
    sec.apic_id = json_u64(*body, "localAPICID").value_or(0U);
    sec.cpuid_valid = json_bool(*body, "cpuidValid").value_or(false);
    if (sec.cpuid_valid)
    {
        if (const nlohmann::json* cpuid = json_obj(*body, "cpuidInfo"))
        {
            sec.cpuid = parse_cpuid(*cpuid);
        }
    }

    const nlohmann::json* error_infos = json_arr(*body, "processorErrorInfo");
    const nlohmann::json* ia_contexts = json_arr(*body, "processorContextInfo");
    if ((error_infos == nullptr) || (ia_contexts == nullptr))
    {
        return sec;
    }

    const std::size_t count =
        std::min(error_infos->size(), ia_contexts->size());
    sec.entries.reserve(count);

    for (std::size_t ei = 0U; ei < count; ++ei)
    {
        const nlohmann::json& err_info = (*error_infos)[ei];
        const nlohmann::json& ia_ctx = (*ia_contexts)[ei];

        Ia32x64Entry entry;

        // -- ProcessorErrorInfo
        // ------------------------------------------------
        if (const nlohmann::json* type = json_obj(err_info, "type"))
        {
            entry.error_type_name = json_str(*type, "name").value_or("Unknown");
        }

        if (entry.error_type_name == "MS Check Error")
        {
            if (const nlohmann::json* ci = json_obj(err_info, "checkInfo"))
            {
                MsCheckInfo ms{};
                if (const nlohmann::json* et = json_obj(*ci, "errorType"))
                {
                    ms.error_type = static_cast<uint8_t>(
                        json_u32(*et, "value").value_or(0U));
                }
                ms.processor_context_corrupt =
                    ci->value("processorContextCorrupt", false);
                ms.uncorrected = ci->value("uncorrected", false);
                ms.precise_ip = ci->value("preciseIP", false);
                ms.restartable_ip = ci->value("restartableIP", false);
                ms.overflow = ci->value("overflow", false);
                entry.ms_check = ms;
            }
        }

        // -- ProcessorContextInfo
        // ----------------------------------------------
        {
            Ia32ProcessorContext ctx{};
            if (const nlohmann::json* rct =
                    json_obj(ia_ctx, "registerContextType"))
            {
                ctx.register_context_type =
                    static_cast<uint16_t>(json_u32(*rct, "value").value_or(0U));
            }
            ctx.register_array_size = static_cast<uint16_t>(
                json_u32(ia_ctx, "registerArraySize").value_or(0U));
            ctx.msr_address = json_u32(ia_ctx, "msrAddress").value_or(0U);
            ctx.mm_register_address =
                json_u64(ia_ctx, "mmRegisterAddress").value_or(0U);

            // register_context_type == 1: MSR registers - parse a single MCA
            // bank. registerArray.data is the raw RegisterArray[] payload
            // (128B, no padding); 16 registers at the same fixed offsets
            // (0x00-0x78) as the EPYC section.
            if (ctx.register_context_type == 1U)
            {
                if (const nlohmann::json* ra =
                        json_obj(ia_ctx, "registerArray"))
                {
                    if (const auto blob_opt = json_bytes(*ra, "data"))
                    {
                        constexpr std::size_t kMinBankBytes =
                            0x80U; // 16 regs Ã— 8B
                        if (blob_opt->size() >= kMinBankBytes)
                        {
                            ctx.bank = parse_mca_bank(
                                std::span<const uint8_t>{*blob_opt}, 0U);
                        }
                    }
                }
            }
            entry.context = ctx;
        }

        sec.entries.push_back(std::move(entry));
    }

    return sec;
}

} // namespace addc::pipeline
