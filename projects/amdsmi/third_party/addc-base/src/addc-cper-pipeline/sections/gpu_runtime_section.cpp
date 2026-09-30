// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

namespace
{

GpuContextType gpuContextTypeFromValue(uint64_t val) noexcept
{
    switch (val)
    {
        case 0U:
            return GpuContextType::Mca;
        case 1U:
            return GpuContextType::Crashdump;
        case 9U:
            return GpuContextType::BootMessage;
        default:
            return GpuContextType::Unknown;
    }
}

std::vector<GpuContextStructure> parseGpuContexts(const nlohmann::json& body)
{
    std::vector<GpuContextStructure> result;
    const nlohmann::json* ctxs = json_arr(body, "gpuContextStructures");
    if (ctxs == nullptr)
    {
        return result;
    }

    result.reserve(ctxs->size());
    for (const auto& ctx : *ctxs)
    {
        GpuContextStructure gs;

        if (ctx.contains("registerContextType"))
        {
            const auto& rct = ctx["registerContextType"];
            uint64_t val = 0xFFFFFFFFU;
            if (rct.is_object())
            {
                val = json_u64(rct, "value").value_or(0xFFFFFFFFU);
            }
            else if (rct.is_number_unsigned())
            {
                val = rct.get<uint64_t>();
            }
            else if (rct.is_number())
            {
                val = static_cast<uint64_t>(rct.get<int64_t>());
            }
            gs.context_type = gpuContextTypeFromValue(val);
        }

        if (auto bytes = json_bytes(ctx, "registerArray"))
        {
            gs.raw_bytes = std::move(*bytes);
        }
        result.push_back(std::move(gs));
    }
    return result;
}

} // namespace

std::optional<AmdGpuRuntimeSection> parseGpuRuntimeSection(
    const nlohmann::json& descriptor, const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "AmdMi300Runtime");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AmdGpuRuntimeSection sec;
    fill_common_metadata(sec, descriptor, "mi300x");
    sec.descriptor_flags = parse_descriptor_flags(descriptor);

    if (const auto* vb = json_obj(*body, "validBits"))
    {
        sec.fw_id_valid = json_bool(*vb, "fwIdValid").value_or(false);
        sec.pcie_devid_valid =
            json_bool(*vb, "pcieDevidValid").value_or(false);
        sec.pldm_bundle_valid =
            json_bool(*vb, "pldmBndlValid").value_or(false);
        if (const auto cnt = json_u32(*vb, "gpuErrorInfoCount"))
        {
            sec.gpu_error_info_count = static_cast<uint8_t>(*cnt);
        }
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

    if (body->contains("gpuErrorInfoStructures") &&
        (*body)["gpuErrorInfoStructures"].is_array())
    {
        for (const auto& eis : (*body)["gpuErrorInfoStructures"])
        {
            sec.error_info_structures.push_back(parse_gpu_error_info(eis));
        }
    }

    sec.contexts = parseGpuContexts(*body);
    return sec;
}

} // namespace addc::pipeline
