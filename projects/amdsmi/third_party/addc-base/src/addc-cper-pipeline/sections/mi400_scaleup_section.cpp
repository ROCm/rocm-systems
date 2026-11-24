// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<AmdMi400ScaleUpSection> parseMi400ScaleUpSection(
    const nlohmann::json& descriptor, const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "AmdMi450ScaleUp");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AmdMi400ScaleUpSection sec;
    fill_common_metadata(sec, descriptor, "mi400");
    // The Scale Up GUID is MI400 Series-only, so the section type owns the
    // platform.
    sec.platform = "mi400";
    sec.platform_name = std::string{platform_display_name(sec.platform)};
    sec.descriptor_flags = parse_descriptor_flags(descriptor);

    fill_gpu_header(sec, *body);

    if (const auto* infos = json_arr(*body, "scaleUpErrorInfoStructures"))
    {
        for (const auto& eis : *infos)
        {
            sec.error_info_structures.push_back(parse_gpu_error_info(eis));
        }
        const nlohmann::json* vb =
            infos->empty() ? nullptr : json_obj(infos->front(), "validBits");
        if (vb != nullptr &&
            json_bool(*vb, "eventDataFormatVersionValid").value_or(false))
        {
            sec.event_data_format_version =
                json_str(infos->front(), "eventDataFormatVersion")
                    .value_or(std::string{});
        }
    }

    const auto* ctxs = json_arr(*body, "scaleUpContextStructures");
    if (ctxs == nullptr)
    {
        return sec;
    }
    for (const auto& ctx : *ctxs)
    {
        ScaleUpContext sc;
        sc.context_type = static_cast<uint16_t>(
            json_u32(ctx, "registerContextType").value_or(0U));
        sc.array_size = static_cast<uint16_t>(
            json_u32(ctx, "registerArraySize").value_or(0U));
        if (const auto it = ctx.find("relativeToErr");
            it != ctx.end() && it->is_number_integer())
        {
            sc.relative_to_err = it->get<int32_t>();
        }
        sc.timestamp = json_str(ctx, "timestamp").value_or(std::string{});

        sc.raw_data =
            json_bytes(ctx, "data").value_or(std::vector<uint8_t>{});
        sec.contexts.push_back(std::move(sc));
    }
    return sec;
}

} // namespace addc::pipeline
