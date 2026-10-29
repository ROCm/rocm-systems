// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/decode_context.hpp"
#include "addc/pipeline/section_data.hpp"
#include "gpu_decode_helpers.hpp"
#include "section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<AmdGpuMi450RuntimeSection> parseGpuMi450RuntimeSection(
    const nlohmann::json& descriptor, const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "AmdMi300Runtime");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AmdGpuMi450RuntimeSection sec;
    fill_common_metadata(sec, descriptor, "mi450");
    sec.descriptor_flags = parse_descriptor_flags(descriptor);

    fill_gpu_header(sec, *body);

    if (body->contains("gpuErrorInfoStructures") &&
        (*body)["gpuErrorInfoStructures"].is_array())
    {
        for (const auto& eis : (*body)["gpuErrorInfoStructures"])
        {
            sec.error_info_structures.push_back(parse_gpu_error_info(eis));
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
        GpuMi450RuntimeContext mc;

        if (ctx.contains("registerContextType"))
        {
            const auto& rct = ctx["registerContextType"];
            uint64_t val = 0xFFFFU;
            if (rct.is_number_unsigned())
            {
                val = rct.get<uint64_t>();
            }
            else if (rct.is_number())
            {
                val = static_cast<uint64_t>(rct.get<int64_t>());
            }
            mc.context_type = static_cast<uint16_t>(val);
        }
        if (const auto v = json_u64(ctx, "registerArraySize"))
        {
            mc.array_size = static_cast<uint16_t>(*v);
        }

        if (auto bytes = json_bytes(ctx, "registerArray"))
        {
            if (auto bank = gpu::decode_runtime_mca_bank(*bytes))
            {
                mc.bank = bank;
            }
        }

        sec.contexts.push_back(mc);
    }

    return sec;
}

} // namespace addc::pipeline
