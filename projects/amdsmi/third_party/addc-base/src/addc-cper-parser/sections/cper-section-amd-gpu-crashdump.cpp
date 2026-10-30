// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-gpu-crashdump.hpp"

#include "../base64.hpp"
#include "../endian.hpp"
#include "addc/cper/amd_section_ids.hpp"
#include "addc/detail/ascii.hpp"
#include "addc/detail/format.hpp"

#include <algorithm>
#include <string>

namespace addc::cper::sections
{

using addc::cper::read_u16_le;
using addc::cper::read_u32_le;
using addc::cper::read_u64_le;

namespace
{

nlohmann::json parseValidBits(uint64_t v)
{
    nlohmann::json j = nlohmann::json::object();
    j["fwIdValid"] = bool((v >> 1U) & 0x1U);
    j["gpuErrorInfoCount"] = (v >> 2U) & 0x3FU;
    j["gpuContextInfoCount"] = (v >> 8U) & 0x3FU;
    j["pcieDevidValid"] = bool((v >> 14U) & 0x1U);
    j["pldmBndlValid"] = bool((v >> 15U) & 0x1U);
    return j;
}

// MI300 crashdump Error Info. Firmware uses valid bits 5/6 (as in the runtime
// section); the MS Check structure is kept raw.
nlohmann::json parseMi300ErrorInfo(std::span<const uint8_t> body,
                                   std::size_t base)
{
    if (base + 64U > body.size())
    {
        return nullptr;
    }
    const auto d = body.subspan(base, 64U);
    const uint64_t vb = read_u64_le(d, 16U);
    const Guid type = Guid::parse(std::span<const uint8_t, 16>{d.data(), 16U});
    nlohmann::json j = nlohmann::json::object();
    j["errorStructureType"] = type.to_string();
    j["validBits"] = nlohmann::json{
        {"fruManufacturerPartNumberValid", bool((vb >> 5U) & 0x1U)},
        {"redfishEventLogIdValid", bool((vb >> 6U) & 0x1U)},
    };
    j["msCheckStructure"] = read_u64_le(d, 24U);
    j["fruManufacturerPartNumber"] =
        addc::detail::read_ascii_field(d, 32U, 16U);
    j["reserved"] = addc::format("0x{:016x}", read_u64_le(d, 48U));
    j["redfishEventLogId"] = addc::detail::read_ascii_field(d, 56U, 8U);
    return j;
}

// Context structure: 16-byte header (type, array size, reserved) and the
// register array. MI300 reports the array alone; MI450 keeps the header in
// `data` and wraps the type.
nlohmann::json parseContext(std::span<const uint8_t> body, std::size_t base,
                            AmdGpuLayout layout, ParseContext& context)
{
    if (base + 16U > body.size())
    {
        return nullptr;
    }
    const uint16_t ctx_type = read_u16_le(body, base);
    const uint16_t declared_arr_size = read_u16_le(body, base + 2U);
    const std::size_t available_arr_size = body.size() - base - 16U;
    const auto effective_arr_size = static_cast<uint16_t>(
        std::min<std::size_t>(declared_arr_size, available_arr_size));
    if (effective_arr_size != declared_arr_size &&
        !context.repair("register_array_size_substituted",
                        "register_array_size", declared_arr_size,
                        effective_arr_size,
                        "declared register array exceeds the GPU context body"))
    {
        return nullptr;
    }

    nlohmann::json j = nlohmann::json::object();
    if (layout == AmdGpuLayout::Mi450)
    {
        j["registerContextType"] = {
            {"value", static_cast<uint64_t>(ctx_type)},
        };
        j["registerArraySize"] = static_cast<uint64_t>(effective_arr_size);
        j["data"] = detail::encode_binary(
            body.subspan(base, 16U + effective_arr_size), context);
    }
    else
    {
        j["registerContextType"] = static_cast<uint64_t>(ctx_type);
        j["registerArraySize"] = static_cast<uint64_t>(effective_arr_size);
        j["registerArray"] = detail::encode_binary(
            body.subspan(base + 16U, effective_arr_size), context);
    }
    return j;
}

} // anonymous namespace

bool matches_amd_mi300_crashdump(const RecordHeader& /*header*/,
                                 const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kAmdCrashdumpGuid &&
           program_gen(desc.revision_major) == 0x2U &&
           ras_gen(desc.revision_major) == 0x2U;
}

bool matches_amd_mi450_crashdump(const RecordHeader& /*header*/,
                                 const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kAmdCrashdumpGuid &&
           program_gen(desc.revision_major) == 0x2U &&
           ras_gen(desc.revision_major) == 0x3U;
}

// Body layout:
//   0:  u64  validBits
//   8:  4-byte pcieDevId
//  12:  u32  pldmBundle
//  16: 48-byte fwId
//  64: ErrorInfoStructure[err_count] (each 64 bytes)
//  64 + err_count*64: ContextStructure[ctx_count] (16-byte header +
//  registerArray)

nlohmann::json parse_amd_gpu_crashdump(std::span<const uint8_t> body,
                                       const SectionDescriptor& desc,
                                       AmdGpuLayout layout)
{
    ParseContext context;
    return parse_amd_gpu_crashdump(body, desc, layout, context);
}

nlohmann::json parse_amd_gpu_crashdump(std::span<const uint8_t> body,
                                       const SectionDescriptor& /*desc*/,
                                       AmdGpuLayout layout,
                                       ParseContext& context)
{
    if (body.size() < 8U)
    {
        return nullptr;
    }

    const uint64_t valid_raw = read_u64_le(body, 0U);
    const auto valid_bits_j = parseValidBits(valid_raw);
    const uint64_t err_count = (valid_raw >> 2U) & 0x3FU;
    const uint64_t ctx_count = (valid_raw >> 8U) & 0x3FU;

    const std::string pcie_dev_id =
        addc::detail::read_ascii_field(body, 8U, 4U);
    const uint32_t pldm_raw = read_u32_le(body, 12U);
    const std::string fw_id = addc::detail::read_ascii_field(body, 16U, 48U);

    // Error Info structures start at offset 64, 64 bytes each. The first slot
    // is always reserved even when err_count is 0. Iteration is capped by the
    // body size because some CPER generators put the total for the whole
    // record in the count fields.
    nlohmann::json error_infos = nlohmann::json::array();
    std::size_t cur = 64U;
    for (uint64_t i = 0U; i < err_count; ++i)
    {
        if (cur + 64U > body.size())
        {
            break;
        }
        auto entry = layout == AmdGpuLayout::Mi450
                         ? parse_amd_gpu_error_info(body, cur, layout)
                         : parseMi300ErrorInfo(body, cur);
        if (!entry.is_null())
        {
            error_infos.push_back(std::move(entry));
        }
        cur += 64U;
    }
    if (error_infos.size() != err_count &&
        !context.repair(
            "gpu_error_count_truncated", "gpu_error_info_count", err_count,
            error_infos.size(),
            "declared GPU error entries do not fit in the section body"))
    {
        return nullptr;
    }
    // Context structures start after the error info region, at offset 128 or
    // later.
    if (cur < 128U)
    {
        cur = 128U;
    }

    // Some CPER generators set ctx_count to 0 although a context structure is
    // present. Attempt at least one read and stop if its array size is 0.
    nlohmann::json ctx_structs = nlohmann::json::array();
    const uint64_t iters = std::max<uint64_t>(1U, ctx_count);
    for (uint64_t i = 0U; i < iters; ++i)
    {
        if (cur + 16U > body.size())
        {
            break;
        }
        const uint16_t arr_size = read_u16_le(body, cur + 2U);
        if (arr_size == 0U && ctx_count == 0U)
        {
            break;
        }
        auto entry = parseContext(body, cur, layout, context);
        if (entry.is_null())
        {
            break;
        }
        const auto effective_arr_size =
            entry["registerArraySize"].get<std::size_t>();
        ctx_structs.push_back(std::move(entry));
        cur += 16U + effective_arr_size;
    }
    const uint64_t expected_context_count =
        ctx_count == 0U && !ctx_structs.empty() ? 1U : ctx_count;
    if (ctx_count == 0U && expected_context_count == 1U &&
        !context.repair(
            "gpu_context_count_substituted", "gpu_context_info_count", 0U, 1U,
            "a bounded GPU context is present despite a zero declared count"))
    {
        return nullptr;
    }
    if (ctx_structs.size() != expected_context_count &&
        !context.repair(
            "gpu_context_count_truncated", "gpu_context_info_count",
            expected_context_count, ctx_structs.size(),
            "declared GPU context entries do not fit in the section body"))
    {
        return nullptr;
    }

    nlohmann::json result = nlohmann::json::object();
    result["validBits"] = valid_bits_j;
    result["pcieDevId"] = pcie_dev_id;
    result["pldmBundle"] = parse_amd_gpu_pldm_bundle(pldm_raw, layout);
    result["fwId"] = fw_id;
    result["gpuErrorInfoStructures"] = std::move(error_infos);
    result["gpuContextStructures"] = std::move(ctx_structs);
    return result;
}

} // namespace addc::cper::sections
