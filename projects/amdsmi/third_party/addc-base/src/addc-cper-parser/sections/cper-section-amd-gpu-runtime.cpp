// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-gpu-runtime.hpp"

#include "../base64.hpp"
#include "../endian.hpp"
#include "addc/cper/amd_section_ids.hpp"
#include "addc/detail/ascii.hpp"
#include "addc/detail/format.hpp"

#include <string>

namespace addc::cper::sections
{

using addc::cper::read_u16_le;
using addc::cper::read_u32_le;
using addc::cper::read_u64_le;

namespace
{

// --- ValidBits ---------------------------------------------------------------
nlohmann::json parseValidBitsRuntime(uint64_t v)
{
    return {
        {"fwIdValid", bool((v >> 1U) & 0x1U)},
        {"gpuErrorInfoCount", (v >> 2U) & 0x3FU},
        {"gpuContextInfoCount", (v >> 8U) & 0x3FU},
        {"pcieDevidValid", bool((v >> 14U) & 0x1U)},
        {"pldmBndlValid", bool((v >> 15U) & 0x1U)},
    };
}

// --- Context Structure (runtime - with msrAddress + mmRegisterAddress) -------
nlohmann::json parseContextRuntime(std::span<const uint8_t> body,
                                        std::size_t base,
                                        const ParseContext& context)
{
    if (base + 4U > body.size())
    {
        return nullptr;
    }
    const uint16_t ctx_type = read_u16_le(body, base);
    const uint16_t arr_size = read_u16_le(body, base + 2U);
    const uint32_t msr_addr = read_u32_le(body, base + 4U);
    const uint64_t mm_addr = read_u64_le(body, base + 8U);
    const std::size_t data_off = base + 16U;
    const std::size_t avail =
        (data_off < body.size()) ? body.size() - data_off : 0U;
    const std::size_t take = std::min<std::size_t>(arr_size, avail);
    return {
        {"registerContextType", static_cast<uint64_t>(ctx_type)},
        {"registerArraySize", static_cast<uint64_t>(arr_size)},
        {"msrAddress", addc::format("0x{:08x}", msr_addr)},
        {"mmRegisterAddress", addc::format("0x{:016x}", mm_addr)},
        {"registerArray",
         detail::encode_binary(body.subspan(data_off, take), context)},
    };
}

} // anonymous namespace

// --- AMD GPU runtime classification -----------------------------------------

bool matches_amd_gpu_runtime(const RecordHeader& /*header*/,
                             const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kAmdGpuRuntimeGuid;
}

// --- AMD GPU runtime parsing ------------------------------------------------
// Body layout:
//   0:  u64 validBits
//   8:  4-byte pcieDevId  -> null-terminated ASCII
//  12:  u32 pldmBundle
//  16: 48-byte fwId       -> null-terminated ASCII
//  64 + i*64: ErrorInfoStructure[gpuErrorInfoCount]  (each 64 bytes)
//  after: ContextStructure[] (variable), gpuContextInfoCount of them

nlohmann::json parse_amd_gpu_runtime(std::span<const uint8_t> body,
                                     const SectionDescriptor& desc,
                                     AmdGpuLayout layout)
{
    ParseContext context;
    return parse_amd_gpu_runtime(body, desc, layout, context);
}

nlohmann::json parse_amd_gpu_runtime(std::span<const uint8_t> body,
                                     const SectionDescriptor& /*desc*/,
                                     AmdGpuLayout layout, ParseContext& context)
{
    if (body.size() < 8U)
    {
        return nullptr;
    }

    const uint64_t valid_raw = read_u64_le(body, 0U);
    const auto valid_bits_j = parseValidBitsRuntime(valid_raw);
    const uint64_t err_count = (valid_raw >> 2U) & 0x3FU;
    const uint64_t ctx_count = (valid_raw >> 8U) & 0x3FU;

    const std::string pcie_dev_id =
        addc::detail::read_ascii_field(body, 8U, 4U);
    const uint32_t pldm_raw = read_u32_le(body, 12U);
    const std::string fw_id =
        addc::detail::read_ascii_field(body, 16U, 48U);

    // Per spec (Table 2.4): ErrorInfoStructures at offset 64, each 64 bytes,
    // followed by ContextStructures. Counts are per-section.
    // Note: some CPER generators populate the count fields with the total
    // across all sections in the record rather than per-section. When the
    // declared counts exceed the section body size, fall back to 1.
    uint64_t actual_err = err_count;
    uint64_t actual_ctx = ctx_count;
    if (64U + actual_err * 64U + actual_ctx * 16U > body.size())
    {
        actual_err = 1U;
        actual_ctx = 1U;
        if (err_count != actual_err &&
            !context.repair(
                "gpu_error_count_substituted", "gpu_error_info_count",
                err_count, actual_err,
                "declared GPU entry counts exceed the section body"))
        {
            return nullptr;
        }
        if (ctx_count != actual_ctx &&
            !context.repair(
                "gpu_context_count_substituted", "gpu_context_info_count",
                ctx_count, actual_ctx,
                "declared GPU entry counts exceed the section body"))
        {
            return nullptr;
        }
    }

    nlohmann::json error_infos = nlohmann::json::array();
    std::size_t cur = 64U;
    for (uint64_t i = 0U; i < actual_err; ++i)
    {
        if (cur + 64U > body.size())
        {
            break;
        }
        auto entry = parse_amd_gpu_error_info(body, cur, layout);
        if (!entry.is_null())
        {
            error_infos.push_back(std::move(entry));
        }
        cur += 64U;
    }
    if (error_infos.size() != actual_err &&
        !context.repair("gpu_error_count_truncated", "gpu_error_info_count",
                        actual_err, error_infos.size(),
                        "GPU error entries do not fit in the section body"))
    {
        return nullptr;
    }

    nlohmann::json ctx_structs = nlohmann::json::array();
    for (uint64_t i = 0U; i < actual_ctx; ++i)
    {
        if (cur + 4U > body.size())
        {
            break;
        }
        const uint16_t arr_size = read_u16_le(body, cur + 2U);
        const std::size_t struct_sz = 16U + arr_size;
        if (cur + struct_sz > body.size())
        {
            break;
        }
        auto entry = parseContextRuntime(body, cur, context);
        if (!entry.is_null())
        {
            ctx_structs.push_back(std::move(entry));
        }
        cur += struct_sz;
    }
    if (ctx_structs.size() != actual_ctx &&
        !context.repair("gpu_context_count_truncated", "gpu_context_info_count",
                        actual_ctx, ctx_structs.size(),
                        "GPU context entries do not fit in the section body"))
    {
        return nullptr;
    }

    return {
        {"validBits", valid_bits_j},
        {"pcieDevId", pcie_dev_id},
        {"pldmBundle", parse_amd_gpu_pldm_bundle(pldm_raw, layout)},
        {"fwId", fw_id},
        {"gpuErrorInfoStructures", std::move(error_infos)},
        {"gpuContextStructures", std::move(ctx_structs)},
    };
}

} // namespace addc::cper::sections
