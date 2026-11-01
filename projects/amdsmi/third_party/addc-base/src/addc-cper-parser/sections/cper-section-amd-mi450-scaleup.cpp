// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-mi450-scaleup.hpp"

#include "../base64.hpp"
#include "../endian.hpp"
#include "addc/detail/ascii.hpp"
#include "addc/detail/format.hpp"
#include "cper-section-amd-gpu-common.hpp"

#include <algorithm>
#include <string>

namespace addc::cper::sections
{

using addc::cper::read_u16_le;
using addc::cper::read_u32_le;
using addc::cper::read_u64_le;

namespace
{

inline constexpr Guid kMi450ScaleUpGuid{
    0x5eee25c9U,
    0x83a0U,
    0x234dU,
    {0x82U, 0xbbU, 0x7bU, 0x7fU, 0xc1U, 0x40U, 0xf2U, 0x1fU}};

inline constexpr std::size_t kHeaderSize = 64U;
inline constexpr std::size_t kErrorInfoSize = 64U;
inline constexpr std::size_t kContextHeaderSize = 16U;

// Scale Up Error Info structure: bits 5 and 7 of the valid bits are used;
// offset 24 is reserved (no MS Check structure).
nlohmann::json parseErrorInfo(std::span<const uint8_t> d)
{
    const uint64_t valid_raw = read_u64_le(d, 16U);
    const Guid type =
        Guid::parse(std::span<const uint8_t, 16>{d.data(), 16U});
    return {
        {"errorStructureType", type.to_string()},
        {"validBits",
         {
             {"fruManufacturerPartNumberValid",
              bool((valid_raw >> 5U) & 0x1U)},
             {"redfishEventLogIdValid", bool((valid_raw >> 7U) & 0x1U)},
         }},
        {"fruManufacturerPartNumber",
         addc::detail::read_ascii_field(d, 32U, 16U)},
        {"redfishEventLogId", addc::detail::read_ascii_field(d, 48U, 16U)},
    };
}

} // anonymous namespace

bool matches_amd_mi450_scaleup(const RecordHeader& /*header*/,
                               const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kMi450ScaleUpGuid;
}

nlohmann::json parse_amd_mi450_scaleup(std::span<const uint8_t> body,
                                       const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_amd_mi450_scaleup(body, desc, context);
}

nlohmann::json parse_amd_mi450_scaleup(std::span<const uint8_t> body,
                                       const SectionDescriptor& /*desc*/,
                                       ParseContext& context)
{
    if (body.size() < kHeaderSize)
    {
        return nullptr;
    }

    const uint64_t valid_raw = read_u64_le(body, 0U);
    const uint64_t err_count = (valid_raw >> 2U) & 0x3FU;
    const uint64_t ctx_count = (valid_raw >> 8U) & 0x3FU;

    nlohmann::json error_infos = nlohmann::json::array();
    std::size_t cur = kHeaderSize;
    for (uint64_t i = 0U; i < err_count; ++i)
    {
        if (cur + kErrorInfoSize > body.size())
        {
            break;
        }
        error_infos.push_back(
            parseErrorInfo(body.subspan(cur, kErrorInfoSize)));
        cur += kErrorInfoSize;
    }
    if (error_infos.size() != err_count &&
        !context.repair(
            "scaleup_error_count_truncated", "scaleup_error_info_count",
            err_count, error_infos.size(),
            "declared Scale Up error entries do not fit in the section body"))
    {
        return nullptr;
    }

    // Each context structure is zero-padded to a multiple of 16 bytes.
    nlohmann::json contexts = nlohmann::json::array();
    for (uint64_t i = 0U; i < ctx_count; ++i)
    {
        if (cur + kContextHeaderSize > body.size())
        {
            break;
        }
        const uint16_t ctx_type = read_u16_le(body, cur);
        const uint16_t arr_size = read_u16_le(body, cur + 2U);
        const auto relative = static_cast<int32_t>(read_u32_le(body, cur + 4U));
        const uint64_t timestamp = read_u64_le(body, cur + 8U);
        const std::size_t data_off = cur + kContextHeaderSize;
        if (data_off + arr_size > body.size())
        {
            break;
        }
        contexts.push_back({
            {"registerContextType", static_cast<uint64_t>(ctx_type)},
            {"registerArraySize", static_cast<uint64_t>(arr_size)},
            {"relativeToErr", relative},
            {"timestamp", addc::format("0x{:016x}", timestamp)},
            {"data",
             detail::encode_binary(body.subspan(data_off, arr_size), context)},
        });
        const std::size_t padded = (static_cast<std::size_t>(arr_size) + 15U) &
                                   ~static_cast<std::size_t>(15U);
        cur = data_off + padded;
    }
    if (contexts.size() != ctx_count &&
        !context.repair(
            "scaleup_context_count_truncated", "scaleup_context_info_count",
            ctx_count, contexts.size(),
            "declared Scale Up context entries do not fit in the section body"))
    {
        return nullptr;
    }

    return {
        {"validBits",
         {
             {"fwIdValid", bool((valid_raw >> 1U) & 0x1U)},
             {"scaleUpErrorInfoCount", err_count},
             {"scaleUpContextInfoCount", ctx_count},
             {"pcieDevidValid", bool((valid_raw >> 14U) & 0x1U)},
             {"pldmBndlValid", bool((valid_raw >> 15U) & 0x1U)},
         }},
        {"pcieDevId", addc::detail::read_ascii_field(body, 8U, 4U)},
        {"pldmBundle", parse_amd_gpu_pldm_bundle(read_u32_le(body, 12U),
                                                 AmdGpuLayout::Mi450)},
        {"fwId", addc::detail::read_ascii_field(body, 16U, 48U)},
        {"scaleUpErrorInfoStructures", std::move(error_infos)},
        {"scaleUpContextStructures", std::move(contexts)},
    };
}

} // namespace addc::cper::sections
