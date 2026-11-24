// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-mi400-scaleup.hpp"

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

inline constexpr Guid kMi400ScaleUpGuid{
    0x5eee25c9U,
    0x83a0U,
    0x234dU,
    {0x82U, 0xbbU, 0x7bU, 0x7fU, 0xc1U, 0x40U, 0xf2U, 0x1fU}};

inline constexpr std::size_t kHeaderSize = 72U;
inline constexpr std::size_t kErrorInfoSize = 64U;
inline constexpr std::size_t kContextHeaderSize = 16U;

nlohmann::json parseErrorInfo(std::span<const uint8_t> d)
{
    constexpr uint64_t kDefinedBits = 0x160U;
    const uint64_t valid_raw = read_u64_le(d, 16U);
    const Guid type =
        Guid::parse(std::span<const uint8_t, 16>{d.data(), 16U});
    return {
        {"errorStructureType", type.to_string()},
        {"validBits",
         {
             {"fruManufacturerPartNumberValid",
              bool((valid_raw >> 5U) & 0x1U)},
             {"redfishEventLogIdValid", bool((valid_raw >> 6U) & 0x1U)},
             {"eventDataFormatVersionValid", bool((valid_raw >> 8U) & 0x1U)},
             {"reservedBits",
              addc::format("0x{:016x}", valid_raw & ~kDefinedBits)},
         }},
        {"eventDataFormatVersion",
         addc::format("0x{:08x}", read_u32_le(d, 24U))},
        {"fruManufacturerPartNumber",
         addc::detail::read_ascii_field(d, 32U, 16U)},
        {"redfishEventLogId", addc::detail::read_ascii_field(d, 56U, 8U)},
    };
}

inline constexpr uint16_t kUaloeEvent = 1U;
inline constexpr uint16_t kUaloeTelemetry = 2U;
inline constexpr std::size_t kEventSize = 8U;
inline constexpr std::size_t kTelemetryHeaderSize = 16U;

nlohmann::json parseUaloeEvent(std::span<const uint8_t> b)
{
    if (b.size() != kEventSize)
    {
        return nullptr;
    }
    const uint32_t word0 = read_u32_le(b, 0U);
    const uint32_t word1 = read_u32_le(b, 4U);
    const uint32_t event_id = word0 & 0xFFU;
    nlohmann::json j = {
        {"eventId", event_id},
        {"eventSeq", (word0 >> 8U) & 0xFFU},
        {"telemetrySeq", (word0 >> 16U) & 0xFFU},
        {"stationId", word0 >> 24U},
    };
    if (event_id == 0U) // STATION_SINGLE
    {
        j["state"] = word1 & 0x3U;
        j["prevState"] = (word1 >> 2U) & 0x3U;
    }
    else if (event_id == 1U) // NETPORT_SINGLE
    {
        j["netportId"] = word1 & 0xFFU;
        j["state"] = (word1 >> 8U) & 0x3U;
        j["prevState"] = (word1 >> 10U) & 0x3U;
    }
    return j;
}

nlohmann::json parseUaloeTelemetry(std::span<const uint8_t> b,
                                   ParseContext& context)
{
    if (b.size() < kTelemetryHeaderSize)
    {
        return nullptr;
    }
    const std::size_t element_count = b[2];
    const std::size_t instance_count = b[3];
    const std::size_t element_length = read_u16_le(b, 4U);
    const std::size_t tag_length = b[10];
    const std::size_t data_size =
        element_length * element_count * instance_count;
    if (b.size() != kTelemetryHeaderSize + tag_length + data_size)
    {
        return nullptr;
    }
    return {
        {"telemetryHeader",
         {
             {"categoryId", read_u16_le(b, 0U)},
             {"elementCount", element_count},
             {"instanceCount", instance_count},
             {"elementLength", element_length},
             {"categoryVersion", read_u32_le(b, 6U)},
             {"tagLength", tag_length},
         }},
        {"categoryTagData",
         detail::encode_binary(b.subspan(kTelemetryHeaderSize, tag_length),
                               context)},
        {"categoryData",
         detail::encode_binary(
             b.subspan(kTelemetryHeaderSize + tag_length, data_size),
             context)},
    };
}

} // anonymous namespace

bool matches_amd_mi400_scaleup(const RecordHeader& /*header*/,
                               const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kMi400ScaleUpGuid;
}

nlohmann::json parse_amd_mi400_scaleup(std::span<const uint8_t> body,
                                       const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_amd_mi400_scaleup(body, desc, context);
}

nlohmann::json parse_amd_mi400_scaleup(std::span<const uint8_t> body,
                                       const SectionDescriptor& /*desc*/,
                                       ParseContext& context)
{
    if (body.size() < kHeaderSize)
    {
        return nullptr;
    }

    const uint64_t valid_raw = read_u64_le(body, 0U);
    const bool err_count_valid = ((valid_raw >> 16U) & 0x1U) != 0U;
    const bool ctx_count_valid = ((valid_raw >> 17U) & 0x1U) != 0U;
    const uint64_t err_count = err_count_valid ? read_u32_le(body, 64U) : 0U;
    const uint64_t ctx_count = ctx_count_valid ? read_u32_le(body, 68U) : 0U;

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
        const auto buffer = body.subspan(data_off, arr_size);
        nlohmann::json ctx = {
            {"registerContextType", static_cast<uint64_t>(ctx_type)},
            {"registerArraySize", static_cast<uint64_t>(arr_size)},
            {"relativeToErr", relative},
            {"timestamp", addc::format("0x{:016x}", timestamp)},
            {"data", detail::encode_binary(buffer, context)},
        };
        // Keep "data": the pipeline reads the raw buffer from it.
        if (ctx_type == kUaloeEvent)
        {
            if (auto event = parseUaloeEvent(buffer); !event.is_null())
            {
                ctx["ualoeEvent"] = std::move(event);
            }
        }
        else if (ctx_type == kUaloeTelemetry)
        {
            if (auto telemetry = parseUaloeTelemetry(buffer, context);
                !telemetry.is_null())
            {
                ctx["ualoeTelemetry"] = std::move(telemetry);
            }
        }
        contexts.push_back(std::move(ctx));
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
             {"pcieDevidValid", bool((valid_raw >> 14U) & 0x1U)},
             {"pldmBndlValid", bool((valid_raw >> 15U) & 0x1U)},
             {"errorInfoCountValid", err_count_valid},
             {"contextCountValid", ctx_count_valid},
         }},
        {"pcieDevId", addc::detail::read_ascii_field(body, 8U, 4U)},
        {"pldmBundle", parse_amd_gpu_pldm_bundle(read_u32_le(body, 12U),
                                                 AmdGpuLayout::Mi400)},
        {"fwId", addc::detail::read_ascii_field(body, 16U, 48U)},
        {"scaleUpErrorInfoCount", err_count},
        {"scaleUpContextInfoCount", ctx_count},
        {"scaleUpErrorInfoStructures", std::move(error_infos)},
        {"scaleUpContextStructures", std::move(contexts)},
    };
}

} // namespace addc::cper::sections
