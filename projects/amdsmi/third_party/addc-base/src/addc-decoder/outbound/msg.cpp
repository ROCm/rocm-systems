// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/decoder/outbound/msg.hpp"

#include "addc/common.hpp"
#include "addc/detail/format.hpp"

namespace addc::decoder
{

namespace
{

[[nodiscard]] int bootTimeAfid(uint8_t sub_type_id)
{
    switch (sub_type_id)
    {
        case 0x01:
            return 25188;
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07:
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B:
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F:
        case 0x10:
        case 0x11:
        case 0x12:
            return 25189 + (sub_type_id - 0x04);
        default:
            return addc::kAfidSentinel;
    }
}

[[nodiscard]] std::vector<int> decodeAfid(const OutboundMsg& msg)
{
    if (msg.code_type == 0xA1)
    {
        return addc::afid_list(bootTimeAfid(msg.sub_type_id));
    }

    std::vector<int> afids;
    if (msg.code_type == 0xBF && msg.sub_type_id == 0x01)
    {
        const uint32_t breakevent = msg.raw[3] | (msg.raw[4] << 8U) |
                                    (msg.raw[5] << 16U) | (msg.raw[6] << 24U);
        for (int bit = 0; bit <= 10; ++bit)
        {
            if ((breakevent & (uint32_t{1} << bit)) != 0U)
            {
                afids.push_back(25252 + bit);
            }
        }
    }
    if (afids.empty())
    {
        afids.push_back(addc::kAfidSentinel);
    }
    return afids;
}

[[nodiscard]] std::string codeTypeCategory(uint8_t code_type)
{
    switch (code_type)
    {
        case 0xA1:
        case 0xBF:
            return "error";
        case 0xB0:
        case 0xBA:
            return "status";
        default:
            return "unknown";
    }
}

[[nodiscard]] bool isSubtypeDecodeAvailable(
    uint8_t code_type, uint8_t sub_type_id)
{
    return code_type == 0xBFU && sub_type_id == 0x01U;
}

} // namespace

DecodedOutbound decode_outbound(const std::array<uint32_t, 8>& raw)
{
    const OutboundMsg msg{
        .code_type = static_cast<uint8_t>(raw[0] & 0xFFU),
        .sub_type_id = static_cast<uint8_t>(raw[1] & 0x1FU),
        .raw = raw,
    };
    return DecodedOutbound{
        .code_type_value = msg.code_type,
        .sub_type_value = msg.sub_type_id,
        .code_type_name = {},
        .sub_type_name = {},
        .severity = {},
        .description = {},
        .corrupted = false,
        .runtime_error_bits = {},
        .afid = decodeAfid(msg),
        .format_version = static_cast<uint8_t>((raw[1] >> 5U) & 0x3U),
        .socket_down_config = static_cast<uint8_t>((raw[1] >> 7U) & 0x1U),
        .smn_node_id = static_cast<uint8_t>(raw[5] & 0xFFU),
        .data_array = nullptr,
    };
}

nlohmann::ordered_json to_json(const DecodedOutbound& d)
{
    nlohmann::ordered_json j;

    const std::string code_type_name =
        d.code_type_name.empty() ? "Unknown" : d.code_type_name;
    const std::string sub_type_name =
        d.sub_type_name.empty() ? "Unknown" : d.sub_type_name;
    const std::string category = codeTypeCategory(d.code_type_value);
    const bool decode_available =
        isSubtypeDecodeAvailable(d.code_type_value, d.sub_type_value);

    j["code_type_category"] = category;
    j["code_type"] = {
        {"name", code_type_name},
        {"category", category},
        {"value", d.code_type_value},
    };
    j["sub_type"] = {
        {"name", sub_type_name},
        {"decode", decode_available ? "available" : "unavailable"},
        {"value", d.sub_type_value},
    };
    j["error_severity"] =
        d.severity.empty() ? nlohmann::json(nullptr)
                           : nlohmann::json(d.severity);

    if (!d.runtime_error_bits.empty())
    {
        j["error_description"] = d.runtime_error_bits;
    }
    else if (!d.description.empty())
    {
        j["error_description"] = d.description;
    }
    else
    {
        j["error_description"] = nullptr;
    }

    j["runtime_error_bits"] = d.runtime_error_bits.empty()
                                  ? nlohmann::json(nullptr)
                                  : nlohmann::json(d.runtime_error_bits);

    if (d.code_type_value == 0xBFU && d.sub_type_value == 0x01U)
    {
        j["error_type"] = "ctrlPlaneErr";
    }

    j["format_version"] = d.format_version;
    j["socket_down_config"] = d.socket_down_config;
    j["smn_node_id"] = addc::format("0x{:02x}", d.smn_node_id);
    return j;
}

} // namespace addc::decoder
