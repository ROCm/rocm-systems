// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/common.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace addc::decoder
{

struct DecodedOutbound
{
    uint8_t code_type_value = 0u;
    uint8_t sub_type_value = 0u;
    std::string code_type_name;
    std::string sub_type_name;
    std::string severity;
    std::string description;
    bool corrupted = false;
    std::vector<std::string> runtime_error_bits;
    std::vector<int> afid = {addc::kAfidSentinel};
    uint8_t format_version = 0u;
    uint8_t socket_down_config = 0u;
    uint8_t smn_node_id = 0u;
    nlohmann::ordered_json data_array;
};

struct OutboundMsg
{
    uint8_t code_type;
    uint8_t sub_type_id;
    std::array<uint32_t, 8> raw;
};

/// Decode the raw fields and AFID knowledge available from public definitions.
/// Optional semantic names and descriptions remain empty when they cannot be
/// derived from those definitions.
[[nodiscard]] DecodedOutbound decode_outbound(
    const std::array<uint32_t, 8>& raw);

[[nodiscard]] nlohmann::ordered_json to_json(const DecodedOutbound& decoded);

} // namespace addc::decoder
