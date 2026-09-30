// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc-common/base64.hpp"
#include "addc/cper/record.hpp"

#include <nlohmann/json.hpp>

#include <span>
#include <vector>

namespace addc::cper::detail
{

using ::addc::detail::base64_encode;

[[nodiscard]] inline nlohmann::json encode_binary(
    std::span<const uint8_t> bytes, const ParseContext& context)
{
    if (context.binary_representation == BinaryRepresentation::Native)
    {
        return nlohmann::json::binary(
            std::vector<uint8_t>{bytes.begin(), bytes.end()});
    }
    return base64_encode(bytes);
}

} // namespace addc::cper::detail
