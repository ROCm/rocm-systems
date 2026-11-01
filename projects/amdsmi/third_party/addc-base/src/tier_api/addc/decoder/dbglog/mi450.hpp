// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/decoder/dbglog/epyc.hpp"

#include <optional>
#include <span>

namespace addc::decoder
{

[[nodiscard]] std::optional<addc::pipeline::McaBankDump>
    parse_mi450_register_array(std::span<const uint8_t> bytes);

[[nodiscard]] std::vector<DecodedDbgLog> decode_mi450_dbglog(
    const std::vector<addc::pipeline::DbgLogGroup>& groups);

} // namespace addc::decoder
