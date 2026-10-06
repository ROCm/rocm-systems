// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/common.hpp"
#include "addc/pipeline/section_data.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace addc::decoder
{

struct DecodedDbgLog
{
    std::string name;
    std::string key_name;
    std::string id;
    nlohmann::ordered_json data_array;
    std::vector<int> afid = {addc::kAfidSentinel};
    bool corrupted = false;
    bool possible_non_mca_fatal = false;
    nlohmann::ordered_json event_report_extra;
    std::optional<addc::pipeline::McaBankDump> mca_bank;
};

using DbgLogDecodeFunction = std::vector<DecodedDbgLog> (*)(
    std::string_view project,
    const std::vector<addc::pipeline::DbgLogGroup>& groups);

[[nodiscard]] std::vector<DecodedDbgLog> decode_base_dbglog(
    std::string_view project,
    const std::vector<addc::pipeline::DbgLogGroup>& groups);

[[nodiscard]] std::vector<DecodedDbgLog> decode_generic_dbglog(
    const std::vector<addc::pipeline::DbgLogGroup>& groups);

[[nodiscard]] nlohmann::ordered_json to_json(const DecodedDbgLog& decoded);

} // namespace addc::decoder
