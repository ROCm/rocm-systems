// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/common.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace addc::decoder
{

struct DecodedPcie
{
    nlohmann::ordered_json event_report;
    nlohmann::ordered_json data_array;
    nlohmann::ordered_json validation;
    std::vector<int> afid = {addc::kAfidSentinel};
};

[[nodiscard]] DecodedPcie decode_pcie(const nlohmann::json& pcie_body,
                                      uint8_t revision_major);

} // namespace addc::decoder
