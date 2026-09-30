// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/common.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace addc::decoder
{

struct DecodedWdt
{
    std::vector<int> afid = {addc::kAfidSentinel};
    nlohmann::ordered_json event_report;
    nlohmann::ordered_json data_array;
};

using WdtDecodeFunction = DecodedWdt (*)(
    std::string_view project, const std::array<uint8_t, 128>& wdt_data,
    uint64_t cpuid_eax);

[[nodiscard]] DecodedWdt decode_base_wdt(
    std::string_view project, const std::array<uint8_t, 128>& wdt_data,
    uint64_t cpuid_eax = 0u);

[[nodiscard]] DecodedWdt decode_generic_wdt(
    const std::array<uint8_t, 128>& wdt_data, uint64_t cpuid_eax = 0u);

[[nodiscard]] nlohmann::ordered_json to_json(const DecodedWdt& decoded);

} // namespace addc::decoder
