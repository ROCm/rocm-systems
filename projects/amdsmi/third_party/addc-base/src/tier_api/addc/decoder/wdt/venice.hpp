// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/decoder/wdt/decoder.hpp"

namespace addc::decoder
{

[[nodiscard]] DecodedWdt decode_venice_base_wdt(
    const std::array<uint8_t, 128>& wdt_data, uint64_t cpuid_eax = 0u);

} // namespace addc::decoder
