// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/decoder/wdt/venice.hpp"

namespace addc::decoder
{

DecodedWdt decode_venice_base_wdt(
    const std::array<uint8_t, 128>& wdt_data, uint64_t cpuid_eax)
{
    auto decoded = decode_generic_wdt(wdt_data, cpuid_eax);
    decoded.afid = addc::afid_list(25316);
    return decoded;
}

} // namespace addc::decoder
