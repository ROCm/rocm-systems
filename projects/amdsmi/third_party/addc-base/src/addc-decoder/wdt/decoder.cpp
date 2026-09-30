// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/decoder/wdt/decoder.hpp"

#ifdef ADDC_HAS_VENICE
#include "addc/decoder/wdt/venice.hpp"
#endif
#include "addc/detail/format.hpp"
#include "addc/product.hpp"

#include <cstdint>

namespace addc::decoder
{

namespace
{

[[nodiscard]] uint32_t readU32Le(const uint8_t* p) noexcept
{
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8U) |
           (static_cast<uint32_t>(p[2]) << 16U) |
           (static_cast<uint32_t>(p[3]) << 24U);
}

[[nodiscard]] std::string formatHex32(uint32_t v)
{
    return addc::format("0x{:08x}", v);
}

inline constexpr std::size_t kNumEntries = 8U;
inline constexpr std::size_t kEntryStride = 16U;

} // namespace

DecodedWdt decode_generic_wdt(const std::array<uint8_t, 128>& wdt_data,
                              uint64_t cpuid_eax)
{
    DecodedWdt result;
    result.afid = addc::afid_list(addc::kAfidSentinel);
    result.event_report["severity"] = nullptr;
    result.event_report["description"] = nullptr;
    result.event_report["wdt_entries"] = nullptr;

    auto arr = nlohmann::ordered_json::array();
    for (std::size_t i = 0U; i < kNumEntries; ++i)
    {
        const uint8_t* entry = wdt_data.data() + i * kEntryStride;
        nlohmann::ordered_json e;
        e["ccm_id"] = formatHex32(readU32Le(entry + 0x00U));
        e["logstat"] = formatHex32(readU32Le(entry + 0x04U));
        e["addrlog_lo"] = formatHex32(readU32Le(entry + 0x08U));
        e["addrlog_hi"] = formatHex32(readU32Le(entry + 0x0CU));
        arr.push_back(std::move(e));
    }
    result.data_array = std::move(arr);
    (void)cpuid_eax;
    return result;
}

DecodedWdt decode_base_wdt(std::string_view project,
                           const std::array<uint8_t, 128>& wdt_data,
                           uint64_t cpuid_eax)
{
    const auto* definition = addc::product::find_compiled_product(project);
    if (definition == nullptr)
    {
        return decode_generic_wdt(wdt_data, cpuid_eax);
    }
    switch (definition->identity.id)
    {
#ifdef ADDC_HAS_VENICE
        case addc::product::ProductId::venice:
            return decode_venice_base_wdt(wdt_data, cpuid_eax);
#endif
        case addc::product::ProductId::firerange:
        case addc::product::ProductId::genoa:
        case addc::product::ProductId::turin:
        case addc::product::ProductId::mi450:
            return decode_generic_wdt(wdt_data, cpuid_eax);
        default:
            // Preserve the former explicit "epyc" fallback without retaining
            // a registry entry or a heap decoder.
            return decode_generic_wdt(wdt_data, cpuid_eax);
    }
}

nlohmann::ordered_json to_json(const DecodedWdt& d)
{
    nlohmann::ordered_json j;
    j["event_report"] = d.event_report;
    j["data_array"] = d.data_array;
    j["afid"] = d.afid;
    return j;
}

} // namespace addc::decoder
