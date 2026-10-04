// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-epyc-pmic.hpp"

#include "addc/detail/format.hpp"

namespace addc::cper::sections
{

namespace
{

// GUID: a7c9d4f2-6e3b-4b91-9a2f-c8715de4136b
inline constexpr Guid kPmicGuid{
    0xa7c9d4f2U,
    0x6e3bU,
    0x4b91U,
    {0x9aU, 0x2fU, 0xc8U, 0x71U, 0x5dU, 0xe4U, 0x13U, 0x6bU}};

constexpr uint32_t readU32(std::span<const uint8_t> d, std::size_t o) noexcept
{
    if (o + 4U > d.size())
    {
        return 0U;
    }
    return static_cast<uint32_t>(d[o]) |
           (static_cast<uint32_t>(d[o + 1U]) << 8U) |
           (static_cast<uint32_t>(d[o + 2U]) << 16U) |
           (static_cast<uint32_t>(d[o + 3U]) << 24U);
}

constexpr uint8_t readU8(std::span<const uint8_t> d, std::size_t o) noexcept
{
    return o < d.size() ? d[o] : 0U;
}

} // namespace

bool matches_amd_epyc_pmic(const RecordHeader& /*header*/,
                           const SectionDescriptor& desc) noexcept
{
    if (!(desc.section_type == kPmicGuid))
    {
        return false;
    }
    const uint32_t program_gen =
        static_cast<uint32_t>(desc.revision_major) & 0xFU;
    const uint32_t ras_gen =
        (static_cast<uint32_t>(desc.revision_major) >> 4U) & 0xFU;
    return program_gen == 0x1U && ras_gen >= 0x3U;
}

nlohmann::json parse_amd_epyc_pmic(
    std::span<const uint8_t> body, const SectionDescriptor& /*desc*/)
{
    nlohmann::json j;

    const uint32_t section_version = readU32(body, 0U);
    const uint32_t section_length = readU32(body, 4U);
    const uint32_t valid_bits = readU32(body, 8U);

    j["section_version"] = addc::format("0x{:08x}", section_version);
    j["section_length"] = addc::format("0x{:08x}", section_length);
    j["valid"] = {
        {"raw", addc::format("0x{:08x}", valid_bits)},
        {"identification_valid", int(bool(valid_bits & 0x1U))},
        {"fault_status_valid", int(bool((valid_bits >> 1U) & 0x1U))},
        {"error_logs_valid", int(bool((valid_bits >> 2U) & 0x1U))},
    };

    j["identification"] = {
        {"socket_id", addc::format("0x{:02x}", readU8(body, 12U))},
        {"channel_id", addc::format("0x{:02x}", readU8(body, 13U))},
        {"dimm_slot", addc::format("0x{:02x}", readU8(body, 14U))},
        {"pmic_index", addc::format("0x{:02x}", readU8(body, 15U))},
    };

    j["fault_status"] = {
        {"Log_0x04", addc::format("0x{:02x}", readU8(body, 16U))},
        {"Log_0x05", addc::format("0x{:02x}", readU8(body, 17U))},
        {"Log_0x06", addc::format("0x{:02x}", readU8(body, 18U))},
        {"Reg_0x08", addc::format("0x{:02x}", readU8(body, 19U))},
        {"Reg_0x09", addc::format("0x{:02x}", readU8(body, 20U))},
        {"Reg_0x0A", addc::format("0x{:02x}", readU8(body, 21U))},
        {"Reg_0x0B", addc::format("0x{:02x}", readU8(body, 22U))},
        {"Reg_0x33", addc::format("0x{:02x}", readU8(body, 23U))},
    };

    nlohmann::json persistent = nlohmann::json::object();
    if (((valid_bits >> 2U) & 0x1U) != 0U)
    {
        for (std::size_t i = 24U; i < body.size(); ++i)
        {
            persistent[addc::format("Log_0x{:03x}", 0x140U + (i - 24U))] =
                addc::format("0x{:02x}", body[i]);
        }
    }
    j["persistent_logs"] = std::move(persistent);

    return j;
}

} // namespace addc::cper::sections
