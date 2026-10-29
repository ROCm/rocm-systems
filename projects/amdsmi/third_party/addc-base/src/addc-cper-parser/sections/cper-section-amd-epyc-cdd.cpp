// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-epyc-cdd.hpp"

#include "../base64.hpp"
#include "../endian.hpp"
#include "addc/detail/format.hpp"

namespace addc::cper::sections
{

using addc::cper::read_u16_le;
using addc::cper::read_u32_le;
using addc::cper::read_u64_le;

namespace
{

// GUID: fdbe4adf-86ec-47e3-89be-693061a0f468
inline constexpr Guid kCddGuid{
    0xfdbe4adfU,
    0x86ecU,
    0x47e3U,
    {0x89U, 0xbeU, 0x69U, 0x30U, 0x61U, 0xa0U, 0xf4U, 0x68U}};

// CoreDebugDumpHeader layout (64 bytes total):
//   valid_bits: uint16_t at offset 0
//   apic_id:   uint64_t at offset 2
//   cpuid_eax: uint64_t at offset 10
//   cpuid_ebx: uint64_t at offset 18
//   cpuid_ecx: uint64_t at offset 26
//   cpuid_edx: uint64_t at offset 34
//   (padding to 64 bytes)
//   payload:   bytes from offset 64 onward
constexpr std::size_t kCddHeaderSize = 64U;

} // namespace

bool matches_amd_epyc_cdd(const RecordHeader& /*header*/,
                          const SectionDescriptor& desc) noexcept
{
    if (!(desc.section_type == kCddGuid))
    {
        return false;
    }
    const uint32_t program_gen =
        static_cast<uint32_t>(desc.revision_major) & 0xFU;
    const uint32_t ras_gen =
        (static_cast<uint32_t>(desc.revision_major) >> 4U) & 0xFU;
    return program_gen == 0x1U && ras_gen >= 0x3U;
}

nlohmann::json parse_amd_epyc_cdd(std::span<const uint8_t> body,
                                  const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_amd_epyc_cdd(body, desc, context);
}

nlohmann::json parse_amd_epyc_cdd(
    std::span<const uint8_t> body, const SectionDescriptor& /*desc*/,
    ParseContext& context)
{
    nlohmann::json j;

    const uint16_t valid_bits = read_u16_le(body, 0U);
    j["validBits"] = {
        {"raw", addc::format("0x{:04x}", valid_bits)},
        {"apicIdValid", bool(valid_bits & 0x1U)},
        {"cpuidValid", bool((valid_bits >> 1U) & 0x1U)},
    };
    j["apicId"] = read_u64_le(body, 2U);
    j["cpuidInfo"] = {
        {"eax", addc::format("0x{:016x}", read_u64_le(body, 10U))},
        {"ebx", addc::format("0x{:016x}", read_u64_le(body, 18U))},
        {"ecx", addc::format("0x{:016x}", read_u64_le(body, 26U))},
        {"edx", addc::format("0x{:016x}", read_u64_le(body, 34U))},
    };

    if (body.size() > kCddHeaderSize)
    {
        j["payload"] =
            detail::encode_binary(body.subspan(kCddHeaderSize), context);
    }
    else
    {
        j["payload"] = detail::encode_binary({}, context);
    }

    return j;
}

} // namespace addc::cper::sections
