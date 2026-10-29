// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-epyc-mtl.hpp"

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

} // namespace

bool matches_amd_epyc_mtl(const RecordHeader& /*header*/,
                          const SectionDescriptor& desc) noexcept
{
    // Match on d2==0x1022 and EPYC Venice or newer (program_gen=1, ras_gen>=3).
    if (desc.section_type.d2 != 0x1022U)
    {
        return false;
    }
    const uint32_t program_gen =
        static_cast<uint32_t>(desc.revision_major) & 0xFU;
    const uint32_t ras_gen =
        (static_cast<uint32_t>(desc.revision_major) >> 4U) & 0xFU;
    return program_gen == 0x1U && ras_gen >= 0x3U;
}

nlohmann::json parse_amd_epyc_mtl(std::span<const uint8_t> body,
                                  const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_amd_epyc_mtl(body, desc, context);
}

nlohmann::json parse_amd_epyc_mtl(
    std::span<const uint8_t> body, const SectionDescriptor& /*desc*/,
    ParseContext& context)
{
    nlohmann::json j;

    j["loggingEnabled"] = addc::format("0x{:08x}", read_u32_le(body, 0U));
    j["tailOffset"] = addc::format("0x{:x}", read_u16_le(body, 4U) & 0xFFFFU);
    j["entries"] = read_u16_le(body, 6U);
    j["logVersion"] = addc::format("0x{:08x}", read_u32_le(body, 8U));
    j["usecTimestamp"] =
        addc::format("0x{:08x}", read_u32_le(body, 12U) & 0x0FFFFFFFU);
    j["mtlRawData"] = detail::encode_binary(body, context);

    return j;
}

} // namespace addc::cper::sections
