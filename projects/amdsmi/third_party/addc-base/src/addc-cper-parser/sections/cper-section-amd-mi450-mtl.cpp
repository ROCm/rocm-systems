// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-amd-mi450-mtl.hpp"

#include "../base64.hpp"
#include "../endian.hpp"
#include "addc/detail/format.hpp"

#include <string_view>

namespace addc::cper::sections
{

using addc::cper::read_u16_le;
using addc::cper::read_u32_le;

namespace
{

constexpr uint32_t kMpxIdMask = 0x3FU << 11U;
constexpr uint32_t kMpxSectionBytes = 8U * 1024U;

constexpr uint16_t guidGpuDeviceId(const Guid& guid) noexcept
{
    return static_cast<uint16_t>(guid.d4[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(guid.d4[1]) << 8U);
}

constexpr uint64_t guidDieOrMpId(const Guid& guid) noexcept
{
    uint64_t value = 0U;
    for (std::size_t i = 0U; i < 6U; ++i)
    {
        value |= static_cast<uint64_t>(guid.d4[i + 2U]) << (8U * i);
    }
    return value;
}

[[nodiscard]] bool hasMi450TopologyName(std::string_view fru) noexcept
{
    return fru.find("_MID") != std::string_view::npos ||
           fru.find("_AID") != std::string_view::npos;
}

} // namespace

bool matches_amd_mi450_mtl(
    const RecordHeader& /*header*/,
    const SectionDescriptor& desc) noexcept
{
    if (desc.section_type.d2 != 0x1022U ||
        (desc.section_type.d1 & ~kMpxIdMask) != 0U ||
        desc.section_length != kMpxSectionBytes)
    {
        return false;
    }

    const uint8_t program_gen = desc.revision_major & 0xFU;
    const uint8_t ras_gen = (desc.revision_major >> 4U) & 0xFU;
    if (ras_gen < 0x3U)
    {
        return false;
    }

    // MI450 normally uses program generation 2. Early firmware used the
    // Venice-compatible 0x31 revision and zero GPU DID; its MID/AID topology
    // FRU is the only product discriminator carried by those descriptors.
    return program_gen == 0x2U ||
           (program_gen == 0x1U && hasMi450TopologyName(desc.fru_text));
}

nlohmann::json parse_amd_mi450_mtl(
    std::span<const uint8_t> body, const SectionDescriptor& desc)
{
    ParseContext context;
    return parse_amd_mi450_mtl(body, desc, context);
}

nlohmann::json parse_amd_mi450_mtl(
    std::span<const uint8_t> body, const SectionDescriptor& desc,
    ParseContext& context)
{
    nlohmann::json j;

    const Guid& guid = desc.section_type;
    const uint64_t die_mp_id = guidDieOrMpId(guid);

    j["scalableGuid"] = {
        {"sectionType", addc::format("0x{:08x}", guid.d1)},
        {"mpxId", (guid.d1 & kMpxIdMask) >> 11U},
        {"vendorId", addc::format("0x{:04x}", guid.d2)},
        {"cpuId", addc::format("0x{:04x}", guid.d3)},
        {"gpuDeviceId", addc::format("0x{:04x}", guidGpuDeviceId(guid))},
        {"dieOrMpId",
         {
             {"raw", die_mp_id},
             {"iodIndex", die_mp_id & 0xFU},
             {"ccdIndex", (die_mp_id >> 4U) & 0x1FU},
             {"xcdIndex", (die_mp_id >> 9U) & 0x1FU},
             {"mpIndex", (die_mp_id >> 14U) & 0x3FU},
             {"socketOrCardIndex", (die_mp_id >> 43U) & 0x1FU},
         }},
    };

    j["loggingEnabled"] = addc::format("0x{:08x}", read_u32_le(body, 0U));
    j["tailOffset"] = addc::format("0x{:x}", read_u16_le(body, 4U));
    j["entries"] = read_u16_le(body, 6U);
    j["logVersion"] = addc::format("0x{:08x}", read_u32_le(body, 8U));
    j["usecTimestamp"] =
        addc::format("0x{:08x}", read_u32_le(body, 12U) & 0x0FFFFFFFU);
    j["mtlRawData"] = detail::encode_binary(body, context);

    return j;
}

} // namespace addc::cper::sections
