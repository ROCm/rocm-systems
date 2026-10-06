// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "cper-section-ainic.hpp"

#include "../base64.hpp"
#include "addc/detail/ascii.hpp"
#include "addc/detail/format.hpp"

#include <string>

namespace addc::cper::sections
{

namespace
{

// --- AINIC section GUIDs -----------------------------------------------------

//   {3a8f1d2e-7c4b-5e96-af01-23b456c78d9e}
inline constexpr Guid kAinicPlatformContextGuid{
    0x3a8f1d2eU,
    0x7c4bU,
    0x5e96U,
    {0xafU, 0x01U, 0x23U, 0xb4U, 0x56U, 0xc7U, 0x8dU, 0x9eU}};

//   {a0b1c2d3-e4f5-6789-abcd-ef0123456789}
inline constexpr Guid kAinicInterfaceGuid{
    0xa0b1c2d3U,
    0xe4f5U,
    0x6789U,
    {0xabU, 0xcdU, 0xefU, 0x01U, 0x23U, 0x45U, 0x67U, 0x89U}};

//   {b1c2d3e4-f506-789a-bcde-f01234567890}
inline constexpr Guid kAinicPoisonGuid{
    0xb1c2d3e4U,
    0xf506U,
    0x789aU,
    {0xbcU, 0xdeU, 0xf0U, 0x12U, 0x34U, 0x56U, 0x78U, 0x90U}};

//   {c2d3e4f5-0617-89ab-cdef-012345678901}
inline constexpr Guid kAinicEccGuid{
    0xc2d3e4f5U,
    0x0617U,
    0x89abU,
    {0xcdU, 0xefU, 0x01U, 0x23U, 0x45U, 0x67U, 0x89U, 0x01U}};

// --- LE readers --------------------------------------------------------------

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

constexpr uint64_t readU64(std::span<const uint8_t> d, std::size_t o) noexcept
{
    if (o + 8U > d.size())
    {
        return 0U;
    }
    uint64_t v{};
    for (std::size_t i = 0U; i < 8U; ++i)
    {
        v |= static_cast<uint64_t>(d[o + i]) << (8U * i);
    }
    return v;
}

} // anonymous namespace

// ===============================================================================
// AinicPlatformContextParser
// ===============================================================================
//
// Body layout (148 bytes):
//   Offset  Len  Field
//     0      4   ValidationBits
//     4     32   BoardSerial     (valid when bit 0)
//    36     32   FwVersion       (valid when bit 1)
//    68     16   DieId           (valid when bit 2)
//    84     32   Category        (valid when bit 3)
//   116     32   Type            (valid when bit 4)

bool matches_ainic_platform_context(
    const RecordHeader& /*header*/,
    const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kAinicPlatformContextGuid;
}

nlohmann::json parse_ainic_platform_context(
    std::span<const uint8_t> body, const SectionDescriptor& /*desc*/)
{
    if (body.size() < 4U)
    {
        return nullptr;
    }

    const uint32_t vbits = readU32(body, 0U);
    nlohmann::json j;
    j["validationBits"] = vbits;

    // Bit 0: BoardSerial (32 bytes at offset 4)
    if ((vbits & (1U << 0U)) != 0U)
    {
        j["boardSerial"] =
            addc::detail::read_ascii_field(body, 4U, 32U);
    }

    // Bit 1: FwVersion (32 bytes at offset 36)
    if ((vbits & (1U << 1U)) != 0U)
    {
        j["fwVersion"] =
            addc::detail::read_ascii_field(body, 36U, 32U);
    }

    // Bit 2: DieId (16 bytes at offset 68)
    if ((vbits & (1U << 2U)) != 0U)
    {
        // Emit as array of 4 Ã— 32-bit LE words
        nlohmann::json die_arr = nlohmann::json::array();
        for (std::size_t i = 0U; i < 4U; ++i)
        {
            die_arr.push_back(readU32(body, 68U + i * 4U));
        }
        j["dieId"] = die_arr;
    }

    // Bit 3: Category (32 bytes at offset 84)
    if ((vbits & (1U << 3U)) != 0U)
    {
        j["category"] =
            addc::detail::read_ascii_field(body, 84U, 32U);
    }

    // Bit 4: Type (32 bytes at offset 116)
    if ((vbits & (1U << 4U)) != 0U)
    {
        j["type"] = addc::detail::read_ascii_field(body, 116U, 32U);
    }

    return j;
}

// ===============================================================================
// AinicInterfaceParser
// ===============================================================================
//
// Body layout (32 bytes):
//   Offset  Len  Field
//     0      4   ValidationBits
//     4      4   ErrorCode       (valid when bit 0)
//     8      4   PortId          (valid when bit 1)
//    12      4   Status
//    16     16   Reserved

bool matches_ainic_interface(const RecordHeader& /*header*/,
                             const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kAinicInterfaceGuid;
}

nlohmann::json parse_ainic_interface(
    std::span<const uint8_t> body, const SectionDescriptor& /*desc*/)
{
    if (body.size() < 4U)
    {
        return nullptr;
    }

    const uint32_t vbits = readU32(body, 0U);
    const uint32_t err_code = readU32(body, 4U);
    const uint32_t port_id = readU32(body, 8U);
    const uint32_t status = readU32(body, 12U);

    nlohmann::json j;
    j["validationBits"] = vbits;

    if ((vbits & (1U << 0U)) != 0U)
    {
        j["errorCode"] = addc::format("0x{:08x}", err_code);
    }

    if ((vbits & (1U << 1U)) != 0U)
    {
        j["portId"] = port_id;
    }

    j["status"] = addc::format("0x{:08x}", status);

    return j;
}

// ===============================================================================
// AinicPoisonParser
// ===============================================================================
//
// Body layout (32 bytes):
//   Offset  Len  Field
//     0      4   ValidationBits
//     4      8   Address         (valid when bit 0)
//    12      4   SourceId        (valid when bit 1)
//    16      4   PoisonFlags
//    20     12   Reserved

bool matches_ainic_poison(const RecordHeader& /*header*/,
                          const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kAinicPoisonGuid;
}

nlohmann::json parse_ainic_poison(std::span<const uint8_t> body,
                                  const SectionDescriptor& /*desc*/)
{
    if (body.size() < 4U)
    {
        return nullptr;
    }

    const uint32_t vbits = readU32(body, 0U);
    const uint64_t address = readU64(body, 4U);
    const uint32_t source_id = readU32(body, 12U);
    const uint32_t poison_flags = readU32(body, 16U);

    nlohmann::json j;
    j["validationBits"] = vbits;

    if ((vbits & (1U << 0U)) != 0U)
    {
        j["address"] = addc::format("0x{:016x}", address);
    }

    if ((vbits & (1U << 1U)) != 0U)
    {
        j["sourceId"] = source_id;
    }

    j["poisonFlags"] = addc::format("0x{:08x}", poison_flags);

    return j;
}

// ===============================================================================
// AinicEccParser
// ===============================================================================
//
// Body layout (836 bytes):
//   Offset  Len  Field
//     0      4   ValidationBits
//     4      8   Address         (valid when bit 0)
//    12      8   Syndrome        (valid when bit 1)
//    20      4   Corrected       (valid when bit 2)
//    24      4   BitPosition
//    28      8   Reserved
//    36    800   Filler

bool matches_ainic_ecc(const RecordHeader& /*header*/,
                       const SectionDescriptor& desc) noexcept
{
    return desc.section_type == kAinicEccGuid;
}

nlohmann::json parse_ainic_ecc(std::span<const uint8_t> body,
                               const SectionDescriptor& /*desc*/)
{
    if (body.size() < 4U)
    {
        return nullptr;
    }

    const uint32_t vbits = readU32(body, 0U);

    nlohmann::json j;
    j["validationBits"] = vbits;

    // Bit 0: Address (8 bytes at offset 4)
    if ((vbits & (1U << 0U)) != 0U)
    {
        const uint64_t address = readU64(body, 4U);
        j["address"] = addc::format("0x{:016x}", address);
    }

    // Bit 1: Syndrome (8 bytes at offset 12)
    if ((vbits & (1U << 1U)) != 0U)
    {
        const uint64_t syndrome = readU64(body, 12U);
        j["syndrome"] = addc::format("0x{:016x}", syndrome);
    }

    // Bit 2: Corrected (4 bytes at offset 20)
    if ((vbits & (1U << 2U)) != 0U)
    {
        const uint32_t corrected = readU32(body, 20U);
        j["corrected"] = corrected;
    }

    // BitPosition (always present, 4 bytes at offset 24)
    j["bitPosition"] = readU32(body, 24U);

    // Filler (800 bytes at offset 36) - base64 encoded
    if (body.size() > 36U)
    {
        const std::size_t filler_len =
            (body.size() >= 836U) ? 800U : body.size() - 36U;
        j["filler"] = detail::base64_encode(body.subspan(36U, filler_len));
    }

    return j;
}

} // namespace addc::cper::sections
