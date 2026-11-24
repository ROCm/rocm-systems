// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// FRU and record identity values in the CPER header and section descriptor.
// AMD in-band (amdgpu) and out-of-band records can write ASCII text into
// fields that the UEFI specification defines as GUIDs or integers.

#include "addc/detail/format.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace addc::pipeline::detail
{

/// Convert parser GUID text back to the 16 bytes in the record. The text
/// shows d1, d2 and d3 as numbers, but the record holds them little-endian.
[[nodiscard]] inline std::optional<std::array<uint8_t, 16>> guid_text_bytes(
    std::string_view guid)
{
    std::array<uint8_t, 16> bytes{};
    std::size_t nibbles = 0U;
    for (const char c : guid)
    {
        if (c == '-')
            continue;
        const int v = (c >= '0' && c <= '9')   ? c - '0'
                      : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                      : (c >= 'A' && c <= 'F') ? c - 'A' + 10
                                               : -1;
        if (v < 0 || nibbles == 32U)
            return std::nullopt;
        bytes[nibbles / 2U] = static_cast<uint8_t>(
            (bytes[nibbles / 2U] << 4U) | static_cast<uint8_t>(v));
        ++nibbles;
    }
    if (nibbles != 32U)
        return std::nullopt;
    std::reverse(bytes.begin(), bytes.begin() + 4);
    std::reverse(bytes.begin() + 4, bytes.begin() + 6);
    std::reverse(bytes.begin() + 6, bytes.begin() + 8);
    return bytes;
}

/// Return the text if the bytes are printable ASCII followed only by NUL
/// bytes. Return nothing for an empty text or for binary data.
[[nodiscard]] inline std::optional<std::string> ascii_text(
    std::span<const uint8_t> bytes)
{
    const auto end = std::find(bytes.begin(), bytes.end(), uint8_t{0U});
    if (end == bytes.begin() ||
        !std::all_of(bytes.begin(), end,
                     [](uint8_t b) { return b >= 0x20U && b <= 0x7EU; }) ||
        !std::all_of(end, bytes.end(), [](uint8_t b) { return b == 0U; }))
    {
        return std::nullopt;
    }
    return std::string(bytes.begin(), end);
}

/// Show a header GUID field as ASCII text when it holds ASCII text.
/// In-band records write "amdgpu" and "0x1002:0x<device ID>" here.
[[nodiscard]] inline std::string guid_or_ascii(std::string_view guid)
{
    if (const auto bytes = guid_text_bytes(guid))
    {
        if (auto text = ascii_text(*bytes))
            return *text;
    }
    return std::string{guid};
}

/// Show the 8-byte Record ID. In-band records write ASCII text such as
/// "5:0"; out-of-band records write an integer counter.
[[nodiscard]] inline std::string record_id_text(uint64_t record_id)
{
    std::array<uint8_t, 8> bytes{};
    for (std::size_t i = 0U; i < bytes.size(); ++i)
        bytes[i] = static_cast<uint8_t>(record_id >> (8U * i));
    if (auto text = ascii_text(bytes))
        return *text;
    return std::to_string(record_id);
}

/// Remove trailing spaces and NUL bytes from the FRU text. MCA FruText
/// from SYND1/SYND2 ends with a space, for example "Perr: EAM1 ".
[[nodiscard]] inline std::string trim_fru_text(std::string text)
{
    const auto last = text.find_last_not_of(std::string_view{" \0", 2U});
    text.erase(last == std::string::npos ? 0U : last + 1U);
    return text;
}

// --- CUID (Component Unique Identifier) --------------------------------------
//
// The current CUID specification defines one encoding: an RFC 9562 UUIDv8
// whose 16 printed octets U are framed from a 122-bit payload P by a normative
// octet mapping. The amdgpu driver and the ROCm amdcuid library write this
// form.
//
// The older v84 specification gave only a bit-range table, which the current
// specification replaced because it is ambiguous about the bit order. APML and
// PMFW read that table as one 128-bit integer, with the version at integer bits
// [51:48] and the variant at bits [65:64]. Base also decodes this v84 form, for
// compatibility only, and reports it as "v84_compat".

/// The form that a CUID was decoded from.
enum class CuidEncoding
{
    Spec,      ///< The current specification's octet framing.
    V84Compat, ///< The v84 integer reading, for compatibility only.
};

/// The kind of a constructed CUID.
enum class CuidKind
{
    Primary,   ///< Built from the component serial and IDs.
    Derived,   ///< HMAC-SHA256 of a PrimaryID; the fields are hash output.
    Temporary, ///< Built by the library from the machine ID; hash output.
};

/// A CUID in the descriptor FRU ID.
struct Cuid
{
    CuidEncoding encoding = CuidEncoding::Spec;
    CuidKind kind = CuidKind::Primary;
    /// The CUID text in the form that its producer prints: U in printed
    /// order (Spec), or the 128-bit integer most significant byte first
    /// (V84Compat).
    std::string text;
    /// The 122-bit payload: bit b is bit b % 8 of payload[b / 8].
    std::array<uint8_t, 16> payload{};
};

/// The specification's framing in reverse: payload P from printed octets U.
[[nodiscard]] inline std::array<uint8_t, 16> cuid_spec_payload(
    const std::array<uint8_t, 16>& u)
{
    std::array<uint8_t, 16> p{};
    std::copy_n(u.begin(), 6U, p.begin());
    p[6] = static_cast<uint8_t>(((u[6] & 0x0FU) << 4U) | (u[7] >> 4U));
    p[7] =
        static_cast<uint8_t>(((u[7] & 0x0FU) << 4U) | ((u[8] & 0x3CU) >> 2U));
    for (std::size_t i = 8U; i < 15U; ++i)
        p[i] = static_cast<uint8_t>(((u[i] & 0x03U) << 6U) | (u[i + 1U] >> 2U));
    p[15] = static_cast<uint8_t>(u[15] & 0x03U);
    return p;
}

/// The v84 integer reading: the integer is `msb_first`, most significant
/// byte first. Payload bits 0:47 are integer bits 0:47, payload bits 48:59
/// are integer bits 52:63, and payload bits 60:121 are integer bits 66:127.
[[nodiscard]] inline std::optional<std::array<uint8_t, 16>> cuid_v84_payload(
    const std::array<uint8_t, 16>& msb_first)
{
    uint64_t hi = 0U;
    uint64_t lo = 0U;
    for (std::size_t i = 0U; i < 8U; ++i)
    {
        hi = (hi << 8U) | msb_first[i];
        lo = (lo << 8U) | msb_first[i + 8U];
    }
    if (((lo >> 48U) & 0xFU) != 0x8U || (hi & 0x3U) != 0x2U)
        return std::nullopt;
    const uint64_t p_lo =
        (lo & UINT64_C(0xFFFFFFFFFFFF)) | (((lo >> 52U) & 0xFFFU) << 48U) |
        (((hi >> 2U) & 0xFU) << 60U);
    const uint64_t p_hi = hi >> 6U;
    std::array<uint8_t, 16> p{};
    for (std::size_t i = 0U; i < 8U; ++i)
    {
        p[i] = static_cast<uint8_t>(p_lo >> (8U * i));
        p[i + 8U] = static_cast<uint8_t>(p_hi >> (8U * i));
    }
    return p;
}

[[nodiscard]] inline uint16_t cuid_vendor_id(const std::array<uint8_t, 16>& p)
{
    return static_cast<uint16_t>(p[12] | (p[13] << 8U));
}

[[nodiscard]] inline uint8_t cuid_component_type(
    const std::array<uint8_t, 16>& p)
{
    return static_cast<uint8_t>(
        ((p[14] >> 6U) & 0x3U) | ((p[15] & 0x3U) << 2U));
}

/// The specification's kind rules. Bit 117 marks a temporary CUID. A
/// DerivedID has 0 in the reserved bits 64:71 and 118:121 and hash output in
/// bits 72:111; a PrimaryID with those reserved values is a platform whose
/// RevisionID, DeviceID and VendorID are 0.
[[nodiscard]] inline CuidKind cuid_kind(const std::array<uint8_t, 16>& p)
{
    if ((p[14] & 0x20U) != 0U)
        return CuidKind::Temporary;
    const bool reserved_zero = p[8] == 0U && cuid_component_type(p) == 0U;
    const bool ids_zero = std::all_of(p.begin() + 9, p.begin() + 14,
                                      [](uint8_t b) { return b == 0U; });
    return reserved_zero && !ids_zero ? CuidKind::Derived : CuidKind::Primary;
}

[[nodiscard]] inline bool is_amd_vendor(uint16_t vendor)
{
    return vendor == 0x1002U || vendor == 0x1022U;
}

/// Decode a CUID from the FRU ID GUID text of the parser.
///
/// The current specification's form is tried first, on the FRU ID bytes in
/// record order. The v84 form is tried next, for compatibility, with the
/// bytes read as a little-endian and then as a big-endian integer, because
/// the byte order that the AMC uses is not confirmed. A value that passes
/// both checks takes the v84 form only when that form gives an AMD vendor ID
/// and the specification's form does not. ASCII text, such as a serial
/// number, is not a CUID. An adopted firmware UUID (version other than 8) is
/// opaque and is not decoded.
[[nodiscard]] inline std::optional<Cuid> decode_cuid(std::string_view fru_id)
{
    const auto u = guid_text_bytes(fru_id);
    if (!u || ascii_text(*u))
        return std::nullopt;

    std::optional<Cuid> spec;
    if (((*u)[6] >> 4U) == 0x8U && ((*u)[8] >> 6U) == 0x2U)
    {
        const auto p = cuid_spec_payload(*u);
        spec = Cuid{
            .encoding = CuidEncoding::Spec,
            .kind = cuid_kind(p),
            .text = addc::format(
                "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-"
                "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                (*u)[0], (*u)[1], (*u)[2], (*u)[3], (*u)[4], (*u)[5], (*u)[6],
                (*u)[7], (*u)[8], (*u)[9], (*u)[10], (*u)[11], (*u)[12],
                (*u)[13], (*u)[14], (*u)[15]),
            .payload = p,
        };
    }

    std::optional<Cuid> v84;
    std::array<uint8_t, 16> reversed{};
    std::reverse_copy(u->begin(), u->end(), reversed.begin());
    for (const auto& msb_first : {reversed, *u})
    {
        if (const auto p = cuid_v84_payload(msb_first))
        {
            v84 = Cuid{
                .encoding = CuidEncoding::V84Compat,
                .kind = cuid_kind(*p),
                .text = addc::format(
                    "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-"
                    "{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                    msb_first[0], msb_first[1], msb_first[2], msb_first[3],
                    msb_first[4], msb_first[5], msb_first[6], msb_first[7],
                    msb_first[8], msb_first[9], msb_first[10], msb_first[11],
                    msb_first[12], msb_first[13], msb_first[14], msb_first[15]),
                .payload = *p,
            };
            break;
        }
    }

    if (spec && v84 && is_amd_vendor(cuid_vendor_id(v84->payload)) &&
        !is_amd_vendor(cuid_vendor_id(spec->payload)))
    {
        return v84;
    }
    return spec ? spec : v84;
}

/// Component type names from the current CUID specification.
[[nodiscard]] inline std::string_view cuid_component_type_name(uint8_t type)
{
    switch (type)
    {
        case 0x0U:
            return "Platform";
        case 0x1U:
            return "CPU";
        case 0x2U:
            return "GPU";
        case 0x3U:
            return "NIC";
        case 0x4U:
            return "NPU";
        case 0x5U:
            return "Storage";
        case 0x6U:
            return "Memory";
        case 0x7U:
            return "Generic PCIe";
        case 0x8U:
            return "Generic";
        case 0x9U:
            return "Rack tray";
        case 0xAU:
            return "Rack";
        case 0xFU:
            return "Other";
        default:
            return "Reserved";
    }
}

/// The "cuid" JSON object. Only a PrimaryID has fields; a DerivedID or a
/// temporary CUID holds hash output.
[[nodiscard]] inline nlohmann::ordered_json cuid_to_json(const Cuid& cuid)
{
    nlohmann::ordered_json j;
    j["encoding"] = cuid.encoding == CuidEncoding::Spec ? "spec" : "v84_compat";
    j["kind"] = cuid.kind == CuidKind::Primary   ? "primary"
                : cuid.kind == CuidKind::Derived ? "derived"
                                                 : "temporary";
    if (cuid.kind != CuidKind::Primary)
        return j;
    const auto& p = cuid.payload;
    uint64_t serial = 0U;
    for (std::size_t i = 0U; i < 8U; ++i)
        serial |= static_cast<uint64_t>(p[i]) << (8U * i);
    const uint8_t type = cuid_component_type(p);
    j["serial"] = addc::format("0x{:016x}", serial);
    j["unit_id"] = static_cast<unsigned>(p[8] | ((p[14] & 0x1FU) << 8U));
    j["revision_id"] = addc::format("0x{:02x}", p[9]);
    j["device_id"] = addc::format("0x{:04x}", p[10] | (p[11] << 8U));
    j["vendor_id"] = addc::format("0x{:04x}", cuid_vendor_id(p));
    j["component_type"] = addc::format("0x{:x}", type);
    j["component_type_name"] = cuid_component_type_name(type);
    return j;
}

/// Set "fru_id" and "cuid" in a platform object.
/// - No valid FRU ID: both are null.
/// - A CUID: "fru_id" shows the CUID text and "cuid" holds the decoded
///   fields.
/// - ASCII text, such as a serial number: "fru_id" shows the text and
///   "cuid" is null.
/// - Other content: "fru_id" shows the GUID text and "cuid" is null.
inline std::optional<Cuid> set_fru_id_and_cuid(nlohmann::ordered_json& p,
                                               std::string_view fru_id)
{
    auto cuid = decode_cuid(fru_id);
    p["fru_id"] = cuid ? nlohmann::ordered_json(cuid->text)
                  : fru_id.empty()
                      ? nlohmann::ordered_json(nullptr)
                      : nlohmann::ordered_json(guid_or_ascii(fru_id));
    p["cuid"] = cuid ? cuid_to_json(*cuid) : nlohmann::ordered_json(nullptr);
    return cuid;
}

} // namespace addc::pipeline::detail
