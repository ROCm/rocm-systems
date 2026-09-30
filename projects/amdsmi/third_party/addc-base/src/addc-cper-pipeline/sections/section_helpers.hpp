// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// Helpers shared across section parser files.

#include "addc-common/base64.hpp"
#include "addc/pipeline/detect.hpp"
#include "addc/pipeline/section_data.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace addc::pipeline::detail
{

static constexpr std::string_view kZeroGuid =
    "00000000-0000-0000-0000-000000000000";

// --- JSON helpers
// -------------------------------------------------------------

[[nodiscard]] inline const nlohmann::json* json_obj(const nlohmann::json& obj,
                                                    std::string_view key)
{
    if (!obj.is_object())
        return nullptr;
    const auto it = obj.find(std::string{key});
    if (it == obj.end() || !it->is_object())
        return nullptr;
    return &(*it);
}

[[nodiscard]] inline const nlohmann::json* json_arr(const nlohmann::json& obj,
                                                    std::string_view key)
{
    if (!obj.is_object())
        return nullptr;
    const auto it = obj.find(std::string{key});
    if (it == obj.end() || !it->is_array())
        return nullptr;
    return &(*it);
}

[[nodiscard]] inline std::optional<std::string> json_str(
    const nlohmann::json& obj, std::string_view key)
{
    if (!obj.is_object())
        return std::nullopt;
    const auto it = obj.find(std::string{key});
    if (it == obj.end() || !it->is_string())
        return std::nullopt;
    return it->get<std::string>();
}

[[nodiscard]] inline std::optional<uint64_t> json_u64(const nlohmann::json& obj,
                                                      std::string_view key)
{
    if (!obj.is_object())
        return std::nullopt;
    const auto it = obj.find(std::string{key});
    if (it == obj.end())
        return std::nullopt;
    if (it->is_number_unsigned())
        return it->get<uint64_t>();
    if (it->is_number_integer())
    {
        auto v = it->get<int64_t>();
        if (v >= 0)
            return static_cast<uint64_t>(v);
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<uint32_t> json_u32(const nlohmann::json& obj,
                                                      std::string_view key)
{
    const auto v = json_u64(obj, key);
    if (!v || *v > static_cast<uint64_t>(UINT32_MAX))
        return std::nullopt;
    return static_cast<uint32_t>(*v);
}

[[nodiscard]] inline std::optional<bool> json_bool(const nlohmann::json& obj,
                                                   std::string_view key)
{
    if (!obj.is_object())
        return std::nullopt;
    const auto it = obj.find(std::string{key});
    if (it == obj.end() || !it->is_boolean())
        return std::nullopt;
    return it->get<bool>();
}

/// Read a binary parser field from either the public Base64 JSON form or the
/// native-byte form used only by the typed analysis path.
[[nodiscard]] inline std::optional<std::vector<uint8_t>> json_bytes(
    const nlohmann::json& obj, std::string_view key)
{
    if (!obj.is_object())
        return std::nullopt;
    const auto it = obj.find(std::string{key});
    if (it == obj.end())
        return std::nullopt;
    if (it->is_binary())
    {
        const auto& bytes = it->get_binary();
        return std::vector<uint8_t>{bytes.begin(), bytes.end()};
    }
    if (it->is_string())
        return ::addc::detail::base64_decode(it->get_ref<const std::string&>());
    return std::nullopt;
}

// --- Descriptor helpers
// -------------------------------------------------------

/// Extract and normalize the section GUID from a CPER IR descriptor.
[[nodiscard]] inline std::string extract_guid(const nlohmann::json& descriptor)
{
    const nlohmann::json* st = json_obj(descriptor, "sectionType");
    if (!st)
        return {};
    std::string guid = json_str(*st, "data").value_or(std::string{});
    std::transform(guid.begin(), guid.end(), guid.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return guid;
}

[[nodiscard]] inline SectionDescriptorFlags parse_descriptor_flags(
    const nlohmann::json& descriptor)
{
    SectionDescriptorFlags flags;
    const nlohmann::json* f = json_obj(descriptor, "flags");
    if (!f)
        return flags;
    if (auto v = json_bool(*f, "primary"))
        flags.primary = *v;
    if (auto v = json_bool(*f, "containmentWarning"))
        flags.containment_warning = *v;
    if (auto v = json_bool(*f, "reset"))
        flags.reset = *v;
    if (auto v = json_bool(*f, "errorThresholdExceeded"))
        flags.error_threshold_exceeded = *v;
    if (auto v = json_bool(*f, "resourceNotAccessible"))
        flags.resource_not_accessible = *v;
    if (auto v = json_bool(*f, "latentError"))
        flags.latent_error = *v;
    if (auto v = json_bool(*f, "propagated"))
        flags.propagated = *v;
    if (auto v = json_bool(*f, "overflow"))
        flags.overflow = *v;
    return flags;
}

/// Build the addc_version string from descriptor revision fields.
[[nodiscard]] inline std::string addc_version_string(
    const nlohmann::json& descriptor)
{
    const nlohmann::json* rev = json_obj(descriptor, "revision");
    if (!rev)
        return "0.0.0";
    const auto major = json_u32(*rev, "major");
    const auto minor = json_u32(*rev, "minor");
    if (!major || !minor)
        return "0.0.0";
    return std::to_string(*major >> 4u) + "." + std::to_string(*major & 0xFu) +
           "." + std::to_string(*minor);
}

/// Detect platform from descriptor revision, falling back to `fallback`.
[[nodiscard]] inline std::string detect_platform_from_descriptor(
    const nlohmann::json& descriptor, std::string_view fallback = "venice")
{
    const nlohmann::json* rev = json_obj(descriptor, "revision");
    if (!rev)
        return std::string{fallback};
    const auto major = json_u32(*rev, "major");
    if (!major)
        return std::string{fallback};
    auto det = addc::pipeline::detect_platform(*major);
    return det.value_or(std::string{fallback});
}

/// Fill common metadata fields on any section.
template <typename Section>
void fill_common_metadata(Section& sec, const nlohmann::json& descriptor,
                          std::string_view platform_fallback = "venice")
{
    sec.platform =
        detect_platform_from_descriptor(descriptor, platform_fallback);
    sec.platform_name = std::string{platform_display_name(sec.platform)};
    sec.fru = json_str(descriptor, "fruText").value_or(std::string{});
    sec.fru_id = json_str(descriptor, "fruID").value_or(std::string{kZeroGuid});
    sec.addc_version = addc_version_string(descriptor);
}

/// Fill CpuidInfo from a JSON cpuidInfo object.
[[nodiscard]] inline CpuidInfo parse_cpuid(const nlohmann::json& cpuid)
{
    return CpuidInfo{
        .eax = json_u64(cpuid, "eax").value_or(0u),
        .ebx = json_u64(cpuid, "ebx").value_or(0u),
        .ecx = json_u64(cpuid, "ecx").value_or(0u),
        .edx = json_u64(cpuid, "edx").value_or(0u),
    };
}

/// Parse PLDM bundle from a JSON pldmBundle object.
[[nodiscard]] inline GpuPldmBundle parse_pldm_bundle(const nlohmann::json& pb)
{
    GpuPldmBundle b;
    if (const auto v = json_u32(pb, "minor"))
        b.minor = static_cast<uint8_t>(*v);
    if (const auto v = json_u32(pb, "major"))
        b.major = static_cast<uint8_t>(*v);
    if (const auto v = json_u32(pb, "year"))
        b.year = static_cast<uint8_t>(*v);
    if (const auto v = json_u32(pb, "productionDebug"))
        b.production_debug = static_cast<uint8_t>(*v);
    return b;
}

// --- Little-endian binary readers
// ---------------------------------------------

[[nodiscard]] inline uint16_t read_u16_be(std::span<const uint8_t> data,
                                          std::size_t offset) noexcept
{
    if (offset + 2u > data.size())
        return 0u;
    return (static_cast<uint16_t>(data[offset]) << 8u) |
           static_cast<uint16_t>(data[offset + 1u]);
}

[[nodiscard]] inline uint16_t read_u16_le(std::span<const uint8_t> data,
                                          std::size_t offset) noexcept
{
    if (offset + 2u > data.size())
        return 0u;
    return static_cast<uint16_t>(data[offset]) |
           (static_cast<uint16_t>(data[offset + 1u]) << 8u);
}

[[nodiscard]] inline uint32_t read_u32_le(std::span<const uint8_t> data,
                                          std::size_t offset) noexcept
{
    if (offset + 4u > data.size())
        return 0u;
    return static_cast<uint32_t>(data[offset]) |
           (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
           (static_cast<uint32_t>(data[offset + 2u]) << 16u) |
           (static_cast<uint32_t>(data[offset + 3u]) << 24u);
}

[[nodiscard]] inline uint64_t read_u64_le(std::span<const uint8_t> data,
                                          std::size_t offset) noexcept
{
    if (offset + 8u > data.size())
        return 0u;
    return static_cast<uint64_t>(data[offset]) |
           (static_cast<uint64_t>(data[offset + 1u]) << 8u) |
           (static_cast<uint64_t>(data[offset + 2u]) << 16u) |
           (static_cast<uint64_t>(data[offset + 3u]) << 24u) |
           (static_cast<uint64_t>(data[offset + 4u]) << 32u) |
           (static_cast<uint64_t>(data[offset + 5u]) << 40u) |
           (static_cast<uint64_t>(data[offset + 6u]) << 48u) |
           (static_cast<uint64_t>(data[offset + 7u]) << 56u);
}

// --- ProcessorContextInfo binary constants
// ------------------------------------

/// Size of the ProcessorContextInfo binary header (before the payload region).
/// Layout: context_type(2) + array_size(2) + ucode(4) + ppin(8) = 16 bytes.
inline constexpr std::size_t kContextHeaderSize = 16u;

// --- MCA bank parser
// ----------------------------------------------------------

/// Parse one McaBankDump from a 512-byte slot at `offset` within `data`.
/// Out-of-bounds reads return 0.
[[nodiscard]] inline McaBankDump parse_mca_bank(std::span<const uint8_t> data,
                                                std::size_t offset) noexcept
{
    return McaBankDump{
        .ctl = read_u64_le(data, offset + 0x00u),
        .status = read_u64_le(data, offset + 0x08u),
        .addr = read_u64_le(data, offset + 0x10u),
        .misc0 = read_u64_le(data, offset + 0x18u),
        .config = read_u64_le(data, offset + 0x20u),
        .ipid = read_u64_le(data, offset + 0x28u),
        .synd = read_u64_le(data, offset + 0x30u),
        .destat = read_u64_le(data, offset + 0x38u),
        .deaddr = read_u64_le(data, offset + 0x40u),
        .misc1 = read_u64_le(data, offset + 0x48u),
        .synd1 = read_u64_le(data, offset + 0x50u),
        .synd2 = read_u64_le(data, offset + 0x58u),
        .ctl_mask = read_u64_le(data, offset + 0x60u),
        .transsynd = read_u64_le(data, offset + 0x68u),
        .transaddr = read_u64_le(data, offset + 0x70u),
        .transstat = read_u64_le(data, offset + 0x78u),
    };
}

/// Parse a GpuErrorInfoStructure from a JSON error info entry.
[[nodiscard]] inline GpuErrorInfoStructure parse_gpu_error_info(
    const nlohmann::json& eis)
{
    GpuErrorInfoStructure info;
    info.error_structure_type =
        json_str(eis, "errorStructureType").value_or(std::string{});

    if (const auto* vb = json_obj(eis, "validBits"))
    {
        info.check_info_valid =
            json_bool(*vb, "checkInfoValid").value_or(false);
        info.target_addr_id_valid =
            json_bool(*vb, "targetAddressIdentifierValid").value_or(false);
        info.requester_id_valid =
            json_bool(*vb, "requesterIdentifierValid").value_or(false);
        info.responder_id_valid =
            json_bool(*vb, "responderIdentifierValid").value_or(false);
        info.instruction_pointer_valid =
            json_bool(*vb, "instructionPointerValid").value_or(false);
        info.fru_mpn_valid =
            json_bool(*vb, "fruManufacturerPartNumberValid").value_or(false);
        info.redfish_event_log_id_valid =
            json_bool(*vb, "redfishEventLogIdValid").value_or(false);
    }

    if (const auto* mcs = json_obj(eis, "msCheckStructure"))
    {
        if (const auto v = json_u32(*mcs, "errorType"))
            info.error_type = static_cast<uint8_t>(*v);
        auto as_bool = [&](std::string_view key) -> bool {
            if (const auto b = json_bool(*mcs, key))
                return *b;
            if (const auto v = json_u32(*mcs, key))
                return *v != 0u;
            return false;
        };
        info.pcc = as_bool("pcc");
        info.uncorrected = as_bool("uncorrected");
        info.precise_ip = as_bool("preciseIp");
        info.restartable_ip = as_bool("restartableIp");
        info.overflow = as_bool("overflow");
    }

    info.target_identifier =
        json_str(eis, "targetIdentifier").value_or("0x0000000000000000");
    info.requester_identifier =
        json_str(eis, "requesterIdentifier").value_or("0x0000000000000000");
    info.responder_identifier =
        json_str(eis, "responderIdentifier").value_or("0x0000000000000000");
    info.instruction_identifier =
        json_str(eis, "instructionIdentifier").value_or("0x0000000000000000");
    info.fru_manufacturer_part_number =
        json_str(eis, "fruManufacturerPartNumber").value_or(std::string{});
    info.redfish_event_log_id =
        json_str(eis, "redfishEventLogId").value_or(std::string{});

    return info;
}

} // namespace addc::pipeline::detail
