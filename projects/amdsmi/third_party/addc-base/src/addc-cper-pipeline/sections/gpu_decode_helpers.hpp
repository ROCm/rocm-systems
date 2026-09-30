// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/boot/decoder.hpp"
#include "addc/common.hpp"
#include "addc/detail/format.hpp"
#include "addc/pipeline/detect.hpp"
#include "addc/pipeline/section_data.hpp"
#include "addc/pipeline/decode_context.hpp"
#include "addc/pipeline/section_data.hpp"
#include "section_helpers.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <ctime>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace addc::pipeline::gpu
{

[[nodiscard]] inline std::string format_hex64(uint64_t v)
{
    return addc::format("0x{:016x}", v);
}

[[nodiscard]] inline std::string decode_fru_id_ascii(std::string_view guid_str)
{
    std::string result;
    for (std::size_t i = 0u; i < guid_str.size(); i += 2u)
    {
        if (guid_str[i] == '-')
        {
            ++i;
            if (i + 1u >= guid_str.size())
                break;
        }
        if (i + 1u >= guid_str.size())
            break;
        auto hex_val = [](char c) -> int {
            if (c >= '0' && c <= '9')
                return c - '0';
            if (c >= 'a' && c <= 'f')
                return c - 'a' + 10;
            if (c >= 'A' && c <= 'F')
                return c - 'A' + 10;
            return -1;
        };
        const int hi = hex_val(guid_str[i]);
        const int lo = hex_val(guid_str[i + 1u]);
        if (hi < 0 || lo < 0)
            break;
        const char ch = static_cast<char>((hi << 4) | lo);
        if (ch == '\0')
            break;
        result += ch;
    }
    return result;
}

[[nodiscard]] inline nlohmann::ordered_json crashdump_registers_to_json(
    const McaBankDump& b)
{
    nlohmann::ordered_json j;
    j["status"] = format_hex64(b.status);
    j["addr"] = format_hex64(b.addr);
    j["ipid"] = format_hex64(b.ipid);
    j["synd"] = format_hex64(b.synd);
    return j;
}

template <typename GpuSection>
[[nodiscard]] nlohmann::ordered_json make_gpu_metadata(
    const GpuSection& s, const DecodeContext& context, std::size_t ci,
    std::string_view context_type, std::string_view event_class,
    std::size_t event_index)
{
    nlohmann::ordered_json m;
    m["section_index"] = context.section_index;
    m["context_structure_index"] = ci;
    m["context_type"] = context_type;
    m["event_class"] = event_class;
    m["event_index"] = event_index;
    m["timestamp"] = context.timestamp;
    m["addc_version"] = {{"name", s.platform_name}, {"value", s.addc_version}};
    return m;
}

template <typename GpuSection>
[[nodiscard]] nlohmann::ordered_json make_firmware_json(const GpuSection& s)
{
    nlohmann::ordered_json fw = nlohmann::ordered_json::object();
    fw["fw_id"] = s.fw_id_valid && !s.fw_id.empty()
                      ? nlohmann::ordered_json(s.fw_id)
                      : nlohmann::ordered_json(nullptr);
    if (s.pldm_bundle_valid)
    {
        fw["pldm_bundle"] = nlohmann::ordered_json{
            {"minor", s.pldm_bundle.minor},
            {"major", s.pldm_bundle.major},
            {"year", s.pldm_bundle.year},
            {"production_debug", s.pldm_bundle.production_debug},
        };
    }
    else
    {
        fw["pldm_bundle"] = nullptr;
    }
    if (!s.fw_id_valid && !s.pldm_bundle_valid)
        return nullptr;
    return fw;
}

[[nodiscard]] inline nlohmann::ordered_json make_redfish_event_log(
    bool valid, const std::string& raw)
{
    nlohmann::ordered_json obj;
    obj["id"] = nullptr;
    if (!valid || raw.empty())
    {
        obj["timestamp"] = nullptr;
        return obj;
    }
    // Format: {unix_timestamp}[_{disambiguator}] — always decode as timestamp.
    const auto sep = raw.find('_');
    const std::string ts_part =
        (sep != std::string::npos) ? raw.substr(0, sep) : raw;
    try
    {
        const auto ts = static_cast<std::time_t>(std::stoull(ts_part));
        std::tm tm{};
#if defined(_WIN32)
        _gmtime64_s(&tm, &ts);
#else
        gmtime_r(&ts, &tm);
#endif
        obj["timestamp"] = addc::format(
            "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}Z", tm.tm_year + 1900,
            tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    }
    catch (...)
    {
        obj["timestamp"] = nullptr;
    }
    return obj;
}

template <typename GpuSection>
[[nodiscard]] nlohmann::ordered_json make_gpu_pcie_device(const GpuSection& s)
{
    if (!s.pcie_devid_valid || s.pcie_dev_id.empty())
        return nullptr;

    // Base reports the identifier exactly as supplied. A human-readable
    // product name is unavailable at this knowledge level.
    return nlohmann::ordered_json{
        {"id", s.pcie_dev_id},
        {"name", nullptr},
    };
}

template <typename GpuSection>
[[nodiscard]] nlohmann::ordered_json make_gpu_platform(
    const GpuSection& s, bool has_error_info = false,
    const GpuErrorInfoStructure* ei = nullptr)
{
    nlohmann::ordered_json p;
    p["fru"] = s.fru;
    p["fru_id"] = s.fru_id;
    p["cpu"] = nullptr;

    nlohmann::ordered_json gpu_obj = nlohmann::ordered_json::object();

    auto serial = decode_fru_id_ascii(s.fru_id);
    gpu_obj["serial_number"] = serial.empty() ? nlohmann::ordered_json(nullptr)
                                              : nlohmann::ordered_json(serial);

    gpu_obj["pcie_device"] = make_gpu_pcie_device(s);

    if (has_error_info && ei)
    {
        gpu_obj["fru_manufacturer_part_number"] =
            ei->fru_mpn_valid
                ? nlohmann::ordered_json(ei->fru_manufacturer_part_number)
                : nlohmann::ordered_json(nullptr);
        const auto redfish = make_redfish_event_log(
            ei->redfish_event_log_id_valid, ei->redfish_event_log_id);
        gpu_obj["redfish_event_log_id"] = redfish["id"];
        gpu_obj["redfish_event_log_timestamp"] = redfish["timestamp"];
    }
    else
    {
        gpu_obj["fru_manufacturer_part_number"] = nullptr;
        gpu_obj["redfish_event_log_id"] = nullptr;
        gpu_obj["redfish_event_log_timestamp"] = nullptr;
    }
    p["gpu"] = std::move(gpu_obj);
    return p;
}

[[nodiscard]] inline std::optional<McaBankDump> decode_runtime_mca_bank(
    std::span<const uint8_t> bytes)
{
    if (bytes.size() < 128u)
        return std::nullopt;
    return detail::parse_mca_bank(bytes, 0u);
}

[[nodiscard]] inline std::optional<McaBankDump> decode_crashdump_mca_bank(
    std::span<const uint8_t> bytes)
{
    if (bytes.size() < 32u)
        return std::nullopt;
    const auto d = bytes;
    return McaBankDump{
        .status = detail::read_u64_le(d, 0x00u),
        .addr = detail::read_u64_le(d, 0x08u),
        .ipid = detail::read_u64_le(d, 0x10u),
        .synd = detail::read_u64_le(d, 0x18u),
    };
}

[[nodiscard]] inline std::vector<uint64_t> decode_boot_messages(
    std::span<const uint8_t> bytes)
{
    if (bytes.size() < 8u)
        return {};

    std::vector<uint64_t> messages;
    const std::size_t count = bytes.size() / 8u;
    messages.reserve(count);
    for (std::size_t i = 0u; i < count; ++i)
        messages.push_back(detail::read_u64_le(bytes, i * 8u));
    return messages;
}

struct BootAfidResult
{
    int afid = kAfidSentinel;
    bool has_a4 = false;
};

[[nodiscard]] inline BootAfidResult compute_boot_afid(
    const std::vector<uint64_t>& messages)
{
    BootAfidResult r;
    bool has_non_boot_success = false;
    for (const uint64_t v : messages)
    {
        addc::boot::OamBootMsg m{v};
        if (m.msg_0 == 0xA4u)
        {
            r.has_a4 = true;
            if (r.afid == kAfidSentinel)
            {
                r.afid = addc::boot::boot_error_type_to_afid(
                    m.error_type_id, m.format_version, m.msg_0);
            }
        }
        if (m.msg_0 != 0xBAu && m.msg_0 != 0xA4u)
            has_non_boot_success = true;
    }
    if (r.afid == kAfidSentinel && has_non_boot_success)
        r.afid = 5;
    return r;
}

[[nodiscard]] inline nlohmann::ordered_json make_boot_data_array(
    const std::vector<uint64_t>& messages)
{
    nlohmann::ordered_json data = nlohmann::ordered_json::object();
    for (std::size_t i = 0u; i < messages.size(); ++i)
        data[addc::format("boot_msg{}", i)] = format_hex64(messages[i]);
    return data;
}

[[nodiscard]] inline nlohmann::ordered_json make_base_boot_reports(
    const std::vector<uint64_t>& messages, bool has_a4)
{
    nlohmann::ordered_json reports = nlohmann::ordered_json::array();
    for (const uint64_t v : messages)
    {
        const uint8_t msg_0 = static_cast<uint8_t>(v & 0xFFu);
        if (msg_0 == 0xA4u || (!has_a4 && msg_0 != 0xBAu))
        {
            nlohmann::ordered_json r;
            r["error_type"] = nullptr;
            r["severity"] = nullptr;
            r["description"] = nullptr;
            r["socket"] = nullptr;
            r["aid"] = nullptr;
            r["additional_data"] = nullptr;
            reports.push_back(std::move(r));
        }
    }
    return reports;
}

inline nlohmann::ordered_json make_boot_validation()
{
    nlohmann::ordered_json v;
    v["struct"] = true;
    v["bank"] = false;
    v["corrupted"] = false;
    v["possible_non_mca_fatal"] = false;
    v["mtl_raw_data"] = false;
    return v;
}

} // namespace addc::pipeline::gpu
