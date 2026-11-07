// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<AmdEpycPmicSection> parsePmicSection(
    const nlohmann::json& descriptor, const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "AmdEpycPMIC");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AmdEpycPmicSection sec;
    fill_common_metadata(sec, descriptor, "venice");
    sec.descriptor_flags = parse_descriptor_flags(descriptor);

    auto parse_hex32 =
        [](const nlohmann::json& obj, std::string_view key) -> uint32_t {
        const auto it = obj.find(std::string{key});
        if (it == obj.end())
        {
            return 0U;
        }
        if (it->is_number_unsigned())
        {
            return it->get<uint32_t>();
        }
        if (it->is_string())
        {
            try
            {
                return static_cast<uint32_t>(
                    std::stoul(it->get<std::string>(), nullptr, 16));
            }
            catch (...)
            {
                return 0U;
            }
        }
        return 0U;
    };

    auto parse_hex8 =
        [](const nlohmann::json& obj, std::string_view key) -> uint8_t {
        const auto it = obj.find(std::string{key});
        if (it == obj.end())
        {
            return 0U;
        }
        if (it->is_number_unsigned())
        {
            return it->get<uint8_t>();
        }
        if (it->is_string())
        {
            try
            {
                return static_cast<uint8_t>(
                    std::stoul(it->get<std::string>(), nullptr, 16));
            }
            catch (...)
            {
                return 0U;
            }
        }
        return 0U;
    };

    sec.section_version = parse_hex32(*body, "section_version");
    sec.section_length = parse_hex32(*body, "section_length");

    if (const nlohmann::json* vb = json_obj(*body, "valid"))
    {
        sec.valid_bits = parse_hex32(*vb, "raw");
    }

    if (const nlohmann::json* id = json_obj(*body, "identification"))
    {
        sec.socket_id = parse_hex8(*id, "socket_id");
        sec.channel_id = parse_hex8(*id, "channel_id");
        sec.dimm_slot = parse_hex8(*id, "dimm_slot");
        sec.pmic_index = parse_hex8(*id, "pmic_index");
    }

    if (const nlohmann::json* fs = json_obj(*body, "fault_status"))
    {
        sec.log_04 = parse_hex8(*fs, "Log_0x04");
        sec.log_05 = parse_hex8(*fs, "Log_0x05");
        sec.log_06 = parse_hex8(*fs, "Log_0x06");
        sec.reg_08 = parse_hex8(*fs, "Reg_0x08");
        sec.reg_09 = parse_hex8(*fs, "Reg_0x09");
        sec.reg_0A = parse_hex8(*fs, "Reg_0x0A");
        sec.reg_0B = parse_hex8(*fs, "Reg_0x0B");
        sec.reg_33 = parse_hex8(*fs, "Reg_0x33");
    }

    if (const nlohmann::json* pl = json_obj(*body, "persistent_logs"))
    {
        for (auto it = pl->begin(); it != pl->end(); ++it)
        {
            std::string val = it->is_string() ? it->get<std::string>()
                                              : std::to_string(it->get<int>());
            sec.persistent_logs.emplace_back(it.key(), std::move(val));
        }
    }

    return sec;
}

} // namespace addc::pipeline
