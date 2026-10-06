// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<AmdEpycCddSection> parseCddSection(
    const nlohmann::json& descriptor, const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "AmdEpycCDD");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AmdEpycCddSection sec;
    fill_common_metadata(sec, descriptor, "venice");
    sec.descriptor_flags = parse_descriptor_flags(descriptor);

    if (const nlohmann::json* vb = json_obj(*body, "validBits"))
    {
        if (const auto v = json_str(*vb, "raw"))
        {
            try
            {
                sec.valid_bits =
                    static_cast<uint16_t>(std::stoul(*v, nullptr, 16));
            }
            catch (...)
            {
                sec.valid_bits = 0U;
            }
        }
    }
    if (const auto v = json_u64(*body, "apicId"))
    {
        sec.apic_id = *v;
    }

    if (const nlohmann::json* cpuid = json_obj(*body, "cpuidInfo"))
    {
        auto parse_hex64 =
            [](const nlohmann::json& obj, std::string_view key) -> uint64_t {
            const auto it = obj.find(std::string{key});
            if (it == obj.end())
            {
                return 0U;
            }
            if (it->is_number_unsigned())
            {
                return it->get<uint64_t>();
            }
            if (it->is_string())
            {
                try
                {
                    return std::stoull(it->get<std::string>(), nullptr, 16);
                }
                catch (...)
                {
                    return 0U;
                }
            }
            return 0U;
        };
        sec.cpuid_eax = parse_hex64(*cpuid, "eax");
        sec.cpuid_ebx = parse_hex64(*cpuid, "ebx");
        sec.cpuid_ecx = parse_hex64(*cpuid, "ecx");
        sec.cpuid_edx = parse_hex64(*cpuid, "edx");
    }

    if (auto bytes = json_bytes(*body, "payload"))
    {
        sec.payload = std::move(*bytes);
    }

    return sec;
}

} // namespace addc::pipeline
