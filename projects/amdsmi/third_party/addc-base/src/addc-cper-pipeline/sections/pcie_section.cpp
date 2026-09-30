// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<PcieSection> parsePcieSection(const nlohmann::json& descriptor,
                                              const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "PCIe");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    PcieSection sec;
    fill_common_metadata(sec, descriptor, "venice");
    sec.descriptor_flags = parse_descriptor_flags(descriptor);
    sec.raw_body = *body;
    return sec;
}

} // namespace addc::pipeline
