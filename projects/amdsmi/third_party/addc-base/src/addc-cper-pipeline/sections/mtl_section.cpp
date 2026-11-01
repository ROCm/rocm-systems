// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "mtl_section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<AmdEpycMtlSection> parseMtlSection(
    const nlohmann::json& descriptor, const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "AmdEpycMTL");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AmdEpycMtlSection sec;
    fill_common_metadata(sec, descriptor, "venice");
    sec.descriptor_flags = parse_descriptor_flags(descriptor);
    parse_mtl_body(sec, *body);

    return sec;
}

} // namespace addc::pipeline
