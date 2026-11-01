// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "mtl_section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<AmdEpycMtlSection> parseMi450MtlSection(
    const nlohmann::json& descriptor, const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "AmdMi450MTL");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AmdEpycMtlSection sec;
    fill_common_metadata(sec, descriptor, "mi450");

    // Early MI450 MTL firmware reports the Venice-compatible 0x31 descriptor
    // revision, so the parser-selected section type is authoritative here.
    sec.platform = "mi450";
    sec.platform_name = std::string{platform_display_name(sec.platform)};
    sec.descriptor_flags = parse_descriptor_flags(descriptor);
    parse_mtl_body(sec, *body);

    return sec;
}

} // namespace addc::pipeline
