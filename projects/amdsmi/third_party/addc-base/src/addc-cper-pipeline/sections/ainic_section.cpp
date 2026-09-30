// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "section_helpers.hpp"

namespace addc::pipeline
{

using namespace detail;

std::optional<AinicPlatformContextSection> parseAinicSection(
    const nlohmann::json& descriptor, const nlohmann::json& section)
{
    const nlohmann::json* body = json_obj(section, "AinicPlatformContext");
    if (body == nullptr)
    {
        return std::nullopt;
    }

    AinicPlatformContextSection sec;
    sec.descriptor_flags = parse_descriptor_flags(descriptor);
    sec.category = json_str(*body, "category").value_or(std::string{});
    sec.type = json_str(*body, "type").value_or(std::string{});
    sec.board_serial = json_str(*body, "boardSerial").value_or(std::string{});
    sec.fw_version = json_str(*body, "fwVersion").value_or(std::string{});
    sec.fru = json_str(descriptor, "fruText").value_or(std::string{});
    sec.fru_id =
        json_str(descriptor, "fruID").value_or(std::string{kZeroGuid});
    return sec;
}

} // namespace addc::pipeline
