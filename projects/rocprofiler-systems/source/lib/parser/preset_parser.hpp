// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

#include <map>
#include <string>

namespace rocprofsys::parser
{
/**
 * Maps a preset's environment settings onto schema parameters.
 *
 * @p settings is the map rocprofsys::preset_registry::get_settings() returns:
 * environment variable name to textual value. Variables that name a parameter
 * are stored under that parameter's id. Choice aliases are stored as the choice
 * value. Other variables are reported and skipped.
 */
class preset_parser
{
public:
    using settings_t = std::map<std::string, std::string>;

    [[nodiscard]] parse_result parse(const schema&     config_schema,
                                     const settings_t& settings) const;
};

}  // namespace rocprofsys::parser
