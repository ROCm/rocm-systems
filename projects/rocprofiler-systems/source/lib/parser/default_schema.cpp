// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/default_schema.hpp"
#include "parser/schema.hpp"

#include "common/env_vars.hpp"

#include <fmt/format.h>

#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
constexpr double k_verbose_minimum = -1.0;
constexpr double k_verbose_maximum = 4.0;

void
require_added(bool added, std::string_view label)
{
    if(!added)
    {
        throw std::runtime_error(fmt::format("duplicate catalog entry '{}'", label));
    }
}

schema_param
text_param(std::string id, std::string description, std::string category, param_type type)
{
    auto param        = schema_param{};
    param.id          = std::move(id);
    param.description = std::move(description);
    param.category    = std::move(category);
    param.type        = type;
    return param;
}
}  // namespace

schema
get_default_schema()
{
    auto output = text_param("output_path", "Directory for output files", "configuration",
                             param_type::path);
    output.env  = env_vars::OUTPUT_PATH;
    output.arg  = "--output";

    auto preset = text_param("preset", "Named preset to apply", "configuration",
                             param_type::string);
    preset.arg  = "--preset";

    auto config_file = text_param("config_file", "JSON file of parameter values",
                                  "configuration", param_type::path);
    config_file.arg  = "--config";

    auto verbose = text_param("verbose", "Verbosity level from -1 to 4", "logging",
                              param_type::integer);
    verbose.default_value = "0";
    verbose.min           = k_verbose_minimum;
    verbose.max           = k_verbose_maximum;
    verbose.env           = env_vars::VERBOSE;
    verbose.arg           = "--verbose";

    auto debug = text_param("debug", "Enable debug mode", "logging", param_type::boolean);
    debug.default_value = "false";
    debug.flag          = true;
    debug.env           = env_vars::DEBUG_MODE;
    debug.arg           = "--debug";

    auto trace =
        text_param("trace", "Enable Perfetto tracing", "tracing", param_type::boolean);
    trace.default_value = "false";
    trace.flag          = true;
    trace.env           = env_vars::TRACE;
    trace.arg           = "--trace";

    auto profile = text_param("profile", "Enable call-stack profiling", "tracing",
                              param_type::boolean);
    profile.default_value = "false";
    profile.flag          = true;
    profile.env           = env_vars::PROFILE;
    profile.arg           = "--profile";

    auto catalog = schema{};
    catalog.set_version("1.0");
    for(auto& param : std::vector<schema_param>{
            std::move(output), std::move(preset), std::move(config_file),
            std::move(verbose), std::move(debug), std::move(trace), std::move(profile) })
    {
        const auto label = param.id;
        require_added(catalog.add_param(std::move(param)), label);
    }
    return catalog;
}

}  // namespace rocprofsys::parser
