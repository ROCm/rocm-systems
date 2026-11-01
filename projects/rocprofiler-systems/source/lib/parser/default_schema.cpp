// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/default_schema.hpp"
#include "parser/schema.hpp"

#include "common/env_vars.hpp"

#include <fmt/format.h>

#include <array>
#include <stdexcept>
#include <string_view>

namespace rocprofsys::parser
{
namespace
{
constexpr double k_verbose_minimum = -1.0;
constexpr double k_verbose_maximum = 4.0;

auto
default_params()
{
    return std::to_array<schema_param>({
        {
            .id          = "output_path",
            .name        = "Output path",
            .description = "Directory for output files",
            .category    = "configuration",
            .type        = param_type::path,
            .path        = path_kind::directory,
            .env         = env_spec{ .name = env_vars::OUTPUT_PATH },
            .arg         = arg_spec{ .name = "--output", .form = arg_form::value },
        },
        {
            .id          = "preset",
            .name        = "Preset",
            .description = "Named preset to apply",
            .category    = "configuration",
            .type        = param_type::string,
            .arg         = arg_spec{ .name = "--preset", .form = arg_form::value },
        },
        {
            .id          = "config_file",
            .name        = "Config file",
            .description = "JSON file of parameter values",
            .category    = "configuration",
            .type        = param_type::path,
            .path        = path_kind::file,
            .arg         = arg_spec{ .name = "--config", .form = arg_form::value },
        },
        {
            .id            = "verbose",
            .name          = "Verbose",
            .description   = "Verbosity level from -1 to 4",
            .category      = "logging",
            .type          = param_type::integer,
            .default_value = "0",
            .min           = k_verbose_minimum,
            .max           = k_verbose_maximum,
            .env           = env_spec{ .name = env_vars::VERBOSE },
            .arg           = arg_spec{ .name = "--verbose", .form = arg_form::value },
        },
        {
            .id            = "debug",
            .name          = "Debug",
            .description   = "Enable debug mode",
            .category      = "logging",
            .type          = param_type::boolean,
            .default_value = "false",
            .env           = env_spec{ .name = env_vars::DEBUG_MODE },
            .arg           = arg_spec{ .name = "--debug", .form = arg_form::flag },
        },
        {
            .id            = "trace",
            .name          = "Trace",
            .description   = "Enable Perfetto tracing",
            .category      = "tracing",
            .type          = param_type::boolean,
            .default_value = "false",
            .env           = env_spec{ .name = env_vars::TRACE },
            .arg           = arg_spec{ .name = "--trace", .form = arg_form::flag },
        },
        {
            .id            = "profile",
            .name          = "Profile",
            .description   = "Enable call-stack profiling",
            .category      = "tracing",
            .type          = param_type::boolean,
            .default_value = "false",
            .env           = env_spec{ .name = env_vars::PROFILE },
            .arg           = arg_spec{ .name = "--profile", .form = arg_form::flag },
        },
    });
}

void
require_added(bool added, std::string_view label)
{
    if(!added)
    {
        throw std::runtime_error(fmt::format("duplicate catalog entry '{}'", label));
    }
}

void
add_categories(schema& catalog)
{
    require_added(catalog.add_category(category{
                      .id   = "configuration",
                      .name = "Configuration",
                  }),
                  "configuration");
    require_added(catalog.add_category(category{
                      .id   = "logging",
                      .name = "Logging",
                  }),
                  "logging");
    require_added(catalog.add_category(category{
                      .id   = "tracing",
                      .name = "Tracing",
                  }),
                  "tracing");
}
}  // namespace

schema
get_default_schema()
{
    auto catalog = schema{};
    catalog.set_version("1.0");
    add_categories(catalog);
    for(const auto& param : default_params())
    {
        require_added(catalog.add_param(param), param.id);
    }
    return catalog;
}

}  // namespace rocprofsys::parser
