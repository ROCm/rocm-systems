// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys
{
/// What rocsys does with one command line.
struct options
{
    /// Variables rocsys sets before the tool starts.
    parser::value_map_t environment;
    /// Every token rocsys does not own, in order. The tool parses these.
    std::vector<std::string> tool_args;
    /// The application and its arguments, after "--".
    std::vector<std::string> command;
    std::vector<std::string> errors;
};

/// Read the rocsys options in @p args, the tokens after the subcommand. A --traces ROCm
/// API name counts only when its domain is in @p supported_domains, the
/// ROCPROFSYS_ROCM_DOMAINS values the installed rocprofiler-sdk takes. Does not change
/// the process environment.
[[nodiscard]] options
parse_options(const std::vector<std::string>& args,
              const std::vector<std::string>& supported_domains);

/// Each rocsys option on one line, under its category, from the catalog.
[[nodiscard]] std::string
options_summary();

/// The rocsys options in full, from the catalog: one category, such as "tracing", one
/// option, such as "--traces", or every option for "all". --traces lists the ROCm API
/// names whose domain is in @p supported_domains. Nothing for any other topic.
[[nodiscard]] std::optional<std::string>
options_help(std::string_view topic, const std::vector<std::string>& supported_domains);
}  // namespace rocprofsys::rocsys
