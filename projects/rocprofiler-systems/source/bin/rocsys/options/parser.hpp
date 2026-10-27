// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "options/option_spec.hpp"
#include "result.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys::options
{

/// @brief One recognised rocsys option. @ref spec points at the caller's table.
struct parsed_option
{
    const option_spec*       spec   = nullptr;
    std::vector<std::string> values = {};
};

/// @brief Tokens before `--` split into rocsys options and legacy passthrough.
///        @ref app includes the `--` separator when the user wrote one.
struct parsed_command
{
    const verb_spec*           verb        = nullptr;
    std::vector<parsed_option> options     = {};
    std::vector<std::string>   passthrough = {};
    std::vector<std::string>   app         = {};
};

/// @brief Input to @ref parse. Pointers and spans must outlive the call.
struct parse_request
{
    const verb_spec*             verb             = nullptr;
    bool                         strip_subcommand = {};
    std::span<const std::string> argv             = {};
    std::span<const option_spec> options          = {};
    const context*               ctx              = nullptr;
};

/// @brief Pull rocsys options out of the tokens before `--`.
///        Unknown long options and short options stay in passthrough, in order.
///        Repeated options become one @ref parsed_option. When two rows share a
///        name, the row whose required capabilities the verb has is selected.
[[nodiscard]] result<parsed_command>
parse(const parse_request& request);

/// @brief argv0 plus passthrough plus app, which is what delivery forwards.
[[nodiscard]] std::vector<std::string>
command_argv(std::string_view argv0, const parsed_command& command);

}  // namespace rocprofsys::rocsys::options
