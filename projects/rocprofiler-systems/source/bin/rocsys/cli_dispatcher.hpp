// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "verb.hpp"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>

namespace rocprofsys::cli
{

enum class dispatch_kind : std::uint8_t
{
    show_help,
    show_version,
    in_process,
    exec_tool,
    error
};

struct dispatch_result
{
    dispatch_kind kind = dispatch_kind::error;
    // A k_verbs row. Those rows have static storage. Null for help, version, and error.
    const rocprofsys::rocsys::verb_spec* spec = nullptr;
    // True when argv[1] is the verb and must be omitted from the forwarded argv.
    bool        strip_subcommand = false;
    std::string error_message;
};

/**
 * Classify a `rocsys` invocation into help, version, in-process tool, exec, or
 * error. Does not execute anything. @p program is the basename used in error
 * text; the caller computes it once so help and errors use the same name.
 */
[[nodiscard]] dispatch_result
parse_dispatch(int argc, char** argv, std::string_view program);

void
print_help(std::ostream& out, std::string_view program);

}  // namespace rocprofsys::cli
