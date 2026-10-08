// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/tool_runner.hpp"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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

struct subcommand_spec
{
    std::string_view name;
    std::string_view description;
    std::string_view binary_name;
    bool             requires_app = true;
    // Set only for in-process verbs. Exec verbs leave this empty.
    std::optional<common_utils::tool_mode> mode;
    // Inserted after argv0 when forwarding (for example "-o" for rewrite).
    std::string_view extra_flag;
};

struct forward_options
{
    bool             strip_subcommand = false;
    std::string_view argv0_override;
    std::string_view extra_flag;
};

// The first row is the implicit default. parse_dispatch uses it when argv
// has flags or "--" and no subcommand token.
constexpr auto k_subcommands = std::to_array<subcommand_spec>({
    {
        .name         = "profile",
        .description  = "Full trace profile (default)",
        .binary_name  = "rocprof-sys-run",
        .requires_app = true,
        .mode         = common_utils::tool_mode::run,
        .extra_flag   = {},
    },
    {
        .name         = "avail",
        .description  = "Query available counters and settings",
        .binary_name  = "rocprof-sys-avail",
        .requires_app = false,
        .extra_flag   = {},
    },
    {
        .name         = "attach",
        .description  = "Attach to a running process",
        .binary_name  = "rocprof-sys-attach",
        .requires_app = true,
        .extra_flag   = {},
    },
    {
        .name         = "causal",
        .description  = "Causal profiling",
        .binary_name  = "rocprof-sys-causal",
        .requires_app = true,
        .extra_flag   = {},
    },
    {
        .name         = "instrument",
        .description  = "Runtime instrumentation (Dyninst)",
        .binary_name  = "rocprof-sys-instrument",
        .requires_app = true,
        .extra_flag   = {},
    },
    {
        .name         = "rewrite",
        .description  = "Binary rewrite (Dyninst)",
        .binary_name  = "rocprof-sys-instrument",
        .requires_app = true,
        .extra_flag   = "-o",
    },
    {
        .name         = "python",
        .description  = "Python application profiling",
        .binary_name  = "rocprof-sys-python",
        .requires_app = true,
        .extra_flag   = {},
    },
});

struct dispatch_result
{
    dispatch_kind kind = dispatch_kind::error;
    // A k_subcommands row. Those rows have static storage. Null for help,
    // version, and error.
    const subcommand_spec* spec = nullptr;
    // True when argv[1] is the verb and must be omitted from the forwarded argv.
    bool        strip_subcommand = false;
    std::string error_message;
};

/**
 * Classify a `rocsys` invocation into help, version, in-process tool, exec, or
 * error. Does not execute anything.
 */
[[nodiscard]] dispatch_result
parse_dispatch(int argc, char** argv);

/**
 * Build the argument strings for the selected tool. When
 * @p options.strip_subcommand is true, @p argv[1] (the subcommand token) is
 * omitted. A non-empty @p options.argv0_override replaces @p argv[0]. A
 * non-empty @p options.extra_flag is inserted after argv0 unless the payload
 * already contains that flag (or ``-o`` / ``--output`` when the flag is ``-o``).
 */
[[nodiscard]] std::vector<std::string>
make_forwarded_argv(int argc, char** argv, forward_options options);

void
print_help(std::ostream& out, std::string_view program);

}  // namespace rocprofsys::cli
