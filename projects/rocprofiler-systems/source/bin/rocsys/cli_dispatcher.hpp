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
    show_subcommand_help,
    show_option_help,
    show_version,
    run_subcommand,
    error
};

struct subcommand_spec
{
    std::string_view name;
    std::string_view description;
    std::string_view binary_name;
    // True when the subcommand runs an application, which must follow "--". False
    // (avail, attach) refuses "--".
    bool requires_app = true;
    // False: rocsys reads none of its options and passes every argument to the tool,
    // and this subcommand's help leaves the rocsys options out.
    bool takes_rocsys_options = true;
    // What this subcommand's help says it runs. Empty names binary_name.
    std::string_view runs;
    // Set only for in-process verbs. Exec verbs leave this empty.
    std::optional<common_utils::tool_mode> mode;
    // Arguments inserted after tool's executable (for example,
    // "rewrite" will add "-o" to "rocprof-sys-instrument",
    // becoming "rocprof-sys-instrument -o").
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
inline constexpr auto k_subcommands = std::to_array<subcommand_spec>({
    {
        .name         = "profile",
        .description  = "Full trace profile (default)",
        .binary_name  = "rocprof-sys-run",
        .requires_app = true,
        .runs         = {},
        .mode         = common_utils::tool_mode::run,
        .extra_flag   = {},
    },
    {
        .name                 = "avail",
        .description          = "Query available counters and settings",
        .binary_name          = "rocprof-sys-avail",
        .requires_app         = false,
        .takes_rocsys_options = false,
        .runs                 = {},
        .extra_flag           = {},
    },
    {
        .name         = "attach",
        .description  = "Attach to a running process",
        .binary_name  = "rocprof-sys-attach",
        .requires_app = false,
        .runs         = {},
        .extra_flag   = {},
    },
    {
        .name                 = "causal",
        .description          = "Causal profiling",
        .binary_name          = "rocprof-sys-causal",
        .requires_app         = true,
        .takes_rocsys_options = false,
        .runs                 = {},
        .extra_flag           = {},
    },
    {
        .name         = "instrument",
        .description  = "Runtime instrumentation profiling",
        .binary_name  = "rocprof-sys-instrument",
        .requires_app = true,
        .runs         = {},
        .extra_flag   = {},
    },
    {
        .name                 = "rewrite",
        .description          = "Binary rewrite application",
        .binary_name          = "rocprof-sys-instrument",
        .requires_app         = true,
        .takes_rocsys_options = false,
        .runs = "rocprof-sys-instrument with the -o option to generate an instrumented "
                "binary, such as app.inst for ./app",
        .extra_flag = "-o",
    },
    {
        .name         = "python",
        .description  = "Python application profiling",
        .binary_name  = "rocprof-sys-python",
        .requires_app = true,
        .runs         = {},
        .extra_flag   = {},
    },
});

struct dispatch_result
{
    dispatch_kind kind = dispatch_kind::error;
    // A k_subcommands row. Those rows have static storage. Null for show_help,
    // show_version, and error.
    const subcommand_spec* spec = nullptr;
    // True when argv[1] is the verb and must be omitted from the forwarded argv.
    bool strip_subcommand = false;
    // For show_help, the text after "--help=".
    std::string_view topic;
    // For error, and for show_option_help when no option has that name.
    std::string error_message;
};

/**
 * Classify a `rocsys` invocation into help, version, a subcommand to run, or error.
 * Does not execute anything. @p program is the basename used in error text; the caller
 * computes it once so help and errors use the same name.
 * `-h`, `-?`, or `--help` before `--` asks for the subcommand's rocsys help.
 * `--help=<topic>` is the tool's, so it is forwarded. Any other command of a subcommand
 * with requires_app without `--` is an error, and so is a plain word right after the
 * subcommand, even when a help flag follows it. A subcommand without requires_app
 * refuses `--`.
 */
[[nodiscard]] dispatch_result
parse_dispatch(int argc, char** argv, std::string_view program);

/**
 * Build the argument strings for the selected tool. When
 * @p options.strip_subcommand is true, @p argv[1] (the subcommand token) is
 * omitted. A non-empty @p options.argv0_override replaces @p argv[0]. A
 * non-empty @p options.extra_flag is inserted after argv0 unless the payload
 * already contains that flag (or ``-o`` / ``--output`` when the flag is ``-o``).
 */
[[nodiscard]] std::vector<std::string>
make_forwarded_argv(int argc, char** argv, forward_options options);

/// The rocsys help: the subcommands, then @p options_text, the rendered rocsys options.
void
print_help(std::ostream& out, std::string_view program, std::string_view options_text);

/// The rocsys help for @p spec: @p options_text as in print_help() when the row takes the
/// rocsys options, and the tool whose `--help` lists every other option.
void
print_subcommand_help(std::ostream& out, std::string_view program,
                      const subcommand_spec& spec, std::string_view options_text);

}  // namespace rocprofsys::cli
