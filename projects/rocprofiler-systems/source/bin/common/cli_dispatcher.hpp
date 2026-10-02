// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/tool_runner.hpp"

#include <algorithm>
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
    bool             in_process   = false;
    bool             requires_app = true;
    // Set only for in-process verbs. Exec verbs leave this empty.
    std::optional<common_utils::tool_mode> mode;
    bool                                   is_default = false;
    // Inserted after argv0 when forwarding (for example "-o" for rewrite).
    std::string_view extra_flag;
};

struct forward_options
{
    bool             strip_subcommand = false;
    std::string_view argv0_override;
    std::string_view extra_flag;
};

// Exactly one row is is_default. parse_dispatch uses that row for the
// explicit verb and when argv has flags or "--" but no subcommand token.
constexpr auto k_subcommands = std::to_array<subcommand_spec>({
    {
        .name         = "profile",
        .description  = "Full trace profile (default)",
        .binary_name  = "rocprof-sys-run",
        .in_process   = true,
        .requires_app = true,
        .mode         = common_utils::tool_mode::run,
        .is_default   = true,
        .extra_flag   = {},
    },
    {
        .name         = "instrument",
        .description  = "Runtime instrumentation (Dyninst)",
        .binary_name  = "rocprof-sys-instrument",
        .in_process   = false,
        .requires_app = true,
        .is_default   = false,
        .extra_flag   = {},
    },
    {
        .name         = "rewrite",
        .description  = "Binary rewrite (Dyninst)",
        .binary_name  = "rocprof-sys-instrument",
        .in_process   = false,
        .requires_app = true,
        .is_default   = false,
        .extra_flag   = "-o",
    },
    {
        .name         = "causal",
        .description  = "Causal profiling",
        .binary_name  = "rocprof-sys-causal",
        .in_process   = false,
        .requires_app = true,
        .is_default   = false,
        .extra_flag   = {},
    },
    {
        .name         = "avail",
        .description  = "Query available counters and settings",
        .binary_name  = "rocprof-sys-avail",
        .in_process   = false,
        .requires_app = false,
        .is_default   = false,
        .extra_flag   = {},
    },
    {
        .name         = "python",
        .description  = "Python application profiling",
        .binary_name  = "rocprof-sys-python",
        .in_process   = false,
        .requires_app = true,
        .is_default   = false,
        .extra_flag   = {},
    },
    {
        .name         = "attach",
        .description  = "Attach to a running process",
        .binary_name  = "rocprof-sys-attach",
        .in_process   = false,
        .requires_app = true,
        .is_default   = false,
        .extra_flag   = {},
    },
});

[[nodiscard]] constexpr int
default_subcommand_count() noexcept
{
    int count = 0;
    for(const auto& spec : k_subcommands)
    {
        if(spec.is_default)
        {
            ++count;
        }
    }
    return count;
}

[[nodiscard]] constexpr const subcommand_spec&
default_subcommand() noexcept
{
    for(const auto& spec : k_subcommands)
    {
        if(spec.is_default)
        {
            return spec;
        }
    }
    return k_subcommands.front();
}

[[nodiscard]] constexpr bool
subcommand_modes_match_dispatch() noexcept
{
    return std::ranges::all_of(k_subcommands, [](const subcommand_spec& spec) {
        return spec.in_process == spec.mode.has_value();
    });
}

static_assert(default_subcommand_count() == 1,
              "exactly one subcommand_spec must be is_default");
static_assert(default_subcommand().name == "profile",
              "default subcommand verb is profile");
static_assert(default_subcommand().mode == common_utils::tool_mode::run,
              "default subcommand runs in-process as tool_mode::run");
static_assert(subcommand_modes_match_dispatch(),
              "in-process verbs must set mode; exec verbs must leave it empty");

struct dispatch_result
{
    dispatch_kind                          kind = dispatch_kind::error;
    std::optional<common_utils::tool_mode> mode;
    std::string_view                       binary_name;
    std::string_view                       subcommand_name;
    bool                                   strip_subcommand = false;
    std::string_view                       extra_flag;
    std::string                            error_message;
};

// Owns forwarded argument strings. Pointers returned by argv() stay valid
// across copy and move.
struct forwarded_argv
{
    std::vector<std::string> args;
    std::vector<char*>       ptrs;

    forwarded_argv() = default;
    forwarded_argv(const forwarded_argv& other);
    forwarded_argv& operator=(const forwarded_argv& other);
    forwarded_argv(forwarded_argv&& other) noexcept;
    forwarded_argv& operator=(forwarded_argv&& other) noexcept;
    ~forwarded_argv() = default;

    void bind() noexcept;

    [[nodiscard]] int argc() const noexcept
    {
        if(ptrs.empty())
        {
            return 0;
        }
        return static_cast<int>(ptrs.size() - 1);
    }

    [[nodiscard]] char** argv() noexcept { return ptrs.data(); }
};

[[nodiscard]] constexpr const subcommand_spec*
find_subcommand(std::string_view name) noexcept
{
    if(name.empty())
    {
        return nullptr;
    }
    for(const auto& spec : k_subcommands)
    {
        if(spec.name == name)
        {
            return &spec;
        }
    }
    return nullptr;
}

[[nodiscard]] std::string_view
program_name(std::string_view argv0) noexcept;

[[nodiscard]] std::string
directory_of(std::string_view argv0);

[[nodiscard]] std::string
join_sibling_path(std::string_view directory, std::string_view binary_name);

/**
 * Classify a `rocsys` invocation into help, version, in-process tool, exec, or
 * error. Does not execute anything.
 */
[[nodiscard]] dispatch_result
parse_dispatch(int argc, char** argv);

/**
 * Build a null-terminated argv for the selected tool. When
 * @p options.strip_subcommand is true, @p argv[1] (the subcommand token) is
 * omitted. A non-empty @p options.argv0_override replaces @p argv[0]. A
 * non-empty @p options.extra_flag is inserted after argv0 unless the payload
 * already contains that flag (or ``-o`` / ``--output`` when the flag is ``-o``).
 */
[[nodiscard]] forwarded_argv
make_forwarded_argv(int argc, char** argv, forward_options options);

void
print_help(std::ostream& out, std::string_view program);

void
print_version(std::ostream& out, std::string_view program, std::string_view version);

}  // namespace rocprofsys::cli
