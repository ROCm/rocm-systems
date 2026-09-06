// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/tool_runner.hpp"

#include <array>
#include <cstdint>
#include <iosfwd>
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
    std::string_view        name;
    std::string_view        description;
    std::string_view        binary_name;
    bool                    in_process   = false;
    bool                    requires_app = true;
    common_utils::tool_mode mode         = common_utils::tool_mode::sample;
};

inline constexpr std::array<subcommand_spec, 7> subcommands{ {
    { "profile", "Low-overhead sampling profile (default)", "rocprof-sys-sample", true,
      true, common_utils::tool_mode::sample },
    { "trace", "Full trace / run-style profile", "rocprof-sys-run", true, true,
      common_utils::tool_mode::run },
    { "instrument", "Binary instrumentation (Dyninst)", "rocprof-sys-instrument", false,
      true },
    { "causal", "Causal profiling", "rocprof-sys-causal", false, true },
    { "avail", "Query available counters and settings", "rocprof-sys-avail", false,
      false },
    { "python", "Python application profiling", "rocprof-sys-python", false, true },
    { "attach", "Attach to a running process", "rocprof-sys-attach", false, true },
} };

struct dispatch_result
{
    dispatch_kind           kind             = dispatch_kind::error;
    common_utils::tool_mode mode             = common_utils::tool_mode::sample;
    std::string_view        binary_name      = {};
    std::string_view        subcommand_name  = {};
    bool                    strip_subcommand = false;
    std::string             error_message    = {};
};

struct forwarded_argv
{
    std::string        argv0_storage;
    std::vector<char*> ptrs;

    [[nodiscard]] int argc() const noexcept
    {
        return ptrs.empty() ? 0 : static_cast<int>(ptrs.size()) - 1;
    }

    [[nodiscard]] char** argv() noexcept { return ptrs.data(); }
};

[[nodiscard]] constexpr const subcommand_spec*
find_subcommand(std::string_view name) noexcept
{
    for(const auto& spec : subcommands)
    {
        if(spec.name == name) return &spec;
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
 * @p strip_subcommand is true, @p argv[1] (the subcommand token) is omitted.
 * When @p argv0_override is non-empty it replaces @p argv[0].
 */
[[nodiscard]] forwarded_argv
make_forwarded_argv(int argc, char** argv, bool strip_subcommand,
                    std::string_view argv0_override = {});

void
print_help(std::ostream& out, std::string_view program);

void
print_version(std::ostream& out, std::string_view program, std::string_view version);

}  // namespace rocprofsys::cli
