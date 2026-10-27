// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_dispatcher.hpp"

#include "help/renderer.hpp"
#include "options/table.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <string>
#include <string_view>

namespace rocprofsys::cli
{
namespace
{
constexpr std::string_view k_help_short      = "-h";
constexpr std::string_view k_help_alt_short  = "-?";
constexpr std::string_view k_help_long       = "--help";
constexpr std::string_view k_help_long_eq    = "--help=";
constexpr std::string_view k_version_long    = "--version";
constexpr std::size_t      k_min_name_column = 14;
constexpr std::string_view k_docs_url =
    "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/";

using rocprofsys::rocsys::capability;
using rocprofsys::rocsys::has_capability;
using rocprofsys::rocsys::k_verbs;
using rocprofsys::rocsys::verb_spec;

// One column width for every help row so descriptions stay aligned.
[[nodiscard]] constexpr std::size_t
subcommand_name_column() noexcept
{
    auto width = k_min_name_column;
    for(const auto& spec : k_verbs)
    {
        width = std::max(width, spec.name.size() + 1);
    }
    return width;
}

[[nodiscard]] std::string
make_usage(std::string_view program)
{
    return fmt::format("{} [subcommand] [options] -- <app> [app-args]", program);
}

[[nodiscard]] std::string
help_hint(std::string_view program)
{
    return fmt::format("Usage: {}\nhint: run '{} --help' for available subcommands.",
                       make_usage(program), program);
}

[[nodiscard]] bool
is_help_flag(std::string_view token) noexcept
{
    return token == k_help_short || token == k_help_alt_short || token == k_help_long ||
           token.starts_with(k_help_long_eq);
}

[[nodiscard]] dispatch_kind
tool_kind(const verb_spec& spec) noexcept
{
    return has_capability(spec.caps, capability::in_process) ? dispatch_kind::in_process
                                                             : dispatch_kind::exec_tool;
}

[[nodiscard]] dispatch_result
make_tool_result(const verb_spec* spec, bool strip_subcommand) noexcept
{
    return { .kind             = tool_kind(*spec),
             .spec             = spec,
             .strip_subcommand = strip_subcommand };
}

}  // namespace

dispatch_result
parse_dispatch(int argc, char** argv, std::string_view program)
{
    if(argc <= 1 || is_help_flag(argv[1]))
    {
        return { .kind = dispatch_kind::show_help };
    }

    const auto first = std::string_view{ argv[1] };
    if(first == k_version_long)
    {
        return { .kind = dispatch_kind::show_version };
    }

    const auto found = std::ranges::find(k_verbs, first, &verb_spec::name);
    if(found != k_verbs.end())
    {
        // argv = { rocsys, <verb>, ... }. needs_operand requires a token after the verb.
        if(has_capability(found->caps, capability::needs_operand) && argc < 3)
        {
            return { .kind          = dispatch_kind::error,
                     .error_message = fmt::format(
                         "error: missing application argument\n{}", help_hint(program)) };
        }
        return make_tool_result(found, true);
    }

    if(!first.starts_with('-'))
    {
        return { .kind          = dispatch_kind::error,
                 .error_message = fmt::format("error: unknown subcommand '{}'\n{}", first,
                                              help_hint(program)) };
    }

    // Implicit default: first token is a flag or "--". Keep it in the forwarded argv.
    return make_tool_result(&k_verbs.front(), false);
}

void
print_help(std::ostream& out, std::string_view program)
{
    constexpr auto k_name_column = subcommand_name_column();
    std::string    subcommands;
    for(const auto& spec : k_verbs)
    {
        subcommands +=
            fmt::format("  {:<{}}{}\n", spec.name, k_name_column, spec.description);
    }

    out << fmt::format(
        R"(Usage:
  {usage}

ROCm Systems Profiler.
Experimental: This is a preview of the rocsys command-line tool.
Commands and workflows are subject to change.

Subcommands:
{subcommands})",
        fmt::arg("usage", make_usage(program)), fmt::arg("subcommands", subcommands));
    out << '\n';
    rocprofsys::rocsys::help::write_common_options(
        out, rocprofsys::rocsys::options::option_table());
    out << fmt::format(
        R"(
Examples:
  {program} -- ./app
  {program} profile -- ./app
  {program} instrument -- ./app
  {program} rewrite -- ./app

Use '{program} <subcommand> --help' for subcommand-specific options.

Documentation: {docs}
)",
        fmt::arg("program", program), fmt::arg("docs", k_docs_url));
}

}  // namespace rocprofsys::cli
