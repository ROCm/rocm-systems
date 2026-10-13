// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"

#include <fmt/base.h>
#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::cli
{
namespace
{
constexpr std::string_view k_help_short      = "-h";
constexpr std::string_view k_help_alt_short  = "-?";
constexpr std::string_view k_help_long       = "--help";
constexpr std::string_view k_help_long_eq    = "--help=";
constexpr std::string_view k_version_long    = "--version";
constexpr std::string_view k_output_short    = "-o";
constexpr std::string_view k_output_long     = "--output";
constexpr std::string_view k_output_long_eq  = "--output=";
constexpr std::string_view k_output_short_eq = "-o=";
constexpr std::size_t      k_min_name_column = 14;
constexpr std::string_view k_docs_url =
    "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/";

// One column width for every help row so descriptions stay aligned.
[[nodiscard]] constexpr std::size_t
subcommand_name_column() noexcept
{
    auto width = k_min_name_column;
    for(const auto& spec : k_subcommands)
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

[[nodiscard]] constexpr int
payload_index(bool strip_subcommand) noexcept
{
    constexpr int k_after_argv0      = 1;
    constexpr int k_index_after_verb = 2;
    return strip_subcommand ? k_index_after_verb : k_after_argv0;
}

[[nodiscard]] bool
is_help_flag(std::string_view token) noexcept
{
    return token == k_help_short || token == k_help_alt_short || token == k_help_long ||
           token.starts_with(k_help_long_eq);
}

[[nodiscard]] dispatch_kind
tool_kind(const subcommand_spec& spec) noexcept
{
    return spec.mode.has_value() ? dispatch_kind::in_process : dispatch_kind::exec_tool;
}

[[nodiscard]] dispatch_result
make_tool_result(const subcommand_spec* spec, bool strip_subcommand) noexcept
{
    return { .kind             = tool_kind(*spec),
             .spec             = spec,
             .strip_subcommand = strip_subcommand };
}

// Scan the tool's own options (tokens after argv0 / the verb, before "--").
// When @p flag is "-o", also treat --output, --output=…, and -o=… as present.
[[nodiscard]] bool
option_present_before_separator(int argc, char** argv, int start, std::string_view flag)
{
    for(int idx = start; idx < argc; ++idx)
    {
        const auto arg = std::string_view{ argv[idx] };
        if(arg.empty() || arg == "--")
        {
            return false;
        }
        const auto same_flag = arg == flag;
        const auto output_alias =
            flag == k_output_short &&
            (arg == k_output_long || arg.starts_with(k_output_long_eq) ||
             arg.starts_with(k_output_short_eq));
        if(same_flag || output_alias)
        {
            return true;
        }
    }
    return false;
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

    const auto found = std::ranges::find(k_subcommands, first, &subcommand_spec::name);
    if(found != k_subcommands.end())
    {
        // argv = { rocsys, <verb>, ... }: requires_app needs a token after the verb.
        if(found->requires_app && argc < 3)
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
    const auto* chosen = &k_subcommands.front();
    return make_tool_result(chosen, false);
}

std::vector<std::string>
make_forwarded_argv(int argc, char** argv, forward_options options)
{
    std::vector<std::string> args;
    const int                start = payload_index(options.strip_subcommand);
    if(!options.argv0_override.empty())
    {
        args.emplace_back(options.argv0_override);
    }
    else
    {
        args.emplace_back(argv[0]);
    }
    if(!options.extra_flag.empty() &&
       !option_present_before_separator(argc, argv, start, options.extra_flag))
    {
        args.emplace_back(options.extra_flag);
    }
    if(start < argc)
    {
        args.insert(args.end(), argv + start, argv + argc);
    }
    return args;
}

void
print_help(std::ostream& out, std::string_view program)
{
    constexpr auto k_name_column = subcommand_name_column();
    std::string    subcommands;
    for(const auto& spec : k_subcommands)
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
{subcommands}
Examples:
  {program} -- ./app
  {program} profile -- ./app
  {program} instrument -- ./app
  {program} rewrite -- ./app

Use '{program} <subcommand> --help' for subcommand-specific options.

Documentation: {docs}
)",
        fmt::arg("usage", make_usage(program)), fmt::arg("subcommands", subcommands),
        fmt::arg("program", program), fmt::arg("docs", k_docs_url));
}

}  // namespace rocprofsys::cli
