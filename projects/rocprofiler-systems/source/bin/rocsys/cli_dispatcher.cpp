// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_dispatcher.hpp"

#include <fmt/base.h>
#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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
is_bare_help_flag(std::string_view token) noexcept
{
    return token == k_help_short || token == k_help_alt_short || token == k_help_long;
}

[[nodiscard]] bool
is_topic_help_flag(std::string_view token) noexcept
{
    return token.starts_with(k_help_long_eq);
}

[[nodiscard]] bool
is_help_flag(std::string_view token) noexcept
{
    return is_bare_help_flag(token) || is_topic_help_flag(token);
}

/// The topic of "--help=<topic>", or nothing for a bare help flag.
[[nodiscard]] std::string_view
help_topic(std::string_view token) noexcept
{
    return token.starts_with(k_help_long_eq) ? token.substr(k_help_long_eq.size())
                                             : std::string_view{};
}

/// How to see the rocsys options in full.
[[nodiscard]] std::string
details_hint(std::string_view program)
{
    return fmt::format(
        "Use '{0} <option> --help' for one rocsys option in full, such as\n"
        "'{0} traces --help', '{0} --help=<category>' for one category,\n"
        "and '{0} --help=all' for every option.\n",
        program);
}

[[nodiscard]] std::string_view
as_text(const char* arg) noexcept
{
    return arg;
}

/// The arguments from @p start on.
[[nodiscard]] std::span<char*>
payload(int argc, char** argv, int start)
{
    return std::span{ argv, static_cast<std::size_t>(argc) }.subspan(
        static_cast<std::size_t>(std::min(start, argc)));
}

/// The arguments in @p args before "--": the rocsys and tool options.
[[nodiscard]] std::span<char*>
before_separator(std::span<char*> args)
{
    return { args.begin(), std::ranges::find(args, std::string_view{ "--" }, as_text) };
}

// "--help=<topic>" is the tool's topic help, so only the bare help flags match.
[[nodiscard]] bool
asks_for_help(int argc, char** argv, int start)
{
    return std::ranges::any_of(before_separator(payload(argc, argv, start)),
                               is_bare_help_flag, as_text);
}

[[nodiscard]] dispatch_result
make_tool_result(const subcommand_spec* spec, bool strip_subcommand, int argc,
                 char** argv)
{
    const auto kind = asks_for_help(argc, argv, payload_index(strip_subcommand))
                          ? dispatch_kind::show_subcommand_help
                          : dispatch_kind::run_subcommand;
    return { .kind = kind, .spec = spec, .strip_subcommand = strip_subcommand };
}

/// The command before its options: "rocsys", or "rocsys python" when the subcommand is
/// typed.
[[nodiscard]] std::string
command_name(std::string_view program, const subcommand_spec& spec, bool strip_subcommand)
{
    return strip_subcommand ? fmt::format("{} {}", program, spec.name)
                            : std::string{ program };
}

/// "put '--' before the application", showing @p app after "--".
[[nodiscard]] std::string
separator_error(std::string_view command, std::string_view app, std::string_view program)
{
    return fmt::format(
        "error: put '--' before the application: {} [options] -- {} [app-args]\n{}",
        command, app, help_hint(program));
}

/// @p result, or an error when it runs a subcommand with requires_app without "--", or
/// one without requires_app with "--". A tool help topic, as in
/// "rocsys profile --help=all", runs no application.
[[nodiscard]] dispatch_result
check_separator(dispatch_result result, int argc, char** argv, std::string_view program)
{
    if(result.kind != dispatch_kind::run_subcommand)
    {
        return result;
    }
    const auto& spec    = *result.spec;
    const auto  args    = payload(argc, argv, payload_index(result.strip_subcommand));
    const auto  options = before_separator(args);
    const bool  has_separator = options.size() < args.size();
    const auto  command       = command_name(program, spec, result.strip_subcommand);
    if(spec.requires_app && !has_separator &&
       !std::ranges::any_of(options, is_topic_help_flag, as_text))
    {
        return { .kind          = dispatch_kind::error,
                 .error_message = separator_error(command, "<app>", program) };
    }
    if(!spec.requires_app && has_separator)
    {
        return { .kind = dispatch_kind::error,
                 .error_message =
                     fmt::format("error: {0} runs no application, so it takes no '--'\n"
                                 "hint: run '{0} --help' for its options.",
                                 command) };
    }
    return result;
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
    if(argc <= 1)
    {
        return { .kind = dispatch_kind::show_help };
    }

    const auto first = std::string_view{ argv[1] };
    if(is_help_flag(first))
    {
        return { .kind = dispatch_kind::show_help, .topic = help_topic(first) };
    }
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
        // A plain word right after the verb is an application given without "--".
        // Refuse it before routing help: rocsys would otherwise read its options, and
        // --help, from the application's arguments.
        if(found->requires_app && !std::string_view{ argv[2] }.starts_with('-'))
        {
            return { .kind          = dispatch_kind::error,
                     .error_message = separator_error(command_name(program, *found, true),
                                                      argv[2], program) };
        }
        return check_separator(make_tool_result(found, true, argc, argv), argc, argv,
                               program);
    }

    if(!first.starts_with('-'))
    {
        auto message =
            fmt::format("error: unknown subcommand '{}'\n{}", first, help_hint(program));
        if(asks_for_help(argc, argv, payload_index(true)))
        {
            return { .kind          = dispatch_kind::show_option_help,
                     .topic         = first,
                     .error_message = std::move(message) };
        }
        return { .kind = dispatch_kind::error, .error_message = std::move(message) };
    }

    // Implicit default: first token is a flag or "--". Keep it in the forwarded argv.
    const auto* chosen = &k_subcommands.front();
    return check_separator(make_tool_result(chosen, false, argc, argv), argc, argv,
                           program);
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
print_help(std::ostream& out, std::string_view program, std::string_view options_text)
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
rocsys options:
{options}Examples:
  {program} -- ./app
  {program} profile -- ./app
  {program} --traces hip mpi -- ./app
  {program} instrument -- ./app
  {program} rewrite -- ./app

Use '{program} <subcommand> --help' for subcommand-specific options.
{details}
Documentation: {docs}
)",
        fmt::arg("usage", make_usage(program)), fmt::arg("subcommands", subcommands),
        fmt::arg("options", options_text), fmt::arg("details", details_hint(program)),
        fmt::arg("program", program), fmt::arg("docs", k_docs_url));
}

void
print_subcommand_help(std::ostream& out, std::string_view program,
                      const subcommand_spec& spec, std::string_view options_text)
{
    out << fmt::format(
        R"(Usage:
  {program} {name} [options]

{description}. Runs {runs}.

{options}{tool}
  Every other option is passed to {tool} unchanged.
  Run '{tool} --help' to list them.
)",
        fmt::arg("program", program), fmt::arg("name", spec.name),
        fmt::arg("description", spec.description),
        fmt::arg("runs", spec.runs.empty() ? spec.binary_name : spec.runs),
        fmt::arg("tool", spec.binary_name),
        fmt::arg("options",
                 spec.takes_rocsys_options ? options_text : std::string_view{}));
}

}  // namespace rocprofsys::cli
