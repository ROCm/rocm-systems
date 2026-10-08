// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iterator>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::cli
{
namespace
{
constexpr std::string_view k_program_fallback = "rocsys";
constexpr std::string_view k_output_short     = "-o";
constexpr std::string_view k_output_long      = "--output";
constexpr std::string_view k_output_long_eq   = "--output=";
constexpr std::string_view k_output_short_eq  = "-o=";
constexpr std::size_t      k_name_column      = 14;
constexpr std::string_view k_docs_url =
    "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/";
constexpr int k_index_after_verb = 2;

[[nodiscard]] std::string
help_hint(std::string_view program)
{
    return fmt::format("hint: run '{} --help' for available subcommands.", program);
}

[[nodiscard]] std::string_view
arg_at(int argc, char** argv, int index) noexcept
{
    if(index < 0 || index >= argc || argv == nullptr || argv[index] == nullptr)
    {
        return {};
    }
    return argv[index];
}

[[nodiscard]] constexpr int
payload_index(bool strip_subcommand) noexcept
{
    constexpr int k_after_argv0 = 1;
    return strip_subcommand ? k_index_after_verb : k_after_argv0;
}

dispatch_result
make_error(std::string message)
{
    dispatch_result result;
    result.kind          = dispatch_kind::error;
    result.error_message = std::move(message);
    return result;
}

dispatch_result
unknown_subcommand_error(std::string_view program, std::string_view token)
{
    return make_error(
        fmt::format("error: unknown subcommand '{}'\n{}", token, help_hint(program)));
}

}  // namespace

dispatch_result
parse_dispatch(int argc, char** argv)
{
    const auto program =
        std::filesystem::path{ arg_at(argc, argv, 0) }.filename().string();

    if(argc <= 1)
    {
        dispatch_result result;
        result.kind = dispatch_kind::show_help;
        return result;
    }

    const auto first = arg_at(argc, argv, 1);
    if(first.empty())
    {
        return unknown_subcommand_error(program, first);
    }
    if(first == "-h" || first == "-?" || first == "--help" ||
       first.starts_with("--help="))
    {
        dispatch_result result;
        result.kind = dispatch_kind::show_help;
        return result;
    }
    if(first == "--version")
    {
        dispatch_result result;
        result.kind = dispatch_kind::show_version;
        return result;
    }

    const auto found = std::ranges::find(k_subcommands, first, &subcommand_spec::name);
    if(found != k_subcommands.end())
    {
        const auto*    spec         = found;
        constexpr bool k_strip_verb = true;
        if(spec->requires_app && argc <= payload_index(k_strip_verb))
        {
            return make_error(
                fmt::format("error: missing application argument\n"
                            "Usage: {} [subcommand] [options] -- <app> [app-args]\n"
                            "{}",
                            program, help_hint(program)));
        }
        dispatch_result result;
        result.spec = spec;
        result.kind =
            spec->mode.has_value() ? dispatch_kind::in_process : dispatch_kind::exec_tool;
        result.strip_subcommand = k_strip_verb;
        return result;
    }

    if(first.empty() || first.front() != '-')
    {
        return unknown_subcommand_error(program, first);
    }

    // Implicit default: first token is a flag or "--".
    constexpr bool  k_keep_leading_flag = false;
    const auto&     chosen              = k_subcommands.front();
    dispatch_result result;
    result.spec = &chosen;
    result.kind =
        chosen.mode.has_value() ? dispatch_kind::in_process : dispatch_kind::exec_tool;
    result.strip_subcommand = k_keep_leading_flag;
    return result;
}

std::vector<std::string>
make_forwarded_argv(int argc, char** argv, forward_options options)
{
    std::vector<std::string> args;
    const int                start   = payload_index(options.strip_subcommand);
    const auto               invoked = arg_at(argc, argv, 0);
    if(!options.argv0_override.empty())
    {
        args.emplace_back(options.argv0_override);
    }
    else if(!invoked.empty())
    {
        args.emplace_back(invoked);
    }
    else
    {
        args.emplace_back(k_program_fallback);
    }
    auto already_present = false;
    if(!options.extra_flag.empty() && argv != nullptr)
    {
        for(int idx = start; idx < argc; ++idx)
        {
            const auto arg = arg_at(argc, argv, idx);
            if(arg.empty() || arg == "--")
            {
                break;
            }
            const auto same_flag = arg == options.extra_flag;
            const auto output_alias =
                options.extra_flag == k_output_short &&
                (arg == k_output_long || arg.starts_with(k_output_long_eq) ||
                 arg.starts_with(k_output_short_eq));
            if(same_flag || output_alias)
            {
                already_present = true;
                break;
            }
        }
    }
    if(!options.extra_flag.empty() && !already_present)
    {
        args.emplace_back(options.extra_flag);
    }
    if(argc > 0 && argv != nullptr)
    {
        const auto payload = argv + std::min(start, argc);
        std::copy_if(payload, argv + argc, std::back_inserter(args),
                     [](const char* arg) { return arg != nullptr; });
    }
    return args;
}

void
print_help(std::ostream& out, std::string_view program)
{
    out << "Usage:\n"
        << "  " << program << " [subcommand] [options] -- <app> [app-args]\n"
        << "\n"
        << "ROCm Systems Profiler.\n"
        << "Experimental: This is a preview of the rocsys command-line tool.\n"
        << "Commands and workflows are subject to change.\n"
        << "\n"
        << "Subcommands:\n";

    for(const auto& spec : k_subcommands)
    {
        const auto width = std::max(spec.name.size() + 1, k_name_column);
        out << fmt::format("  {:<{}}{}\n", spec.name, width, spec.description);
    }

    out << "\n"
        << "Examples:\n"
        << "  " << program << " -- ./app\n"
        << "  " << program << " profile -- ./app\n"
        << "  " << program << " instrument -- ./app\n"
        << "  " << program << " rewrite -- ./app\n"
        << "\n"
        << "Use '" << program
        << " <subcommand> --help' for subcommand-specific options.\n"
        << "\n"
        << "Documentation: " << k_docs_url << "\n";
}

}  // namespace rocprofsys::cli
