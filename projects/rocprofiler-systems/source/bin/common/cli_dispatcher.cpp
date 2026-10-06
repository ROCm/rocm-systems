// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
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
constexpr int              k_index_after_verb = 2;

[[nodiscard]] constexpr bool
is_injected_flag_present(std::string_view arg, std::string_view flag) noexcept
{
    if(arg == flag)
    {
        return true;
    }
    if(flag != k_output_short)
    {
        return false;
    }
    return arg == k_output_long || arg.starts_with(k_output_long_eq) ||
           arg.starts_with(k_output_short_eq);
}

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

[[nodiscard]] bool
has_payload_args(int argc, bool strip_subcommand) noexcept
{
    return argc > payload_index(strip_subcommand);
}

[[nodiscard]] bool
payload_already_has_flag(int argc, char** argv, int start, std::string_view flag) noexcept
{
    if(flag.empty() || argv == nullptr)
    {
        return false;
    }
    for(int idx = start; idx < argc; ++idx)
    {
        const auto arg = arg_at(argc, argv, idx);
        if(arg.empty() || arg == "--")
        {
            break;
        }
        if(is_injected_flag_present(arg, flag))
        {
            return true;
        }
    }
    return false;
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

void
append_argv0(std::vector<std::string>& args, int argc, char** argv,
             std::string_view replacement)
{
    if(!replacement.empty())
    {
        args.emplace_back(replacement);
        return;
    }
    if(argc > 0 && argv != nullptr && argv[0] != nullptr)
    {
        args.emplace_back(argv[0]);
        return;
    }
    args.emplace_back(k_program_fallback);
}

void
append_payload(std::vector<std::string>& args, int argc, char** argv, int start)
{
    if(argc <= 0 || argv == nullptr)
    {
        return;
    }
    for(int idx = start; idx < argc; ++idx)
    {
        if(argv[idx] != nullptr)
        {
            args.emplace_back(argv[idx]);
        }
    }
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

    if(const auto* spec = find_subcommand(first))
    {
        constexpr bool k_strip_verb = true;
        if(spec->requires_app && !has_payload_args(argc, k_strip_verb))
        {
            return make_error(
                fmt::format("error: missing application argument\n"
                            "Usage: {} [subcommand] [flags] [--] <app> [app-args]\n"
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
    const int                start = payload_index(options.strip_subcommand);
    append_argv0(args, argc, argv, options.argv0_override);
    if(!options.extra_flag.empty() &&
       !payload_already_has_flag(argc, argv, start, options.extra_flag))
    {
        args.emplace_back(options.extra_flag);
    }
    append_payload(args, argc, argv, start);
    return args;
}

void
print_help(std::ostream& out, std::string_view program)
{
    out << "Usage:\n"
        << "  " << program << " [subcommand] [flags] [--] <app> [app-args]\n"
        << "\n"
        << "ROCm Systems Profiler unified command-line entry point.\n"
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
        << " <subcommand> --help' for subcommand-specific options.\n";
}

void
print_version(std::ostream& out, std::string_view program, std::string_view version)
{
    out << program << " version " << version << '\n';
}

}  // namespace rocprofsys::cli
