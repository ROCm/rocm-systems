// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

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
is_help_flag(std::string_view arg) noexcept
{
    return arg == "-h" || arg == "-?" || arg == "--help" || arg.starts_with("--help=");
}

[[nodiscard]] constexpr bool
is_version_flag(std::string_view arg) noexcept
{
    return arg == "--version";
}

[[nodiscard]] constexpr bool
is_flag(std::string_view arg) noexcept
{
    return !arg.empty() && arg.front() == '-';
}

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
    std::ostringstream oss;
    oss << "hint: run '" << program << " --help' for available subcommands.";
    return oss.str();
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
    std::ostringstream oss;
    oss << "error: unknown subcommand '" << token << "'\n" << help_hint(program);
    return make_error(oss.str());
}

dispatch_result
missing_app_error(std::string_view program)
{
    std::ostringstream oss;
    oss << "error: missing application argument\n"
        << "Usage: " << program << " [subcommand] [flags] [--] <app> [app-args]\n"
        << help_hint(program);
    return make_error(oss.str());
}

dispatch_result
from_spec(const subcommand_spec& spec, bool strip_subcommand)
{
    dispatch_result result;
    result.kind = spec.in_process ? dispatch_kind::in_process : dispatch_kind::exec_tool;
    result.mode = spec.mode;
    result.binary_name      = spec.binary_name;
    result.subcommand_name  = spec.name;
    result.strip_subcommand = strip_subcommand;
    result.extra_flag       = spec.extra_flag;
    return result;
}

void
append_argv0(forwarded_argv& result, int argc, char** argv, std::string_view replacement)
{
    if(!replacement.empty())
    {
        result.args.emplace_back(replacement);
        return;
    }
    if(argc > 0 && argv != nullptr && argv[0] != nullptr)
    {
        result.args.emplace_back(argv[0]);
        return;
    }
    result.args.emplace_back(k_program_fallback);
}

void
append_payload(forwarded_argv& result, int argc, char** argv, int start)
{
    if(argc <= 0 || argv == nullptr)
    {
        return;
    }
    for(int idx = start; idx < argc; ++idx)
    {
        if(argv[idx] != nullptr)
        {
            result.args.emplace_back(argv[idx]);
        }
    }
}
}  // namespace

void
forwarded_argv::bind() noexcept
{
    ptrs.clear();
    ptrs.reserve(args.size() + 1);
    for(auto& entry : args)
    {
        ptrs.push_back(entry.data());
    }
    ptrs.push_back(nullptr);
}

forwarded_argv::forwarded_argv(const forwarded_argv& other)
: args(other.args)
{
    bind();
}

forwarded_argv&
forwarded_argv::operator=(const forwarded_argv& other)
{
    if(this != &other)
    {
        args = other.args;
        bind();
    }
    return *this;
}

forwarded_argv::forwarded_argv(forwarded_argv&& other) noexcept
: args(std::move(other.args))
{
    bind();
    other.ptrs.clear();
}

forwarded_argv&
forwarded_argv::operator=(forwarded_argv&& other) noexcept
{
    if(this != &other)
    {
        args = std::move(other.args);
        bind();
        other.ptrs.clear();
    }
    return *this;
}

std::string_view
program_name(std::string_view argv0) noexcept
{
    if(argv0.empty())
    {
        return k_program_fallback;
    }
    const auto pos = argv0.find_last_of('/');
    if(pos == std::string_view::npos)
    {
        return argv0;
    }
    const auto name = argv0.substr(pos + 1);
    if(name.empty())
    {
        return k_program_fallback;
    }
    return name;
}

std::string
directory_of(std::string_view argv0)
{
    const auto pos = argv0.find_last_of('/');
    if(pos == std::string_view::npos)
    {
        return {};
    }
    if(pos == 0)
    {
        return "/";
    }
    return std::string{ argv0.substr(0, pos) };
}

std::string
join_sibling_path(std::string_view directory, std::string_view binary_name)
{
    if(directory.empty())
    {
        return std::string{ binary_name };
    }
    if(directory == "/")
    {
        return std::string{ "/" } + std::string{ binary_name };
    }
    return std::string{ directory } + '/' + std::string{ binary_name };
}

dispatch_result
parse_dispatch(int argc, char** argv)
{
    const auto program = program_name(arg_at(argc, argv, 0));

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
    if(is_help_flag(first))
    {
        dispatch_result result;
        result.kind = dispatch_kind::show_help;
        return result;
    }
    if(is_version_flag(first))
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
            return missing_app_error(program);
        }
        return from_spec(*spec, k_strip_verb);
    }

    if(!is_flag(first))
    {
        return unknown_subcommand_error(program, first);
    }

    // Implicit default: first token is a flag or "--".
    constexpr bool k_keep_leading_flag = false;
    return from_spec(default_subcommand(), k_keep_leading_flag);
}

forwarded_argv
make_forwarded_argv(int argc, char** argv, forward_options options)
{
    forwarded_argv result;
    const int      start = payload_index(options.strip_subcommand);
    append_argv0(result, argc, argv, options.argv0_override);
    if(!options.extra_flag.empty() &&
       !payload_already_has_flag(argc, argv, start, options.extra_flag))
    {
        result.args.emplace_back(options.extra_flag);
    }
    append_payload(result, argc, argv, start);
    result.bind();
    return result;
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
        out << "  " << spec.name;
        const auto name_size = spec.name.size();
        const auto pad       = name_size >= k_name_column ? 1 : k_name_column - name_size;
        out << std::string(pad, ' ') << spec.description << '\n';
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
