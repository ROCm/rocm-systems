// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>

namespace rocprofsys::cli
{
namespace
{
[[nodiscard]] constexpr bool
is_help_flag(std::string_view arg) noexcept
{
    return arg == "-h" || arg == "-?" || arg == "--help" ||
           arg.compare(0, 7, "--help=") == 0;
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

[[nodiscard]] std::string
help_hint(std::string_view program)
{
    std::ostringstream oss;
    oss << "Run '" << program << " --help' for available subcommands.";
    return oss.str();
}

[[nodiscard]] std::string_view
arg_at(int argc, char** argv, int index) noexcept
{
    if(index < 0 || index >= argc || argv == nullptr || argv[index] == nullptr) return {};
    return argv[index];
}

[[nodiscard]] int
payload_begin(bool strip_subcommand) noexcept
{
    return strip_subcommand ? 2 : 1;
}

[[nodiscard]] bool
has_payload_args(int argc, bool strip_subcommand) noexcept
{
    return argc > payload_begin(strip_subcommand);
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
from_spec(const subcommand_spec& spec, bool strip_subcommand)
{
    dispatch_result result;
    result.kind = spec.in_process ? dispatch_kind::in_process : dispatch_kind::exec_tool;
    result.mode = spec.mode;
    result.binary_name      = spec.binary_name;
    result.subcommand_name  = spec.name;
    result.strip_subcommand = strip_subcommand;
    return result;
}
}  // namespace

std::string_view
program_name(std::string_view argv0) noexcept
{
    if(argv0.empty()) return "rocsys";
    const auto pos = argv0.find_last_of('/');
    if(pos == std::string_view::npos) return argv0;
    auto name = argv0.substr(pos + 1);
    return name.empty() ? std::string_view{ "rocsys" } : name;
}

std::string
directory_of(std::string_view argv0)
{
    const auto pos = argv0.find_last_of('/');
    if(pos == std::string_view::npos) return {};
    if(pos == 0) return "/";
    return std::string{ argv0.substr(0, pos) };
}

std::string
join_sibling_path(std::string_view directory, std::string_view binary_name)
{
    if(directory.empty()) return std::string{ binary_name };
    if(directory == "/") return std::string{ "/" } + std::string{ binary_name };
    return std::string{ directory } + '/' + std::string{ binary_name };
}

dispatch_result
parse_dispatch(int argc, char** argv)
{
    const auto prog = program_name(arg_at(argc, argv, 0));

    if(argc <= 1)
    {
        dispatch_result result;
        result.kind = dispatch_kind::show_help;
        return result;
    }

    const auto first = arg_at(argc, argv, 1);
    if(first.empty())
        return make_error(std::string{ prog } + ": unknown subcommand ''\n" +
                          help_hint(prog));

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
        const bool strip = true;
        if(spec->requires_app && !has_payload_args(argc, strip))
        {
            std::ostringstream oss;
            oss << prog << ": missing application argument\n"
                << "Usage: " << prog << " [subcommand] [flags] [--] <app> [app-args]\n"
                << help_hint(prog);
            return make_error(oss.str());
        }
        return from_spec(*spec, strip);
    }

    if(!is_flag(first))
    {
        std::ostringstream oss;
        oss << prog << ": unknown subcommand '" << first << "'\n" << help_hint(prog);
        return make_error(oss.str());
    }

    // Implicit profile: first token is a flag or "--".
    return from_spec(subcommands[0], false);
}

forwarded_argv
make_forwarded_argv(int argc, char** argv, bool strip_subcommand,
                    std::string_view argv0_override)
{
    forwarded_argv result;
    const int      start = payload_begin(strip_subcommand);
    result.ptrs.reserve(static_cast<std::size_t>(argc > 0 ? argc + 1 : 2));

    if(!argv0_override.empty())
    {
        result.argv0_storage.assign(argv0_override.begin(), argv0_override.end());
        result.ptrs.push_back(result.argv0_storage.data());
    }
    else if(argc > 0 && argv != nullptr && argv[0] != nullptr)
    {
        result.ptrs.push_back(argv[0]);
    }
    else
    {
        result.argv0_storage = "rocsys";
        result.ptrs.push_back(result.argv0_storage.data());
    }

    if(argc > 0 && argv != nullptr)
    {
        for(int i = start; i < argc; ++i)
        {
            if(argv[i] != nullptr) result.ptrs.push_back(argv[i]);
        }
    }
    result.ptrs.push_back(nullptr);
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

    constexpr std::size_t name_width = 14;
    for(const auto& spec : subcommands)
    {
        out << "  " << spec.name;
        const auto pad =
            spec.name.size() >= name_width ? 1 : name_width - spec.name.size();
        out << std::string(pad, ' ') << spec.description << '\n';
    }

    out << "\n"
        << "Examples:\n"
        << "  " << program << " -- ./app\n"
        << "  " << program << " profile -- ./app\n"
        << "  " << program << " trace -- ./app\n"
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
