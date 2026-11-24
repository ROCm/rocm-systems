// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_dispatcher.hpp"
#include "common/common_utils.hpp"
#include "common/defines.h"
#include "common/path.hpp"
#include "common/tool_runner.hpp"
#include "rocsys/options.hpp"

#include "backends/rocprofiler_sdk/backend.hpp"
#include "backends/rocprofiler_sdk/wrapper.hpp"
#include "core/sdk/tracing-config-deps.hpp"
#include "core/sdk/tracing-config.hpp"

#include <timemory/utility/argparse.hpp>

#include <stdlib.h>  // POSIX ::setenv

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace
{
constexpr int k_exec_failure_status = 127;

// enable_version exits on success. The exit below runs only if that action did not.
[[noreturn]] int
exit_with_version(std::string_view program)
{
    auto name   = std::string{ program };
    auto parser = tim::argparse::argument_parser{ name };
    parser.enable_version(name, ROCPROFSYS_ARGPARSE_VERSION_INFO);

    auto flag = std::string{ "--version" };
    auto argv = std::array<char*, 2>{ name.data(), flag.data() };
    // enable_version's action calls exit(EXIT_SUCCESS) from parse_args.
    parser.parse_args(2, argv.data());
    std::exit(EXIT_FAILURE);
}

[[nodiscard]] std::string
executable_directory()
{
    // Resolve symlinks. A /usr/bin/rocsys link must find siblings next to the
    // real binary, not next to the link. Same /proc/self/exe assumption as
    // get_rocprofsys_root().
    std::error_code error;
    const auto      canon = std::filesystem::canonical("/proc/self/exe", error);
    if(error)
    {
        return {};
    }
    return rocprofsys::common::path::parent_path(canon.string());
}

[[nodiscard]] int
dispatch_in_process(int argc, char** argv, const rocprofsys::cli::dispatch_result& parsed)
{
    if(parsed.spec == nullptr || !parsed.spec->mode.has_value())
    {
        return EXIT_FAILURE;
    }
    auto forwarded = rocprofsys::cli::make_forwarded_argv(
        argc, argv,
        rocprofsys::cli::forward_options{ .strip_subcommand = parsed.strip_subcommand });
    auto pointers = rocprofsys::common_utils::to_c_argv(forwarded);
    return rocprofsys::common_utils::run_tool(static_cast<int>(forwarded.size()),
                                              pointers.data(), *parsed.spec->mode);
}

[[nodiscard]] int
dispatch_exec(int argc, char** argv, const rocprofsys::cli::dispatch_result& parsed,
              std::string_view program)
{
    if(parsed.spec == nullptr)
    {
        return k_exec_failure_status;
    }
    const auto& spec      = *parsed.spec;
    const auto  directory = executable_directory();
    if(directory.empty())
    {
        std::cerr << program << ": failed to resolve install directory\n";
        return k_exec_failure_status;
    }
    const auto path = (std::filesystem::path{ directory } / spec.binary_name).string();
    auto       forwarded = rocprofsys::cli::make_forwarded_argv(
        argc, argv,
        rocprofsys::cli::forward_options{ .strip_subcommand = parsed.strip_subcommand,
                                                .argv0_override = spec.binary_name,
                                                .extra_flag     = spec.extra_flag });
    auto pointers = rocprofsys::common_utils::to_c_argv(forwarded);
    // path is absolute (sibling of this binary); no PATH search.
    execv(path.c_str(), pointers.data());
    const int exec_error = errno;
    std::cerr << program << ": failed to execute '" << path
              << "': " << std::strerror(exec_error) << '\n';
    return k_exec_failure_status;
}

/// The ROCPROFSYS_ROCM_DOMAINS values the installed rocprofiler-sdk takes, the list the
/// tool checks the variable against.
[[nodiscard]] std::vector<std::string>
supported_rocm_domains()
{
    using tracing_config_t = rocprofsys::rocprofiler_sdk::tracing_config<
        rocprofsys::backends::rocprofiler_sdk::backend<
            rocprofsys::rocprofiler_sdk::wrapper>,
        rocprofsys::rocprofiler_sdk::default_externals>;
    return tracing_config_t::get_domain_choices();
}

/// Start the tool of @p parsed with @p argv.
[[nodiscard]] int
start_tool(int argc, char** argv, const rocprofsys::cli::dispatch_result& parsed,
           std::string_view program)
{
    return parsed.spec->mode.has_value() ? dispatch_in_process(argc, argv, parsed)
                                         : dispatch_exec(argc, argv, parsed, program);
}

/// Read the rocsys options, set the variables they write, and start the tool with the
/// command line that is left. A subcommand that takes no rocsys options gets the whole
/// command line.
[[nodiscard]] int
run_subcommand(int argc, char** argv, const rocprofsys::cli::dispatch_result& parsed,
               std::string_view program)
{
    if(!parsed.spec->takes_rocsys_options)
    {
        return start_tool(argc, argv, parsed, program);
    }
    const int  first   = parsed.strip_subcommand ? 2 : 1;
    const auto options = rocprofsys::rocsys::parse_options(
        std::vector<std::string>(argv + first, argv + argc), supported_rocm_domains());
    if(!options.errors.empty())
    {
        for(const auto& message : options.errors)
        {
            std::cerr << "error: " << message << '\n';
        }
        std::cerr << "hint: run '" << program << " --help' for the rocsys options.\n";
        return EXIT_FAILURE;
    }
    for(const auto& [name, value] : options.environment)
    {
        if(::setenv(name.c_str(), value.c_str(), 1) != 0)
        {
            std::cerr << program << ": failed to set " << name << '\n';
            return EXIT_FAILURE;
        }
    }

    // The tool sees the command line without the rocsys options.
    auto tool_argv = std::vector<std::string>(argv, argv + first);
    tool_argv.insert(tool_argv.end(), options.tool_args.begin(), options.tool_args.end());
    if(!options.command.empty())
    {
        tool_argv.emplace_back("--");
        tool_argv.insert(tool_argv.end(), options.command.begin(), options.command.end());
    }
    auto pointers = rocprofsys::common_utils::to_c_argv(tool_argv);
    return start_tool(static_cast<int>(tool_argv.size()), pointers.data(), parsed,
                      program);
}

/// Print @p text, or @p error when the catalog has no such help.
[[nodiscard]] int
print_help_text(const std::optional<std::string>& text, std::string_view error)
{
    if(!text)
    {
        std::cerr << error << '\n';
        return EXIT_FAILURE;
    }
    std::cout << *text;
    return EXIT_SUCCESS;
}

/// The rocsys help, or the help topic after "--help=": a category, or "all".
[[nodiscard]] int
print_rocsys_help(std::string_view program, std::string_view topic)
{
    if(topic.empty())
    {
        rocprofsys::cli::print_help(std::cout, program,
                                    rocprofsys::rocsys::options_summary());
        return EXIT_SUCCESS;
    }
    const auto error = "error: unknown help topic '" + std::string{ topic } +
                       "'\nhint: run '" + std::string{ program } +
                       " --help' for the categories.";
    return print_help_text(
        rocprofsys::rocsys::options_help(topic, supported_rocm_domains()), error);
}

/// One rocsys option in full, as "rocsys traces --help" asks for "--traces".
[[nodiscard]] int
print_option_help(const rocprofsys::cli::dispatch_result& parsed)
{
    return print_help_text(
        rocprofsys::rocsys::options_help("--" + std::string{ parsed.topic },
                                         supported_rocm_domains()),
        parsed.error_message);
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto program = std::filesystem::path{ argv[0] }.filename().string();
    const auto parsed  = rocprofsys::cli::parse_dispatch(argc, argv, program);

    switch(parsed.kind)
    {
        case rocprofsys::cli::dispatch_kind::show_help:
            return print_rocsys_help(program, parsed.topic);
        case rocprofsys::cli::dispatch_kind::show_subcommand_help:
            rocprofsys::cli::print_subcommand_help(std::cout, program, *parsed.spec,
                                                   rocprofsys::rocsys::options_summary());
            return EXIT_SUCCESS;
        case rocprofsys::cli::dispatch_kind::show_option_help:
            return print_option_help(parsed);
        case rocprofsys::cli::dispatch_kind::show_version:
            return exit_with_version(program);
        case rocprofsys::cli::dispatch_kind::error:
            std::cerr << parsed.error_message << '\n';
            return EXIT_FAILURE;
        case rocprofsys::cli::dispatch_kind::run_subcommand:
            return run_subcommand(argc, argv, parsed, program);
        default: return EXIT_FAILURE;
    }
}
