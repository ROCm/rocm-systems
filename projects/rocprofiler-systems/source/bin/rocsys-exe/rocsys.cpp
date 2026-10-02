// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"
#include "common/defines.h"
#include "common/path.hpp"
#include "common/tool_runner.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <unistd.h>

namespace
{
constexpr int k_exec_failure_status = 127;

[[nodiscard]] std::string
executable_directory(int argc, char** argv)
{
    const char* invoked = "";
    if(argc > 0 && argv != nullptr && argv[0] != nullptr)
    {
        invoked = argv[0];
    }
    auto exe_dir = rocprofsys::cli::directory_of(invoked);
    if(!exe_dir.empty())
    {
        return exe_dir;
    }
    // Same Linux /proc/self/exe assumption as get_rocprofsys_root().
    return rocprofsys::common::path::parent_path(
        rocprofsys::common::path::realpath("/proc/self/exe"));
}

[[nodiscard]] int
dispatch_in_process(int argc, char** argv, const rocprofsys::cli::dispatch_result& parsed)
{
    if(!parsed.mode.has_value())
    {
        return EXIT_FAILURE;
    }
    auto forwarded = rocprofsys::cli::make_forwarded_argv(
        argc, argv,
        rocprofsys::cli::forward_options{ .strip_subcommand = parsed.strip_subcommand });
    return rocprofsys::common_utils::run_tool(forwarded.argc(), forwarded.argv(),
                                              *parsed.mode);
}

[[nodiscard]] int
dispatch_exec(int argc, char** argv, const rocprofsys::cli::dispatch_result& parsed,
              std::string_view program)
{
    const auto path = rocprofsys::cli::join_sibling_path(executable_directory(argc, argv),
                                                         parsed.binary_name);
    auto       forwarded = rocprofsys::cli::make_forwarded_argv(
        argc, argv,
        rocprofsys::cli::forward_options{ .strip_subcommand = parsed.strip_subcommand,
                                                .argv0_override = parsed.binary_name,
                                                .extra_flag     = parsed.extra_flag });
    execvp(path.c_str(), forwarded.argv());
    std::cerr << program << ": failed to execute '" << path
              << "': " << std::strerror(errno) << '\n';
    return k_exec_failure_status;
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto  parsed  = rocprofsys::cli::parse_dispatch(argc, argv);
    const char* invoked = "rocsys";
    if(argc > 0 && argv != nullptr && argv[0] != nullptr)
    {
        invoked = argv[0];
    }
    const auto program = rocprofsys::cli::program_name(invoked);

    switch(parsed.kind)
    {
        case rocprofsys::cli::dispatch_kind::show_help:
            rocprofsys::cli::print_help(std::cout, program);
            return EXIT_SUCCESS;
        case rocprofsys::cli::dispatch_kind::show_version:
            rocprofsys::cli::print_version(std::cout, program, ROCPROFSYS_VERSION_STRING);
            return EXIT_SUCCESS;
        case rocprofsys::cli::dispatch_kind::error:
            std::cerr << parsed.error_message << '\n';
            return EXIT_FAILURE;
        case rocprofsys::cli::dispatch_kind::in_process:
            return dispatch_in_process(argc, argv, parsed);
        case rocprofsys::cli::dispatch_kind::exec_tool:
            return dispatch_exec(argc, argv, parsed, program);
        default: return EXIT_FAILURE;
    }
}
