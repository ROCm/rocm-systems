// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"
#include "common/defines.h"
#include "common/tool_runner.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace
{
constexpr int k_exec_failure_status = 127;

// Pointers aim into @p args and stay valid for the life of that vector.
[[nodiscard]] std::vector<char*>
c_argv(std::vector<std::string>& args)
{
    std::vector<char*> pointers;
    pointers.reserve(args.size() + 1);
    for(auto& arg : args)
    {
        pointers.push_back(arg.data());
    }
    pointers.push_back(nullptr);
    return pointers;
}

[[nodiscard]] std::string
executable_directory(const char* invoked)
{
    const auto exe_dir = std::filesystem::path{ invoked }.parent_path();
    if(!exe_dir.empty())
    {
        return exe_dir.string();
    }
    // Same Linux /proc/self/exe assumption as get_rocprofsys_root().
    std::error_code error;
    const auto      canon = std::filesystem::canonical("/proc/self/exe", error);
    if(error)
    {
        return std::filesystem::path{ "/proc/self/exe" }.parent_path().string();
    }
    return canon.parent_path().string();
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
    auto pointers = c_argv(forwarded);
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
    const auto  directory = executable_directory(argv[0]);
    const auto  path = (std::filesystem::path{ directory } / spec.binary_name).string();
    auto        forwarded = rocprofsys::cli::make_forwarded_argv(
        argc, argv,
        rocprofsys::cli::forward_options{ .strip_subcommand = parsed.strip_subcommand,
                                                 .argv0_override = spec.binary_name,
                                                 .extra_flag     = spec.extra_flag });
    auto pointers = c_argv(forwarded);
    execvp(path.c_str(), pointers.data());
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
    if(argc > 0)
    {
        invoked = argv[0];
    }
    const auto program = std::filesystem::path{ invoked }.filename().string();

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
