// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"
#include "common/common_utils.hpp"
#include "common/defines.h"
#include "common/path.hpp"
#include "common/tool_runner.hpp"

#include <timemory/utility/argparse.hpp>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>

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
}  // namespace

int
main(int argc, char** argv)
{
    const auto program = std::filesystem::path{ argv[0] }.filename().string();
    const auto parsed  = rocprofsys::cli::parse_dispatch(argc, argv, program);

    switch(parsed.kind)
    {
        case rocprofsys::cli::dispatch_kind::show_help:
            rocprofsys::cli::print_help(std::cout, program);
            return EXIT_SUCCESS;
        case rocprofsys::cli::dispatch_kind::show_version:
            return exit_with_version(program);
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
