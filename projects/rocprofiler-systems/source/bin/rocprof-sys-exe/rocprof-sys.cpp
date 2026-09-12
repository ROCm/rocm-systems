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
#include <unistd.h>

namespace
{
constexpr int exec_failure_status = 127;
}  // namespace

int
main(int argc, char** argv)
{
    using rocprofsys::cli::directory_of;
    using rocprofsys::cli::dispatch_kind;
    using rocprofsys::cli::join_sibling_path;
    using rocprofsys::cli::make_forwarded_argv;
    using rocprofsys::cli::parse_dispatch;
    using rocprofsys::cli::print_help;
    using rocprofsys::cli::print_version;
    using rocprofsys::cli::program_name;

    const auto parsed = parse_dispatch(argc, argv);
    const auto prog   = program_name(
        (argc > 0 && argv != nullptr && argv[0] != nullptr) ? argv[0] : "rocsys");

    switch(parsed.kind)
    {
        case dispatch_kind::show_help: print_help(std::cout, prog); return EXIT_SUCCESS;
        case dispatch_kind::show_version:
            print_version(std::cout, prog, ROCPROFSYS_VERSION_STRING);
            return EXIT_SUCCESS;
        case dispatch_kind::error:
            std::cerr << parsed.error_message << '\n';
            return EXIT_FAILURE;
        case dispatch_kind::in_process:
        {
            auto fwd = make_forwarded_argv(argc, argv, parsed.strip_subcommand);
            return rocprofsys::common_utils::run_tool(fwd.argc(), fwd.argv(),
                                                      parsed.mode);
        }
        case dispatch_kind::exec_tool:
        {
            auto dir = directory_of(
                (argc > 0 && argv != nullptr && argv[0] != nullptr) ? argv[0] : "");
            if(dir.empty())
            {
                dir = rocprofsys::common::path::parent_path(
                    rocprofsys::common::path::realpath("/proc/self/exe"));
            }
            const auto path = join_sibling_path(dir, parsed.binary_name);
            auto       fwd  = make_forwarded_argv(argc, argv, parsed.strip_subcommand,
                                                  parsed.binary_name);
            execvp(path.c_str(), fwd.argv());
            std::cerr << prog << ": failed to execute '" << path
                      << "': " << std::strerror(errno) << '\n';
            return exec_failure_status;
        }
    }

    return EXIT_FAILURE;
}
