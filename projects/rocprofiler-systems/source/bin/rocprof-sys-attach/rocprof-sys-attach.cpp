// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "common/path.hpp"
#include "logger/debug.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <rocprofiler-sdk-rocattach/rocattach.h>

namespace
{
struct attach_options
{
    int                      pid = -1;
    std::string              output_path;
    std::vector<std::string> profile_format;
};

void
print_usage(const char* prog_name)
{
    std::cout << "Usage: " << prog_name << " -p <pid> [OPTIONS]\n"
              << "\n"
              << "Attach to a running process for profiling.\n"
              << "\n"
              << "Options:\n"
              << "  -p <pid>             Process ID to attach to (required)\n"
              << "  -o, --output PATH    Output path for profiling results\n"
              << "  -F, --format FORMAT[,FORMAT,...]\n"
              << "                       Output format(s): perfetto, rocpd\n"
              << "  -h, --help           Show this help message\n"
              << "\n"
              << "Environment variables:\n"
              << "  ROCPROFSYS_OUTPUT_PATH       Output directory for profiling data\n"
              << "  ROCPROFSYS_TRACE             Enable perfetto trace output\n"
              << "  ROCPROFSYS_USE_ROCPD         Enable rocpd database output\n"
              << "  ROCPROF_ATTACH_TOOL_LIBRARY  Path to the tool library\n"
              << "\n"
              << "Once attached, press ENTER to detach from the process.\n";
}

void
setup_session_env()
{
    const auto* output_use_current_time_env_name =
        rocprofsys::env_vars::OUTPUT_USE_CURRENT_TIME;
    const auto* reattach_add_session_id_env_name =
        rocprofsys::env_vars::REATTACH_ADD_SESSION_ID;

    // enable the use of the current time for the output path
    setenv(output_use_current_time_env_name, "true", 1);

    // enable the re-attach to add a session ID to the output path
    setenv(reattach_add_session_id_env_name, "true", 1);
}

/**
 * Picks the tool library path, as seen by process @p pid: the user's
 * ROCPROF_ATTACH_TOOL_LIBRARY, else this installation's library if the target can see it,
 * else the library shipped next to the target's own librocprofiler-register (e.g. inside
 * a container with its own ROCm installation).
 */
std::optional<std::string>
resolve_tool_library(pid_t pid)
{
    namespace path = rocprofsys::common::path;

    if(const auto* user_library = std::getenv("ROCPROF_ATTACH_TOOL_LIBRARY"))
    {
        if(!path::is_missing_in_target(pid, user_library))
        {
            return std::string{ user_library };
        }
        LOG_ERROR("Tool library '{}' from ROCPROF_ATTACH_TOOL_LIBRARY does not exist in "
                  "the mount namespace of process {}.",
                  user_library, pid);
        return std::nullopt;
    }

    const auto own_library = path::get_internal_libpath("librocprof-sys-dl.so");
    if(!path::is_missing_in_target(pid, own_library))
    {
        return own_library;
    }

    if(auto target_library = path::find_library_in_loaded_dir(
           pid, "librocprof-sys-dl.so", "librocprofiler-register.so"))
    {
        LOG_INFO("Found tool library in the ROCm installation of process {}", pid);
        return target_library;
    }

    LOG_ERROR(
        "Tool library '{}' does not exist in the mount namespace of process {}, and "
        "no librocprof-sys-dl.so was found in its ROCm installation. Set "
        "ROCPROF_ATTACH_TOOL_LIBRARY to the path as seen by the target.",
        own_library, pid);
    return std::nullopt;
}

void
setup_output_env(const std::string& output_path)
{
    const auto* existing_output_path = getenv(rocprofsys::env_vars::OUTPUT_PATH);
    if(output_path.empty() && existing_output_path != nullptr)
    {
        LOG_INFO("Output path: {}", existing_output_path);
        return;
    }

    const auto* const pwd = getenv("PWD");
    const auto        output =
        output_path.empty() ? fmt::format("{}/rocprof-sys-output", pwd) : output_path;

    setenv(rocprofsys::env_vars::OUTPUT_PATH, output.c_str(), 1);
    LOG_INFO("Output path: {}", output);
}

void
setup_output_format_env(const std::vector<std::string>& formats)
{
    if(formats.empty())
    {
        return;
    }

    auto const has_format = [&formats](const std::string& fmt) {
        return std::ranges::find(formats, fmt) != formats.end();
    };

    // setenv("ROCPROFSYS_PROFILE", "false", 1);

    if(has_format("perfetto") || has_format("rocpd"))
    {
        setenv(rocprofsys::env_vars::TRACE, has_format("perfetto") ? "true" : "false", 1);
        setenv(rocprofsys::env_vars::USE_ROCPD, has_format("rocpd") ? "true" : "false",
               1);
    }

    LOG_INFO("Output format: {}", fmt::join(formats, " "));
}

bool
is_option(const char* arg, const char* short_opt, const char* long_opt)
{
    return std::strcmp(arg, short_opt) == 0 || std::strcmp(arg, long_opt) == 0;
}

const char*
consume_arg(int& i, int argc, char* argv[], const char* opt_name)
{
    if(i + 1 >= argc)
    {
        LOG_ERROR("{} requires an argument.", opt_name);
        std::exit(EXIT_FAILURE);
    }
    return argv[++i];
}

void
parse_pid(attach_options& opts, const char* arg)
{
    try
    {
        opts.pid = std::stoi(arg);
        if(opts.pid <= 0)
        {
            LOG_ERROR("PID must be a positive integer.");
            std::exit(EXIT_FAILURE);
        }
    } catch(const std::exception&)
    {
        LOG_ERROR("Invalid PID '{}'.", arg);
        std::exit(EXIT_FAILURE);
    }
}

void
parse_formats(attach_options& opts, const char* arg)
{
    std::string       token;
    std::stringstream ss(arg);
    while(std::getline(ss, token, ','))
    {
        if(token == "perfetto" || token == "rocpd")
        {
            opts.profile_format.push_back(token);
        }
        else
        {
            LOG_ERROR("Invalid format '{}'. Valid options: perfetto, rocpd", token);
            std::exit(EXIT_FAILURE);
        }
    }
}

attach_options
parse_args(int argc, char* argv[])
{
    attach_options opts;

    for(int i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];

        if(is_option(arg, "-h", "--help"))
        {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }

        if(is_option(arg, "-p", "--pid"))
        {
            parse_pid(opts, consume_arg(i, argc, argv, "-p"));
            continue;
        }

        if(is_option(arg, "-o", "--output"))
        {
            opts.output_path = consume_arg(i, argc, argv, "-o/--output");
            continue;
        }

        if(is_option(arg, "-F", "--format"))
        {
            parse_formats(opts, consume_arg(i, argc, argv, "-F/--format"));
            continue;
        }

        LOG_ERROR("Unknown option '{}'.", arg);
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    return opts;
}

void
print_banner()
{
    std::cout << R"(
  ____   ___   ____ __  __   ______   ______ _____ _____ __  __ ____       _  _____ _____  _    ____ _   _
 |  _ \ / _ \ / ___|  \/  | / ___\ \ / / ___|_   _| ____|  \/  / ___|     / \|_   _|_   _|/ \  / ___| | | |
 | |_) | | | | |   | |\/| | \___ \\ V /\___ \ | | |  _| | |\/| \___ \    / _ \ | |   | | / _ \| |   | |_| |
 |  _ <| |_| | |___| |  | |  ___) || |  ___) || | | |___| |  | |___) |  / ___ \| |   | |/ ___ \ |___|  _  |
 |_| \_\\___/ \____|_|  |_| |____/ |_| |____/ |_| |_____|_|  |_|____/  /_/   \_\_|   |_/_/   \_\____|_| |_|

)" << "\n";
}

}  // namespace

int
main(int argc, char* argv[])
{
    print_banner();
    if(argc < 2)
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    auto const opts = parse_args(argc, argv);

    if(opts.pid < 0)
    {
        LOG_ERROR("-p <pid> is required.");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    setup_session_env();
    setup_output_env(opts.output_path);
    setup_output_format_env(opts.profile_format);

    const auto pid = opts.pid;

    const auto tool_library = resolve_tool_library(pid);
    if(!tool_library)
    {
        return EXIT_FAILURE;
    }
    LOG_INFO("Using tool library: {}", *tool_library);
    setenv("ROCPROF_ATTACH_TOOL_LIBRARY", tool_library->c_str(), 1);
    setenv("ROCP_TOOL_LIBRARIES", tool_library->c_str(), 1);

    LOG_INFO("Trying to attach to process {}", pid);

    auto result = rocattach_attach(pid);
    if(result != ROCATTACH_STATUS_SUCCESS)
    {
        LOG_ERROR("Failed to attach to process {}", pid);
        return EXIT_FAILURE;
    }

    LOG_INFO("Attached to process {}. Press ENTER to detach.", pid);
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    result = rocattach_detach(pid);
    if(result != ROCATTACH_STATUS_SUCCESS)
    {
        LOG_ERROR("Failed to detach from process {}", pid);
        return EXIT_FAILURE;
    }

    LOG_INFO("Detached from process {}", pid);

    return EXIT_SUCCESS;
}
