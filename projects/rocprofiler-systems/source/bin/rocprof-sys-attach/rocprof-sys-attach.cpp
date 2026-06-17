// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "common/path.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/ranges.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <rocprofiler-sdk-rocattach/rocattach.h>

namespace
{
struct attach_options
{
    int                      pid            = -1;
    std::string              output_path    = {};
    std::vector<std::string> profile_format = {};
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
setup_tool_library_env()
{
    const auto* attach_tool_library_env_name = "ROCPROF_ATTACH_TOOL_LIBRARY";
    const auto* rocp_tool_libraries_env_name = "ROCP_TOOL_LIBRARIES";
    const auto* output_use_current_time_env_name =
        rocprofsys::env_vars::OUTPUT_USE_CURRENT_TIME;
    const auto* reattach_add_session_id_env_name =
        rocprofsys::env_vars::REATTACH_ADD_SESSION_ID;

    // enable the use of the current time for the output path
    setenv(output_use_current_time_env_name, "true", 1);

    // enable the re-attach to add a session ID to the output path
    setenv(reattach_add_session_id_env_name, "true", 1);

    const auto* existing = std::getenv(attach_tool_library_env_name);
    if(existing != nullptr)
    {
        setenv(rocp_tool_libraries_env_name, existing, 0);
        LOG_INFO("Using tool library: {}", existing);
        return;
    }

    const auto path =
        rocprofsys::common::path::get_internal_libpath("librocprof-sys-dl.so");
    if(!path.empty())
    {
        setenv(attach_tool_library_env_name, path.c_str(), 0);
        setenv(rocp_tool_libraries_env_name, path.c_str(), 0);
        LOG_INFO("Using tool library: {}", path);
    }
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
    if(formats.empty()) return;

    auto has_format = [&formats](const std::string& fmt) {
        return std::find(formats.begin(), formats.end(), fmt) != formats.end();
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

    auto opts = parse_args(argc, argv);

    if(opts.pid < 0)
    {
        LOG_ERROR("-p <pid> is required.");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    setup_tool_library_env();
    setup_output_env(opts.output_path);
    setup_output_format_env(opts.profile_format);

    const auto pid = opts.pid;

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
