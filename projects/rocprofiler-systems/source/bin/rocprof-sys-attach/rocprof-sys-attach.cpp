// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/path.hpp"

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
    const auto* env_name = "ROCPROF_ATTACH_TOOL_LIBRARY";

    const auto* existing = std::getenv(env_name);
    if(existing != nullptr)
    {
        std::cout << "[rocprof-sys-attach] Using tool library: " << existing << std::endl;
        return;
    }

    const auto path =
        rocprofsys::common::path::get_internal_libpath("librocprof-sys-dl.so");
    if(!path.empty())
    {
        setenv(env_name, path.c_str(), 0);
        std::cout << "[rocprof-sys-attach] Using tool library: " << path << std::endl;
    }
}

void
setup_output_env(const std::string& output_path)
{
    if(output_path.empty()) return;

    setenv("ROCPROFSYS_OUTPUT_PATH", output_path.c_str(), 1);
    std::cout << "[rocprof-sys-attach] Output path: " << output_path << std::endl;
}

void
setup_output_format_env(const std::vector<std::string>& formats)
{
    if(formats.empty()) return;

    auto has_format = [&formats](const std::string& fmt) {
        return std::find(formats.begin(), formats.end(), fmt) != formats.end();
    };

    if(has_format("perfetto"))
    {
        setenv("ROCPROFSYS_TRACE", "true", 1);
    }

    if(has_format("rocpd"))
    {
        setenv("ROCPROFSYS_USE_ROCPD", "true", 1);
    }

    std::cout << "[rocprof-sys-attach] Output format:";
    for(const auto& fmt : formats)
        std::cout << " " << fmt;
    std::cout << std::endl;
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
        std::cerr << "Error: " << opt_name << " requires an argument.\n";
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
            std::cerr << "Error: PID must be a positive integer.\n";
            std::exit(EXIT_FAILURE);
        }
    } catch(const std::exception&)
    {
        std::cerr << "Error: Invalid PID '" << arg << "'.\n";
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
            std::cerr << "Error: Invalid format '" << token
                      << "'. Valid options: perfetto, rocpd\n";
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

        if(is_option(arg, "-p", "-p"))
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

        std::cerr << "Error: Unknown option '" << arg << "'.\n";
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    return opts;
}
}  // namespace

int
main(int argc, char* argv[])
{
    if(argc < 2)
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    auto opts = parse_args(argc, argv);

    if(opts.pid < 0)
    {
        std::cerr << "Error: -p <pid> is required.\n\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    setup_tool_library_env();
    setup_output_env(opts.output_path);
    setup_output_format_env(opts.profile_format);

    const auto pid = opts.pid;

    std::cout << "[rocprof-sys-attach] Trying to attach to process " << pid << std::endl;

    auto result = rocattach_attach(pid);
    if(result != ROCATTACH_STATUS_SUCCESS)
    {
        std::cerr << "[rocprof-sys-attach] Failed to attach to process " << pid
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[rocprof-sys-attach] Attached to process " << pid
              << ". Press ENTER to detach." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    result = rocattach_detach(pid);
    if(result != ROCATTACH_STATUS_SUCCESS)
    {
        std::cerr << "[rocprof-sys-attach] Failed to detach from process " << pid
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[rocprof-sys-attach] Detached from process " << pid << std::endl;

    // Print output location info
    if(!opts.profile_format.empty())
    {
        std::string output_dir =
            opts.output_path.empty() ? "rocprof-sys-output/" : opts.output_path;
        std::cout << "[rocprof-sys-attach] Output written to: " << output_dir
                  << std::endl;

        for(const auto& fmt : opts.profile_format)
        {
            if(fmt == "perfetto")
            {
                std::cout << "[rocprof-sys-attach]   - Perfetto trace: perfetto-trace-"
                          << pid << ".proto" << std::endl;
            }
            else if(fmt == "rocpd")
            {
                std::cout << "[rocprof-sys-attach]   - RocPD database: rocpd-" << pid
                          << ".db" << std::endl;
            }
        }
    }

    return EXIT_SUCCESS;
}
