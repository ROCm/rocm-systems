// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <rocprofiler-sdk-rocattach/rocattach.h>

namespace
{
void
print_usage(const char* prog_name)
{
    std::cout << "Usage: " << prog_name << " -p <pid>\n"
              << "\n"
              << "Attach to a running process for profiling.\n"
              << "\n"
              << "Options:\n"
              << "  -p <pid>      Process ID to attach to\n"
              << "  -h, --help    Show this help message\n"
              << "\n"
              << "Once attached, press ENTER to detach from the process.\n";
}
}  // namespace

int
main(int argc, char* argv[])
{
    // Check for help flag
    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    // Validate arguments: require "-p <pid>"
    if(argc < 3 || std::strcmp(argv[1], "-p") != 0)
    {
        std::cerr << "Error: Missing or invalid arguments.\n\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const auto* _attach_pid = argv[2];

    std::cout << "[rocprof-sys-attach] Trying to attach to process " << _attach_pid
              << std::endl;

    int pid = 0;
    try
    {
        pid = std::stoi(_attach_pid);
        if(pid <= 0)
        {
            std::cerr << "Error: Invalid PID '" << _attach_pid
                      << "'. PID must be a positive integer.\n";
            return EXIT_FAILURE;
        }
    } catch(const std::invalid_argument&)
    {
        std::cerr << "Error: Invalid PID '" << _attach_pid << "'. Not a valid number.\n";
        return EXIT_FAILURE;
    } catch(const std::out_of_range&)
    {
        std::cerr << "Error: PID '" << _attach_pid << "' is out of range.\n";
        return EXIT_FAILURE;
    }

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

    return EXIT_SUCCESS;
}
