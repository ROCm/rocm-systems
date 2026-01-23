// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

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

    auto pid    = atoi(_attach_pid);
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
