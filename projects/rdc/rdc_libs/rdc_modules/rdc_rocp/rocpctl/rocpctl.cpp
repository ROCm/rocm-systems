// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>

#include <unistd.h>
#include <chrono>
#include <thread>
#include <cstdint>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <cstdlib>

#include "RocpCounterSampler.h"

#define HIP_CALL(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr, "%s\n", hipGetErrorString(err));                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)


// specify list of metrics in arguments to collect
int
main(int argc, char** argv)
{
    int ntotdevice = 0;
    HIP_CALL(hipGetDeviceCount(&ntotdevice));

    // 
    // Check if any GPU devices are available
    if(ntotdevice == 0) {
        std::cerr << "No GPU devices found. Exiting." << std::endl;
        return -1;
    }

    std::vector<std::string> metric_fields;
    uint64_t duration = 1000;    // Default sampling duration (in microseconds)
    uint32_t ptl_delay = 10;     // Default PTL delay (in milliseconds)
    long ndevice = ntotdevice;   // Use actual device count

    if(ndevice > ntotdevice) ndevice = ntotdevice;
    if(ndevice < 1) ndevice = ntotdevice;

    // Parse arguments first to get ptl_delay before creating guard
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == nullptr) continue;
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS] [METRICS...]\n\n"
                      << "Options:\n"
                      << "  -h, --help          Show this help message and exit\n"
                      << "  -d <duration>       Sampling duration in microseconds (default: 1000)\n"
                      << "  -p <delay>          PTL delay in milliseconds (default: 10)\n\n"
                      << "Arguments:\n"
                      << "  METRICS             List of metric fields to collect\n\n"
                      << "Example:\n"
                      << "  " << argv[0] << " -d 2000 -p 20 metric1 metric2\n";
            return 0;
        } else if (arg == "-d") {
            if (i + 1 >= argc || argv[i + 1] == nullptr) {
                std::cerr << "Option -d requires a numeric argument" << std::endl;
                return -1;
            }
            try {
                duration = std::stoull(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid value for -d: " << argv[i] << std::endl;
                return -1;
            }
        } else if (arg == "-p") {
            if (i + 1 >= argc || argv[i + 1] == nullptr) {
                std::cerr << "Option -p requires a numeric argument" << std::endl;
                return -1;
            }
            try {
                ptl_delay = std::stoul(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid value for -p: " << argv[i] << std::endl;
                return -1;
            }
        } else {                
            metric_fields.push_back(arg);            
        }
    }

    try {        
        int rc = amd::rocp::CounterSampler::runSample(metric_fields, duration, ptl_delay);
        if (rc != 0) {
            std::cerr << "run sample err: " << rc << "\n"; 
            return -1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return -1;
    }
    return 0;
}
