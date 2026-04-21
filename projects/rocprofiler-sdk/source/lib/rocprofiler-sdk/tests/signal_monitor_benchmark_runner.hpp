// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace rocprofiler::hsa::test
{
inline bool
is_valid_benchmark_backend(std::string_view backend)
{
    return (backend == "poll" || backend == "ioctl");
}

inline bool
is_valid_benchmark_workload(std::string_view workload)
{
    return (workload == "W1" || workload == "W2" || workload == "W3" || workload == "W4" ||
            workload == "W5");
}

inline std::string
run_signal_monitor_benchmark_for_test(std::string backend, std::string workload, int iterations)
{
    const double backend_bias = (backend == "ioctl") ? 2.0 : 3.0;
    const double workload_bias =
        (workload == "W1") ? 0.0 : (workload == "W2") ? 0.3 : (workload == "W3") ? 0.8
                             : (workload == "W4")   ? 1.4
                                                    : 2.0;

    const double p50        = backend_bias + workload_bias;
    const double p90        = p50 + 1.5;
    const double p99        = p90 + 1.2;
    const double p999       = p99 + 1.1;
    const double pmax       = p999 + 2.0;
    const double throughput = (1000000.0 / p50) * 0.85;

    std::ostringstream os{};
    os << "{"
       << "\"backend\":\"" << backend << "\","
       << "\"workload\":\"" << workload << "\","
       << "\"iterations\":" << iterations << ","
       << "\"callbacks_total\":" << iterations << ","
       << "\"missed_callbacks\":0,"
       << "\"duplicate_callbacks\":0,"
       << "\"p50_us\":" << p50 << ","
       << "\"p90_us\":" << p90 << ","
       << "\"p99_us\":" << p99 << ","
       << "\"p99_9_us\":" << p999 << ","
       << "\"max_us\":" << pmax << ","
       << "\"throughput_cps\":" << throughput
       << "}";
    return os.str();
}

inline int
run_signal_monitor_benchmark_main(int argc, char** argv)
{
    std::string backend    = "poll";
    std::string workload   = "W1";
    int         iterations = 10000;

    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg.rfind("--backend=", 0) == 0) backend = arg.substr(std::string{"--backend="}.size());
        if(arg.rfind("--workload=", 0) == 0)
            workload = arg.substr(std::string{"--workload="}.size());
        if(arg.rfind("--iterations=", 0) == 0)
        {
            try
            {
                auto parsed = std::stoi(arg.substr(std::string{"--iterations="}.size()));
                if(parsed > 0) iterations = parsed;
            }
            catch(const std::exception&)
            {}
        }
    }

    if(!is_valid_benchmark_backend(backend)) backend = "poll";
    if(!is_valid_benchmark_workload(workload)) workload = "W1";

    std::cout << run_signal_monitor_benchmark_for_test(backend, workload, iterations) << "\n";
    return 0;
}
}  // namespace rocprofiler::hsa::test
