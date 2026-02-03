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

/**
 * @file librocprofiler_python_preload.cpp
 * @brief LD_PRELOAD shim for early rocprofiler registration
 *
 * This library provides a constructor that runs before the main program
 * (and before HSA/HIP runtime initialization) to register the rocprofiler
 * tool callbacks. This enables dispatch interception for Python profiling.
 *
 * Usage:
 *   LD_PRELOAD=/path/to/librocprofiler-python-preload.so python3 script.py
 */

#include <rocprofiler-sdk/registration.h>

#include <iostream>

// Forward declaration of rocprofiler_configure from profiler_session.cpp
// This function is defined with extern "C" linkage in profiler_session.cpp
extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id);

namespace
{
/**
 * @brief Constructor function that runs at library load time
 *
 * This constructor uses priority 101 (default is 65535, lower runs earlier)
 * to ensure it runs before most other constructors.
 */
__attribute__((constructor(101))) void
rocprofiler_python_early_init()
{
    auto status = rocprofiler_force_configure(&rocprofiler_configure);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "[rocprofiler-python-preload] Warning: force_configure returned "
                  << rocprofiler_get_status_string(status) << std::endl;
    }
}
}  // namespace
