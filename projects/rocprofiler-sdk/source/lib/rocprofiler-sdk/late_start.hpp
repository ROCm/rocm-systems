// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace late_start
{

/**
 * @brief Information about a detected runtime
 */
struct runtime_info
{
    void*       handle      = nullptr;  ///< dlopen handle to the runtime library
    const char* name        = nullptr;  ///< Name of the runtime library
    bool        initialized = false;    ///< Whether runtime is initialized
};

/**
 * @brief Detect if HSA runtime is loaded and initialized
 */
runtime_info
detect_hsa_runtime();

/**
 * @brief Detect if HIP runtime is loaded and initialized
 */
runtime_info
detect_hip_runtime();

/**
 * @brief Wrap HSA API tables for late-start profiling
 */
rocprofiler_status_t
wrap_hsa_tables(runtime_info& info);

/**
 * @brief Wrap HIP API tables for late-start profiling
 */
rocprofiler_status_t
wrap_hip_tables(runtime_info& info);

/**
 * @brief Get current late-start state
 * @return 0=not started, 1=starting, 2=started
 */
int
get_state();

/**
 * @brief Check if late-start is active
 * @return true if state == 2
 */
bool
is_active();

/**
 * @brief Restore original runtime tables
 */
rocprofiler_status_t
restore_tables();

}  // namespace late_start
}  // namespace rocprofiler
