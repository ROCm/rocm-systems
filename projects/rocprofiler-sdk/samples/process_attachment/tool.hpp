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
 * @file samples/process_attachment/tool.hpp
 *
 * @brief Header for the process attachment tool library
 */

#pragma once 

#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/experimental/registration.h>
#include <cstdint>

namespace attachment_tool
{
/**
 * @brief Initialize the attachment tool
 * This function is called when the tool is loaded via process attachment
 */
int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data);

/**
 * @brief Finalize the attachment tool
 * This function is called when the tool is unloaded
 */
void
tool_fini(void* tool_data);

/**
 * @brief Get statistics about the profiling session
 */
void
get_statistics();

/**
 * @brief Enable/disable kernel tracing
 */
void
set_kernel_tracing(bool enable);

/**
 * @brief Enable/disable HIP API tracing
 */
void
set_hip_api_tracing(bool enable);

}  // namespace attachment_tool