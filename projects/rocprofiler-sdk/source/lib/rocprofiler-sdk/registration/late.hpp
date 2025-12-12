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
namespace registration
{
namespace late
{
/**
 * @brief Invoke rocprofiler-register to re-propagate all registered API tables
 *
 * This function calls rocprofiler_register_invoke_all_registrations() which triggers
 * rocprofiler-register to re-send all previously registered runtime API tables to
 * rocprofiler-sdk via the rocprofiler_set_api_table() entry point.
 *
 * This enables "late-start" profiling where rocprofiler-sdk is loaded after runtimes
 * (HSA, HIP, ROCTX, RCCL, etc.) have already initialized and registered their API
 * tables with rocprofiler-register.
 *
 * The function works by:
 * 1. Loading rocprofiler-register library (if not already loaded)
 * 2. Finding rocprofiler_register_invoke_all_registrations() symbol
 * 3. Calling it to trigger re-propagation of stored registrations
 * 4. Each registered runtime's API tables are sent via rocprofiler_set_api_table()
 *
 * @return ROCPROFILER_STATUS_SUCCESS if successful or no runtimes registered yet
 * @return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE if rocprofiler-register not loaded
 * @return ROCPROFILER_STATUS_ERROR on other failures
 */
rocprofiler_status_t
invoke_register_propagation();

}  // namespace late
}  // namespace registration
}  // namespace rocprofiler
