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

/**
 * @file late_start.cpp
 * @brief Late-start functionality for rocprofiler-sdk
 *
 * Enables rocprofiler-sdk to receive API tables that were registered with
 * rocprofiler-register before the SDK was loaded. This allows applications to
 * dynamically enable profiling after GPU runtimes have already initialized.
 *
 * Instead of directly accessing runtime symbols (which bypasses the architecture),
 * this implementation properly integrates with rocprofiler-register by calling
 * rocprofiler_register_invoke_all_registrations() to trigger re-propagation of
 * all registered API tables through the normal rocprofiler_set_api_table() flow.
 */

#include "lib/rocprofiler-sdk/late_start.hpp"
#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <dlfcn.h>

namespace rocprofiler
{
namespace late_start
{
namespace
{
// Define minimal types for rocprofiler-register API to avoid build dependency
// These match the definitions in rocprofiler-register.h
struct rocprofiler_register_registration_info_t
{
    size_t      size;
    const char* common_name;
    uint32_t    lib_version;
    uint64_t    api_table_length;
};

using rocprofiler_register_registration_info_cb_t =
    int (*)(rocprofiler_register_registration_info_t*, void*);

using rocprofiler_register_iterate_registration_info_fn_t =
    int (*)(rocprofiler_register_registration_info_cb_t, void*);

using rocprofiler_register_invoke_all_fn_t = int (*)();

// Callback to count number of registered API tables
int
count_registration_callback(rocprofiler_register_registration_info_t* info, void* data)
{
    auto* count = static_cast<size_t*>(data);
    (*count)++;
    ROCP_TRACE << "Found registration: " << (info->common_name ? info->common_name : "<null>")
               << " (version: " << info->lib_version
               << ", api_table_length: " << info->api_table_length << ")";
    return 0;  // Continue iterating
}
}  // namespace

rocprofiler_status_t
invoke_register_propagation()
{
    ROCP_INFO << "Invoking rocprofiler-register to re-propagate all registered API tables";

    // Step 1: Get the rocprofiler_register_iterate_registration_info function
    // Use dlsym(nullptr, ...) since rocprofiler-register should already be loaded
    auto* iterate_info_fn = reinterpret_cast<rocprofiler_register_iterate_registration_info_fn_t>(
        dlsym(nullptr, "rocprofiler_register_iterate_registration_info"));

    if(!iterate_info_fn)
    {
        ROCP_WARNING << "rocprofiler-register library is not loaded. "
                     << "This is expected if no runtimes have initialized yet, or if "
                     << "rocprofiler-register is not being used. Late-start profiling "
                     << "requires rocprofiler-register to store runtime API tables.";
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }

    // Step 2: Check if any dispatch tables have been registered
    // Count the number of registrations using iterate_registration_info
    size_t registration_count = 0;
    iterate_info_fn(count_registration_callback, &registration_count);

    if(registration_count == 0)
    {
        ROCP_INFO << "No runtime API tables have been registered with rocprofiler-register yet. "
                  << "This is normal if runtimes initialize after rocprofiler_force_configure(). "
                  << "Runtimes that initialize later will be automatically profiled.";
        return ROCPROFILER_STATUS_SUCCESS;
    }

    ROCP_TRACE << "Found " << registration_count << " registered API tables";

    // Step 3: Get the rocprofiler_register_invoke_all_registrations function
    // This function re-propagates all stored API table registrations to rocprofiler-sdk
    auto* invoke_all_fn = reinterpret_cast<rocprofiler_register_invoke_all_fn_t>(
        dlsym(nullptr, "rocprofiler_register_invoke_all_registrations"));

    if(!invoke_all_fn)
    {
        ROCP_ERROR << "Failed to find rocprofiler_register_invoke_all_registrations symbol. "
                   << "The loaded rocprofiler-register version may be too old or incompatible. "
                   << "Please update to a newer version of rocprofiler-register.";
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }

    // Step 4: Invoke the function to re-propagate all registrations
    // This will call rocprofiler_set_api_table() for each registered runtime
    ROCP_TRACE << "Calling rocprofiler_register_invoke_all_registrations()";
    int ret = invoke_all_fn();

    // Step 5: Check result and interpret return codes
    // Return codes from rocprofiler-register (see rocprofiler-register.h):
    //   0 (ROCP_REG_SUCCESS)  = Successfully propagated registrations
    //   1 (ROCP_REG_NO_TOOLS) = No tools found (rocprofiler-sdk symbols not visible yet)
    //                           This can happen if called very early, but is not an error
    //                           since the SDK is now loaded and will receive future registrations
    //   Other values          = Actual errors (deadlock, invalid arguments, etc.)

    if(ret == 0)
    {
        ROCP_INFO << "Successfully re-propagated all registered API tables. "
                  << "Late-start profiling is now active for all registered runtimes.";
        return ROCPROFILER_STATUS_SUCCESS;
    }
    else if(ret == 1)  // ROCP_REG_NO_TOOLS
    {
        ROCP_INFO << "No runtime API tables have been registered with rocprofiler-register yet. "
                  << "This is normal if runtimes initialize after rocprofiler_force_configure(). "
                  << "Runtimes that initialize later will be automatically profiled.";
        return ROCPROFILER_STATUS_SUCCESS;  // Not an error - normal startup order
    }
    else
    {
        ROCP_ERROR << "rocprofiler_register_invoke_all_registrations() returned error code: " << ret
                   << ". Late-start profiling may not be functional.";
        return ROCPROFILER_STATUS_ERROR;
    }
}

}  // namespace late_start
}  // namespace rocprofiler
