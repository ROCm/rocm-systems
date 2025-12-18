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
 * @file late.cpp
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

#include "lib/rocprofiler-sdk/registration/late.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/registration/iterate.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <dlfcn.h>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace registration
{
namespace late
{
namespace
{
using rocprofiler_register_invoke_all_fn_t = int (*)();
}  // namespace

rocprofiler_status_t
invoke_register_propagation()
{
    ROCP_INFO << "Invoking rocprofiler-register to re-propagate all registered API tables";

    auto registered_tables = registration::iterate::get_runtime_registrations();

    if(!registered_tables.has_value())
    {
        return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_REGISTER_VERSION;
    }
    else if(registered_tables.has_value() && registered_tables->empty())
    {
        ROCP_INFO << "No runtime API tables have been registered with rocprofiler-register yet. "
                  << "This is normal if runtimes initialize after rocprofiler_force_configure(). "
                  << "Runtimes that initialize later will be automatically profiled.";
        return ROCPROFILER_STATUS_SUCCESS;
    }

    ROCP_TRACE << "Found " << registered_tables->size() << " registered API tables";

    // Step 3: Get the rocprofiler_register_invoke_all_registrations function
    // This function re-propagates all stored API table registrations to rocprofiler-sdk
    auto* invoke_all_fn = reinterpret_cast<rocprofiler_register_invoke_all_fn_t>(
        dlsym(nullptr, "rocprofiler_register_invoke_all_registrations"));

    if(!invoke_all_fn)
    {
        ROCP_ERROR << fmt::format(
            "Found {} registered API tables ({}) but rocprofiler_register_invoke_all_registrations "
            "is not available. The loaded rocprofiler-register version (pre-v0.5.0) does not "
            "support late initialization of profiling. Please update to rocprofiler-register "
            "v0.5.0 or newer (ROCm 7.0 or newer).",
            registered_tables->size(),
            fmt::join(*registered_tables, ", "));
        return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_REGISTER_VERSION;
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

}  // namespace late
}  // namespace registration
}  // namespace rocprofiler
