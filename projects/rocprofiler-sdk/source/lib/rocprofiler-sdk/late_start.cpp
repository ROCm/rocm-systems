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
 * Enables rocprofiler-sdk to be loaded and wrap HSA/HIP APIs after the runtimes
 * have already been initialized. This allows applications to dynamically enable
 * profiling after GPU initialization has occurred.
 */

#include "lib/rocprofiler-sdk/late_start.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hip/hip.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/intercept_table.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <dlfcn.h>
#include <atomic>
#include <mutex>

namespace rocprofiler
{
namespace late_start
{
//------------------------------------------------------------------------------
// State Management
//------------------------------------------------------------------------------

/// Late-start state: 0=not started, 1=starting, 2=started
std::atomic<int> late_start_state{0};

/// Mutex for thread-safe late-start operations
std::mutex late_start_mutex;

/// Saved HSA API table pointer for restoration
HsaApiTable* saved_hsa_table = nullptr;

/// Saved HIP runtime dispatch table pointer for restoration
HipDispatchTable* saved_hip_table = nullptr;

/// Saved HIP compiler dispatch table pointer for restoration
HipCompilerDispatchTable* saved_hip_compiler_table = nullptr;

//------------------------------------------------------------------------------
// Runtime Detection Functions
//------------------------------------------------------------------------------

runtime_info
detect_hsa_runtime()
{
    runtime_info info = {nullptr, "libhsa-runtime64.so.1", false};

    ROCP_TRACE << "Detecting HSA runtime...";

    // Check if HSA is loaded (RTLD_NOLOAD doesn't load, just checks)
    info.handle = dlopen(info.name, RTLD_NOLOAD | RTLD_LAZY);
    if(!info.handle)
    {
        // Try alternate name without version suffix
        info.name   = "libhsa-runtime64.so";
        info.handle = dlopen(info.name, RTLD_NOLOAD | RTLD_LAZY);
    }

    if(info.handle)
    {
        ROCP_TRACE << "HSA runtime library " << info.name << " is loaded";

        // Check if initialized by looking for the table accessor
        using get_table_fn = const HsaApiTable* (*) ();
        auto get_table =
            reinterpret_cast<get_table_fn>(dlsym(info.handle, "hsa_table_interface_get_table"));

        if(get_table)
        {
            ROCP_TRACE << "Found hsa_table_interface_get_table symbol";

            // Call the accessor to get the table
            const HsaApiTable* table = get_table();

            // HSA is initialized if table is non-null and has a core table
            info.initialized = (table != nullptr && table->core_ != nullptr);

            if(info.initialized)
            {
                ROCP_INFO << "HSA runtime is loaded and initialized";
            }
            else
            {
                ROCP_TRACE << "HSA runtime is loaded but not initialized";
            }
        }
        else
        {
            ROCP_TRACE << "hsa_table_interface_get_table symbol not found";
        }
    }
    else
    {
        ROCP_TRACE << "HSA runtime library not loaded";
    }

    return info;
}

runtime_info
detect_hip_runtime()
{
    runtime_info info = {nullptr, "libamdhip64.so", false};

    ROCP_TRACE << "Detecting HIP runtime...";

    // Check if HIP is loaded (RTLD_NOLOAD doesn't load, just checks)
    info.handle = dlopen(info.name, RTLD_NOLOAD | RTLD_LAZY);

    if(info.handle)
    {
        ROCP_TRACE << "HIP runtime library " << info.name << " is loaded";

        // Check if initialized by looking for the dispatch table accessor
        using get_table_fn = HipDispatchTable* (*) ();
        auto get_table = reinterpret_cast<get_table_fn>(dlsym(info.handle, "GetHipDispatchTable"));

        if(get_table)
        {
            ROCP_TRACE << "Found GetHipDispatchTable symbol";

            // Call the accessor to get the dispatch table
            HipDispatchTable* table = get_table();

            // HIP is initialized if table is non-null
            info.initialized = (table != nullptr);

            if(info.initialized)
            {
                ROCP_INFO << "HIP runtime is loaded and initialized";
            }
            else
            {
                ROCP_TRACE << "HIP runtime is loaded but not initialized";
            }
        }
        else
        {
            ROCP_TRACE << "GetHipDispatchTable symbol not found";
        }
    }
    else
    {
        ROCP_TRACE << "HIP runtime library not loaded";
    }

    return info;
}

//------------------------------------------------------------------------------
// Table Wrapping Functions
//------------------------------------------------------------------------------

rocprofiler_status_t
wrap_hsa_tables(runtime_info& hsa_info)
{
    ROCP_INFO << "Wrapping HSA API tables for late-start profiling";

    // Step 1: Get hsa_table_interface_get_table function from runtime
    using get_table_fn = HsaApiTable* (*) ();
    auto get_table =
        reinterpret_cast<get_table_fn>(dlsym(hsa_info.handle, "hsa_table_interface_get_table"));

    if(!get_table)
    {
        ROCP_ERROR << "Failed to find hsa_table_interface_get_table symbol";
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_AVAILABLE;
    }

    // Step 2: Get the HSA API table from runtime
    // Note: The table is returned as const, but we need to modify it to install wrappers.
    // This is safe because the table IS mutable in practice (the runtime modifies it).
    HsaApiTable* table = const_cast<HsaApiTable*>(get_table());

    if(!table || !table->core_)
    {
        ROCP_ERROR << "HSA API table or core table is null";
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_AVAILABLE;
    }

    ROCP_TRACE << "Retrieved HSA API table from runtime";

    // Step 3: Save the table pointer for potential restoration
    saved_hsa_table = table;

    // Step 4: Get library version from runtime
    uint64_t lib_version = 0;
    if(table->core_->hsa_system_get_info_fn)
    {
        uint16_t major = 0, minor = 0;
        table->core_->hsa_system_get_info_fn(HSA_SYSTEM_INFO_VERSION_MAJOR, &major);
        table->core_->hsa_system_get_info_fn(HSA_SYSTEM_INFO_VERSION_MINOR, &minor);
        lib_version = (static_cast<uint64_t>(major) << 32) | minor;
        ROCP_TRACE << "HSA runtime version: " << major << "." << minor;
    }

    // Use instance 0 for late-start (first/only instance)
    constexpr uint64_t lib_instance = 0;

    // Step 5: Copy original function pointers to internal storage
    ROCP_TRACE << "Copying original HSA function pointers";
    hsa::copy_table(table->core_, lib_instance);
    hsa::copy_table(table->amd_ext_, lib_instance);
    hsa::copy_table(table->image_ext_, lib_instance);
    hsa::copy_table(table->finalizer_ext_, lib_instance);
    if(table->tools_)
    {
        hsa::copy_table(table->tools_, lib_instance);
    }

    // Step 6: Initialize supporting subsystems
    ROCP_TRACE << "Initializing HSA supporting subsystems";

    // 6a. Construct agent cache (needed for resource tracking)
    agent::construct_agent_cache(table);

    // 6b. Initialize queue controller (for queue interception)
    hsa::queue_controller_init(table);

    // 6c. Initialize code object tracking
    code_object::initialize(table);

    // Step 7: Install wrapper functions based on registered contexts
    ROCP_TRACE << "Installing HSA wrapper functions";
    hsa::update_table(table->core_, lib_instance);
    hsa::update_table(table->amd_ext_, lib_instance);
    hsa::update_table(table->image_ext_, lib_instance);
    hsa::update_table(table->finalizer_ext_, lib_instance);
    if(table->tools_)
    {
        hsa::update_table(table->tools_, lib_instance);
    }

    // Step 8: Notify tools about table registration
    ROCP_TRACE << "Notifying tools about HSA table registration";
    intercept_table::notify_intercept_table_registration(
        ROCPROFILER_HSA_TABLE, lib_version, lib_instance, std::make_tuple(table));

    ROCP_INFO << "HSA API tables successfully wrapped for late-start profiling";
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
wrap_hip_tables(runtime_info& hip_info)
{
    ROCP_INFO << "Wrapping HIP API tables for late-start profiling";

    // Use instance 0 for late-start (first/only instance)
    constexpr uint64_t lib_instance = 0;
    constexpr uint64_t lib_version  = HIP_VERSION;

    bool any_table_wrapped = false;

    //--------------------------------------------------------------------------
    // Step 1: Wrap HIP Runtime API Table
    //--------------------------------------------------------------------------
    ROCP_TRACE << "Processing HIP runtime dispatch table";

    // 1a. Get GetHipDispatchTable function from runtime
    using get_runtime_table_fn = HipDispatchTable* (*) ();
    auto get_runtime_table =
        reinterpret_cast<get_runtime_table_fn>(dlsym(hip_info.handle, "GetHipDispatchTable"));

    if(get_runtime_table)
    {
        ROCP_TRACE << "Found GetHipDispatchTable symbol";

        // 1b. Get the runtime dispatch table from HIP runtime
        HipDispatchTable* runtime_table = get_runtime_table();

        if(runtime_table)
        {
            ROCP_TRACE << "Retrieved HIP runtime dispatch table";

            // 1c. Save the table pointer for potential restoration
            saved_hip_table = runtime_table;

            // 1d. Copy original function pointers to internal storage
            ROCP_TRACE << "Copying original HIP runtime function pointers";
            hip::copy_table(runtime_table, lib_instance);

            // 1e. Install wrapper functions based on registered contexts
            ROCP_TRACE << "Installing HIP runtime wrapper functions";
            hip::update_table(runtime_table);

            // 1f. Notify tools about table registration
            ROCP_TRACE << "Notifying tools about HIP runtime table registration";
            intercept_table::notify_intercept_table_registration(ROCPROFILER_HIP_RUNTIME_TABLE,
                                                                 lib_version,
                                                                 lib_instance,
                                                                 std::make_tuple(runtime_table));

            ROCP_INFO << "HIP runtime dispatch table successfully wrapped";
            any_table_wrapped = true;
        }
        else
        {
            ROCP_WARNING << "GetHipDispatchTable returned null table";
        }
    }
    else
    {
        ROCP_TRACE << "GetHipDispatchTable symbol not found";
    }

    //--------------------------------------------------------------------------
    // Step 2: Wrap HIP Compiler API Table
    //--------------------------------------------------------------------------
    ROCP_TRACE << "Processing HIP compiler dispatch table";

    // 2a. Get GetHipCompilerDispatchTable function from runtime
    using get_compiler_table_fn = HipCompilerDispatchTable* (*) ();
    auto get_compiler_table     = reinterpret_cast<get_compiler_table_fn>(
        dlsym(hip_info.handle, "GetHipCompilerDispatchTable"));

    if(get_compiler_table)
    {
        ROCP_TRACE << "Found GetHipCompilerDispatchTable symbol";

        // 2b. Get the compiler dispatch table from HIP runtime
        HipCompilerDispatchTable* compiler_table = get_compiler_table();

        if(compiler_table)
        {
            ROCP_TRACE << "Retrieved HIP compiler dispatch table";

            // 2c. Save the table pointer for potential restoration
            saved_hip_compiler_table = compiler_table;

            // 2d. Copy original function pointers to internal storage
            ROCP_TRACE << "Copying original HIP compiler function pointers";
            hip::copy_table(compiler_table, lib_instance);

            // 2e. Install wrapper functions based on registered contexts
            ROCP_TRACE << "Installing HIP compiler wrapper functions";
            hip::update_table(compiler_table);

            // 2f. Notify tools about table registration
            ROCP_TRACE << "Notifying tools about HIP compiler table registration";
            intercept_table::notify_intercept_table_registration(ROCPROFILER_HIP_COMPILER_TABLE,
                                                                 lib_version,
                                                                 lib_instance,
                                                                 std::make_tuple(compiler_table));

            ROCP_INFO << "HIP compiler dispatch table successfully wrapped";
            any_table_wrapped = true;
        }
        else
        {
            ROCP_WARNING << "GetHipCompilerDispatchTable returned null table";
        }
    }
    else
    {
        ROCP_TRACE << "GetHipCompilerDispatchTable symbol not found";
    }

    //--------------------------------------------------------------------------
    // Step 3: Return status
    //--------------------------------------------------------------------------
    if(!any_table_wrapped)
    {
        ROCP_ERROR << "Failed to wrap any HIP dispatch tables";
        return ROCPROFILER_STATUS_ERROR_HIP_NOT_AVAILABLE;
    }

    ROCP_INFO << "HIP API tables successfully wrapped for late-start profiling";
    return ROCPROFILER_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------
// State Query Functions
//------------------------------------------------------------------------------

int
get_state()
{
    return late_start_state.load();
}

bool
is_active()
{
    return late_start_state.load() == 2;
}

rocprofiler_status_t
restore_tables()
{
    // NOTE: This function assumes the caller has already acquired late_start_mutex
    // NOTE: This function assumes the caller has already verified state == 2

    ROCP_INFO << "Restoring runtime tables";

    // Step 1: Restore HSA tables (if wrapped)
    if(saved_hsa_table)
    {
        ROCP_TRACE << "Restoring HSA API tables";

        // Restore all HSA sub-tables that were wrapped
        if(saved_hsa_table->core_)
        {
            ROCP_TRACE << "Restoring HSA core table";
            hsa::restore_table(saved_hsa_table->core_);
        }

        if(saved_hsa_table->amd_ext_)
        {
            ROCP_TRACE << "Restoring HSA AMD extension table";
            hsa::restore_table(saved_hsa_table->amd_ext_);
        }

        if(saved_hsa_table->image_ext_)
        {
            ROCP_TRACE << "Restoring HSA image extension table";
            hsa::restore_table(saved_hsa_table->image_ext_);
        }

        if(saved_hsa_table->finalizer_ext_)
        {
            ROCP_TRACE << "Restoring HSA finalizer extension table";
            hsa::restore_table(saved_hsa_table->finalizer_ext_);
        }

        if(saved_hsa_table->tools_)
        {
            ROCP_TRACE << "Restoring HSA tools table";
            hsa::restore_table(saved_hsa_table->tools_);
        }

        // Clear saved table pointer
        saved_hsa_table = nullptr;
        ROCP_INFO << "HSA API tables restored";
    }

    // Step 2: Restore HIP runtime table (if wrapped)
    if(saved_hip_table)
    {
        ROCP_TRACE << "Restoring HIP runtime dispatch table";
        hip::restore_table(saved_hip_table);

        // Clear saved table pointer
        saved_hip_table = nullptr;
        ROCP_INFO << "HIP runtime dispatch table restored";
    }

    // Step 3: Restore HIP compiler table (if wrapped)
    if(saved_hip_compiler_table)
    {
        ROCP_TRACE << "Restoring HIP compiler dispatch table";
        hip::restore_table(saved_hip_compiler_table);

        // Clear saved table pointer
        saved_hip_compiler_table = nullptr;
        ROCP_INFO << "HIP compiler dispatch table restored";
    }

    // Step 4: Reset state to 0 (not started)
    late_start_state.store(0);
    ROCP_INFO << "Late-start profiling stopped - original functions restored";

    return ROCPROFILER_STATUS_SUCCESS;
}

}  // namespace late_start
}  // namespace rocprofiler

//------------------------------------------------------------------------------
// Internal API Implementations (for unit tests)
//------------------------------------------------------------------------------

extern "C"
{
__attribute__((visibility("default"))) rocprofiler_status_t
rocprofiler_start_late_internal(uint32_t flags)
{
    using namespace rocprofiler::late_start;

    ROCP_INFO << "rocprofiler_start_late_internal() called with flags=0x" << std::hex << flags;

    // Ensure only one thread does late-start
    std::lock_guard<std::mutex> lock(late_start_mutex);

    // Check state with atomic compare-exchange (0 → 1)
    int expected = 0;
    if(!late_start_state.compare_exchange_strong(expected, 1))
    {
        if(expected == 2)
        {
            // Already late-started
            ROCP_ERROR << "Late-start already completed";
            return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;
        }
        // Another thread is currently starting (state == 1), wait for it
        ROCP_WARNING << "Another thread is performing late-start, waiting...";
        while(late_start_state.load() == 1)
        {
            std::this_thread::yield();
        }
        // The other thread completed, return success
        return ROCPROFILER_STATUS_SUCCESS;
    }

    // State is now 1 (starting). If we fail, we'll reset to 0. On success, set to 2.

    // Step 1: Handle AUTO flag - set flags to ALL if AUTO is set
    bool auto_mode = (flags & ROCPROFILER_LATE_START_AUTO) != 0;
    if(auto_mode)
    {
        ROCP_TRACE << "AUTO flag detected, enabling all runtime flags";
        flags = ROCPROFILER_LATE_START_ALL;
    }

    rocprofiler_status_t status      = ROCPROFILER_STATUS_SUCCESS;
    bool                 any_wrapped = false;

    // Step 2: Ensure rocprofiler is configured
    if(rocprofiler::registration::get_init_status() == 0)
    {
        ROCP_TRACE << "rocprofiler not yet configured, invoking client configures";
        // Force configuration if not already done
        rocprofiler::registration::initialize();
    }

    // Step 3: Wrap HSA if requested and available
    if(flags & ROCPROFILER_LATE_START_HSA)
    {
        ROCP_TRACE << "HSA flag set, detecting HSA runtime";
        auto hsa_info = detect_hsa_runtime();

        if(hsa_info.initialized)
        {
            ROCP_TRACE << "HSA runtime detected and initialized, wrapping tables";
            status = wrap_hsa_tables(hsa_info);

            if(status == ROCPROFILER_STATUS_SUCCESS)
            {
                ROCP_INFO << "HSA tables successfully wrapped";
                any_wrapped = true;
            }
            else
            {
                ROCP_ERROR << "Failed to wrap HSA tables: " << status;
            }
        }
        else if(!auto_mode)
        {
            // HSA was explicitly requested but not available
            ROCP_ERROR << "HSA runtime requested but not available (not loaded or not "
                          "initialized)";
            late_start_state.store(0);
            return ROCPROFILER_STATUS_ERROR_HSA_NOT_AVAILABLE;
        }
        else
        {
            ROCP_TRACE << "HSA runtime not available (AUTO mode, continuing with other "
                          "runtimes)";
        }
    }

    // Step 4: Wrap HIP if requested and available
    if(flags & ROCPROFILER_LATE_START_HIP)
    {
        ROCP_TRACE << "HIP flag set, detecting HIP runtime";
        auto hip_info = detect_hip_runtime();

        if(hip_info.initialized)
        {
            ROCP_TRACE << "HIP runtime detected and initialized, wrapping tables";
            status = wrap_hip_tables(hip_info);

            if(status == ROCPROFILER_STATUS_SUCCESS)
            {
                ROCP_INFO << "HIP tables successfully wrapped";
                any_wrapped = true;
            }
            else
            {
                ROCP_ERROR << "Failed to wrap HIP tables: " << status;
            }
        }
        else if(!auto_mode)
        {
            // HIP was explicitly requested but not available
            ROCP_ERROR << "HIP runtime requested but not available (not loaded or not "
                          "initialized)";
            late_start_state.store(0);
            return ROCPROFILER_STATUS_ERROR_HIP_NOT_AVAILABLE;
        }
        else
        {
            ROCP_TRACE << "HIP runtime not available (AUTO mode, continuing)";
        }
    }

    // Step 5: Check if at least one runtime was wrapped
    if(!any_wrapped)
    {
        ROCP_ERROR << "No runtimes were wrapped (no runtimes available or none requested)";
        late_start_state.store(0);
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }

    // Step 6: Success - mark as started (state 1 → 2)
    late_start_state.store(2);
    ROCP_INFO << "Late-start completed successfully - profiling is now active";
    return ROCPROFILER_STATUS_SUCCESS;
}

__attribute__((visibility("default"))) rocprofiler_status_t
rocprofiler_is_late_start_internal(int* is_late_start)
{
    using namespace rocprofiler::late_start;

    // Validate pointer argument
    if(!is_late_start)
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    // Return current late-start state (2 = started, 0 or 1 = not started)
    *is_late_start = (late_start_state.load() == 2) ? 1 : 0;

    return ROCPROFILER_STATUS_SUCCESS;
}

__attribute__((visibility("default"))) rocprofiler_status_t
rocprofiler_stop_late_internal(void)
{
    using namespace rocprofiler::late_start;

    ROCP_INFO << "rocprofiler_stop_late_internal() called";

    // Quick state check before acquiring lock (fail-fast for invalid calls)
    if(late_start_state.load() != 2)
    {
        ROCP_ERROR << "stop_late called but late-start not active (state="
                   << late_start_state.load() << ")";
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }

    // Thread-safe state modification
    std::lock_guard<std::mutex> lock(late_start_mutex);

    // Double-check state after acquiring lock (TOCTOU protection)
    if(late_start_state.load() != 2)
    {
        ROCP_ERROR << "stop_late: state changed before lock acquisition";
        return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
    }

    return restore_tables();
}

}  // extern "C"
