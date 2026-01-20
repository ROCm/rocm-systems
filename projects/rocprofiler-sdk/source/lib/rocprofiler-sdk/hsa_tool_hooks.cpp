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
 * @file hsa_tool_hooks.cpp
 * @brief HSA tool hook functions called by HSA runtime during lifecycle events.
 *
 * This file exports symbols that HSA runtime discovers via dlsym() to notify
 * profiling tools of lifecycle events. The critical function is OnUnload(),
 * which is called by HSA before destroying the async signal handler thread.
 */

#include "lib/rocprofiler-sdk/hsa_tool_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/async_copy.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include "lib/common/logging.hpp"

#include <atomic>

namespace rocprofiler
{
namespace hsa_tool_hooks
{
namespace
{
// Track whether OnUnload has been called
std::atomic<int> on_unload_status{0};  // 0=not called, -1=in progress, 1=completed
}  // namespace

int
get_on_unload_status()
{
    return on_unload_status.load(std::memory_order_acquire);
}

void
set_on_unload_status(int status)
{
    on_unload_status.store(status, std::memory_order_release);
}

bool
sync_all_async_operations()
{
    ROCP_INFO << "Synchronizing all async operations before HSA unload...";

    // Sync async memory copies - these use HSA signal handlers
    hsa::async_copy_sync();
    ROCP_INFO << "Async copy sync complete";

    // Sync queue controller - wait for kernel completion signals
    hsa::queue_controller_sync();
    ROCP_INFO << "Queue controller sync complete";

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    // Sync PC sampling service - flush any pending samples
    pc_sampling::service_sync();
    ROCP_INFO << "PC sampling sync complete";
#endif

    ROCP_INFO << "All async operations synchronized";
    return true;
}

}  // namespace hsa_tool_hooks
}  // namespace rocprofiler

/**
 * @brief HSA tool unload hook.
 *
 * This function is called by HSA runtime before destroying the async signal
 * handler thread. This is the last opportunity to wait for async operations
 * that depend on the HSA async handler thread.
 *
 * Important: This function is called BEFORE hsa_shut_down() completes, while
 * HSA is still functional. After this returns, the async thread will be killed.
 *
 * The function must:
 * 1. Synchronize all pending async memory copy operations
 * 2. Synchronize all pending kernel dispatch completion callbacks
 * 3. Flush any pending PC sampling data
 * 4. NOT perform full finalization (that happens at atexit)
 *
 * @note This symbol is discovered by HSA runtime via dlsym(RTLD_DEFAULT, "OnUnload")
 */
extern "C" __attribute__((visibility("default"))) void
OnUnload()
{
    using namespace rocprofiler;

    // Guard against multiple calls
    int expected = 0;
    if(!hsa_tool_hooks::on_unload_status.compare_exchange_strong(
           expected, -1, std::memory_order_acq_rel))
    {
        ROCP_INFO << "OnUnload already called or in progress (status=" << expected << ")";
        return;
    }

    ROCP_INFO << "OnUnload called by HSA runtime - synchronizing async operations";

    // Check if we've already finalized
    if(registration::get_fini_status() > 0)
    {
        ROCP_INFO << "OnUnload: rocprofiler already finalized, nothing to sync";
        hsa_tool_hooks::set_on_unload_status(1);
        return;
    }

    // Check if initialization ever happened
    if(registration::get_init_status() < 1)
    {
        ROCP_INFO << "OnUnload: rocprofiler not initialized, nothing to sync";
        hsa_tool_hooks::set_on_unload_status(1);
        return;
    }

    // Synchronize all async operations while HSA async thread is still alive
    bool sync_success = hsa_tool_hooks::sync_all_async_operations();

    if(!sync_success)
    {
        ROCP_CI_LOG(WARNING) << "OnUnload: async operation synchronization timed out";
    }

    // Mark OnUnload as completed
    hsa_tool_hooks::set_on_unload_status(1);

    ROCP_INFO << "OnUnload completed successfully";
}
