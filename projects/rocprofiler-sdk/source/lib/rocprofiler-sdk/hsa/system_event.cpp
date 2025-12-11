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

#include "lib/rocprofiler-sdk/hsa/system_event.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/hsa/async_copy.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

namespace rocprofiler
{
namespace hsa
{
namespace
{
// Track whether HSA is shutting down
std::atomic<bool>& get_hsa_shutdown_flag()
{
    static std::atomic<bool> _v{false};
    return _v;
}

hsa_status_t
system_event_handler(const hsa_amd_event_t* event, void* /*data*/)
{
    if(event && event->event_type == HSA_AMD_GPU_MEMORY_FAULT_EVENT)
    {
        ROCP_ERROR << "HSA AMD GPU memory fault event detected";
        return HSA_STATUS_ERROR;
    }

    if(event && event->event_type == HSA_AMD_SYSTEM_SHUTDOWN_EVENT)
    {
        ROCP_INFO << "HSA AMD system shutdown event detected - marking HSA as shutting down";

        // Set shutdown flag BEFORE attempting any sync operations
        // This prevents the sync functions from trying to wait on HSA signals
        // since HSA's async threads are already being shut down
        get_hsa_shutdown_flag().store(true, std::memory_order_release);

        // Note: We do NOT call async_copy_sync() or queue_controller_sync() here
        // because HSA is already shutting down and the async threads that service
        // the signal waits are being destroyed. Calling these would hang.
    }

    return HSA_STATUS_SUCCESS;
}
}  // namespace

bool
is_hsa_shutting_down()
{
    return get_hsa_shutdown_flag().load(std::memory_order_acquire);
}

void
system_event_init(hsa_api_table_t* _orig, uint64_t _tbl_instance)
{
    if(_tbl_instance > 0) return;  // Only register once

    if(!_orig || !_orig->amd_ext_) return;

    auto* amd_ext_table = _orig->amd_ext_;
    if(!amd_ext_table->hsa_amd_register_system_event_handler_fn)
    {
        ROCP_WARNING << "hsa_amd_register_system_event_handler not available";
        return;
    }

    auto status = amd_ext_table->hsa_amd_register_system_event_handler_fn(
        system_event_handler, nullptr);

    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << "Failed to register HSA system event handler: " << status;
    }
    else
    {
        ROCP_INFO << "Successfully registered HSA system event handler for shutdown events";
    }
}
}  // namespace hsa
}  // namespace rocprofiler
