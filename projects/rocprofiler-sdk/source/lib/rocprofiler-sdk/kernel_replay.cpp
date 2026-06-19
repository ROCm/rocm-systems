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

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/hsa/memory_tracker.hpp"

#include <rocprofiler-sdk/kernel_replay.h>

#include <memory>

extern "C" {

rocprofiler_status_t
rocprofiler_configure_kernel_replay_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void*                                      dispatch_callback_args,
    rocprofiler_dispatch_counting_record_cb_t  record_callback,
    void*                                      record_callback_args)
{
    if(!dispatch_callback || !record_callback) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if(ctx->kernel_replay) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;


     // TODO[amd-vkale]: an optimization but should we allow kernel replay  without counter collection? Could be useful for warmups of replay.  <-- Mythreya and Vivek's question 
    if(ctx->dispatch_counter_collection) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    auto st = rocprofiler::counters::configure_callback_dispatch(context_id,
                                                                 dispatch_callback,
                                                                 dispatch_callback_args,
                                                                 record_callback,
                                                                 record_callback_args);
    if(st != ROCPROFILER_STATUS_SUCCESS) return st;

    ctx->kernel_replay = std::make_unique<rocprofiler::context::kernel_replay_service>();
    ctx->kernel_replay->enabled.wlock([](bool& v) { v = true; });

    // Turn on the memory tracker so allocation/free hooks begin populating the inventory that
    // snap/restore consumes. Until this point the installed hooks are a single relaxed atomic load
    // past the chained call.
    rocprofiler::hsa::memory_tracker::set_tracking_enabled(true);

    return ROCPROFILER_STATUS_SUCCESS;
}

}  // extern "C"
