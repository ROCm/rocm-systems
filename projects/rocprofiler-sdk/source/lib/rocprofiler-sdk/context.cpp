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

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/common/container/small_vector.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/counters/device_counting.hpp"
#include "lib/rocprofiler-sdk/counters/sample_processing.hpp"
#include "lib/rocprofiler-sdk/hsa/async_copy.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <algorithm>
#include <atomic>
#include <vector>

namespace
{
constexpr auto rocprofiler_context_none = ROCPROFILER_CONTEXT_NONE;

uint64_t
synch_on_context(const rocprofiler_context_id_t context, const uint64_t timeout_ns)
{
    // The copy drain reports whether it completed. queue_controller_sync reports only its
    // per-queue waits: its completion-monitor wait has no deadline and no return value, so a
    // monitor that never drains cannot show up in _dispatches_drained.
    auto outstanding_records = rocprofiler::hsa::async_copy_sync(timeout_ns);
    outstanding_records += rocprofiler::hsa::queue_controller_sync(timeout_ns);

    // Every record in the ring copied, published, popped and processed, and every submitted
    // completion finished, with signal-less left running.
    outstanding_records += rocprofiler::kfd::drain_signal_less_completions(timeout_ns);
    // Ordered after the dispatch drains and not beside them: a dispatch's counter sample is
    // handed off while its completion is being handled, so a drained dispatch is one whose
    // sample is already counted. Draining first would wait on a ring that is still filling.
    outstanding_records += rocprofiler::counters::callback_thread_drain(timeout_ns);

    // Peform device counting for the specified context.
    const auto* _ctx = rocprofiler::context::get_registered_context(context);
    if(_ctx != nullptr && _ctx->device_counter_collection)
    {
        outstanding_records +=
            rocprofiler::counters::drain_device_counting_samples(_ctx, timeout_ns);
    }
    return outstanding_records;
}

uint64_t
peek_outstanding_records(const rocprofiler_context_id_t context)
{
    auto outstanding_records = rocprofiler::hsa::async_copy_peek_count();
    outstanding_records += rocprofiler::hsa::queue_controller_peek_active_async_packets();
    outstanding_records += rocprofiler::kfd::peek_active_signal_less_packets();
    outstanding_records += rocprofiler::counters::callback_thread_peek();

    const auto* _ctx = rocprofiler::context::get_registered_context(context);
    if(_ctx != nullptr && _ctx->device_counter_collection)
    {
        outstanding_records += rocprofiler::counters::peek_device_counting_samples(_ctx);
    }
    return outstanding_records;
}

}  // namespace

extern "C" {
rocprofiler_status_t
rocprofiler_create_context(rocprofiler_context_id_t* context_id)
{
    if(context_id->handle != 0) return ROCPROFILER_STATUS_ERROR_CONTEXT_ID_NOT_ZERO;
    // context already registered
    if(rocprofiler::context::get_registered_context(*context_id))
        return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    // always set to none first
    *context_id = rocprofiler_context_none;

    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto cfg_id = rocprofiler::context::allocate_context();
    if(!cfg_id) return ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR;
    *context_id = *cfg_id;
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_start_context(rocprofiler_context_id_t context_id)
{
    if(context_id == rocprofiler_context_none ||
       !rocprofiler::context::get_registered_context(context_id))
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // if currently finalizing or finalized, don't allow starting a context
    if(rocprofiler::registration::get_fini_status() != 0)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    return rocprofiler::context::start_context(context_id);
}

rocprofiler_status_t
rocprofiler_stop_context(rocprofiler_context_id_t context_id)
{
    if(context_id == rocprofiler_context_none ||
       !rocprofiler::context::get_registered_context(context_id))
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // if finalized, context is already stopped
    if(rocprofiler::registration::get_fini_status() > 0) return ROCPROFILER_STATUS_SUCCESS;

    return rocprofiler::context::stop_context(context_id);
}

rocprofiler_status_t
rocprofiler_context_is_active(rocprofiler_context_id_t context_id, int* status)
{
    // make sure that status is zero (if not active) regardless of initial value
    *status = 0;

    // return context not found if not registered
    if(context_id == rocprofiler_context_none ||
       !rocprofiler::context::get_registered_context(context_id))
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // set status to 1 if found in active context array
    for(const auto* itr : rocprofiler::context::get_active_contexts())
    {
        if(itr && itr->context_idx == context_id.handle)
        {
            *status = 1;
            return ROCPROFILER_STATUS_SUCCESS;
        }
    }

    // context was not found in active array so status value is still zero
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_context_is_valid(rocprofiler_context_id_t context_id, int* status)
{
    *status = 0;

    if(context_id == rocprofiler_context_none ||
       !rocprofiler::context::get_registered_context(context_id))
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    for(const auto& itr : rocprofiler::context::get_registered_contexts())
    {
        if(itr && itr->context_idx == context_id.handle)
        {
            auto _ret = rocprofiler::context::validate_context(itr);
            *status   = (_ret == ROCPROFILER_STATUS_SUCCESS) ? 1 : 0;
            return _ret;
        }
    }

    return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
}

rocprofiler_status_t
rocprofiler_synchronize_context(const rocprofiler_context_id_t                   context,
                                const rocprofiler_synchronize_context_options_t* options,
                                rocprofiler_synchronize_context_result_t*        result)
{
    // Return if context is not found or invalid
    if(context == rocprofiler_context_none ||
       !rocprofiler::context::get_registered_context(context))
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // Finalization has already run these drains.
    if(rocprofiler::registration::get_fini_status() > 0) return ROCPROFILER_STATUS_SUCCESS;

    // Extract options
    auto context_flags = uint64_t{0};
    auto timeout_ns    = uint64_t{0};
    if(options != nullptr)
    {
        context_flags = options->flags;
        timeout_ns    = options->timeout_ns;
    }
    if(result == nullptr && context_flags == ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG)
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    auto return_status = ROCPROFILER_STATUS_SUCCESS;
    // Get the number of existing records before any synch if required
    if(result != nullptr)
    {
        result->num_records = peek_outstanding_records(context);
    }

    // Perform synchronization only if ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG is not selected
    if(context_flags != ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG)
    {
        auto outstanding_records = synch_on_context(context, timeout_ns);
        if(outstanding_records > 0)
        {
            return_status = ROCPROFILER_STATUS_ERROR_TIMEOUT;
        }
        if(result != nullptr)
        {
            // If synched, return best effort estimate of remaining work after timeouts
            result->num_timeout_records = outstanding_records;
        }
    }
    // Flush unless requested not to
    if(context_flags != ROCPROFILER_SYNCHRONIZE_CONTEXT_NOFLUSH)
    {}
    return return_status;
}
}
