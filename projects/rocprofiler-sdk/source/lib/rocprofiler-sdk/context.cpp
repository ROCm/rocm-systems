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
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/counters/device_counting.hpp"
#include "lib/rocprofiler-sdk/counters/sample_processing.hpp"
#include "lib/rocprofiler-sdk/hsa/async_copy.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less_gate.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <atomic>
#include <vector>

namespace
{
constexpr auto rocprofiler_context_none = ROCPROFILER_CONTEXT_NONE;

uint64_t
synch_on_context(const rocprofiler_context_id_t context, const uint64_t timeout_ns)
{
    auto outstanding_records = rocprofiler::hsa::async_copy_sync(timeout_ns);
    outstanding_records += rocprofiler::hsa::queue_controller_sync(timeout_ns);
    outstanding_records += rocprofiler::kfd::drain_signal_less_completions(timeout_ns);

    // Ordered after the dispatch drains, not beside them: a dispatch's counter sample is handed
    // off while its completion is being handled, so a drained dispatch is one whose sample is
    // already counted. Draining first would wait on a ring that is still filling.
    outstanding_records += rocprofiler::counters::callback_thread_drain(timeout_ns);

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

// Deliver the records now sitting in every buffer this context owns. Goes through the public
// rocprofiler_flush_buffer rather than buffer::flush because that is the only one of the two that
// also drains the PC sampling agent buffers.
//
// Buffer ids are snapshotted before any flush: a flush runs the tool's buffer callback on this
// thread, and a callback that creates or destroys a context would deadlock against a registry
// held across it.
rocprofiler_status_t
flush_context_buffers(const rocprofiler_context_id_t context)
{
    auto _buffer_ids = rocprofiler::common::container::small_vector<rocprofiler_buffer_id_t, 8>{};
    if(const auto* _buffers = rocprofiler::buffer::get_buffers())
    {
        for(const auto& itr : *_buffers)
            if(itr && itr->context_id == context.handle)
                _buffer_ids.emplace_back(rocprofiler_buffer_id_t{itr->buffer_id});
    }

    // Every buffer is flushed even after one fails, so a buffer that went away cannot hide the
    // records sitting in the ones behind it. The caller is told about the first real failure.
    auto _status = ROCPROFILER_STATUS_SUCCESS;
    for(const auto& itr : _buffer_ids)
    {
        auto _itr_status = rocprofiler_flush_buffer(itr);
        // Destroyed between the snapshot and the flush: it has no records left to deliver.
        if(_itr_status == ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND) continue;
        if(_itr_status != ROCPROFILER_STATUS_SUCCESS && _status == ROCPROFILER_STATUS_SUCCESS)
            _status = _itr_status;
    }
    return _status;
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
    if(context == rocprofiler_context_none ||
       !rocprofiler::context::get_registered_context(context))
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // Cleared up front so the runtime-written fields are readable on every status returned
    // below. size is the caller's input and is left alone.
    if(result != nullptr)
    {
        result->num_records         = 0;
        result->num_timeout_records = 0;
    }

    // Finalization has already run these drains.
    if(rocprofiler::registration::get_fini_status() > 0) return ROCPROFILER_STATUS_SUCCESS;

    auto context_flags = uint64_t{0};
    auto timeout_ns    = uint64_t{0};
    if(options != nullptr)
    {
        context_flags = options->flags;
        timeout_ns    = options->timeout_ns;
    }

    // flags is a bitset, so every test below masks. Comparing it outright silently inverts both
    // flags the moment a caller combines them.
    const auto _nohang  = (context_flags & ROCPROFILER_SYNCHRONIZE_CONTEXT_NOHANG) != 0;
    const auto _noflush = (context_flags & ROCPROFILER_SYNCHRONIZE_CONTEXT_NOFLUSH) != 0;

    // NOHANG only populates the result, so without one there is nothing to ask for.
    if(result == nullptr && _nohang) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto return_status = ROCPROFILER_STATUS_SUCCESS;
    if(result != nullptr)
    {
        result->num_records = peek_outstanding_records(context);
    }

    if(!_nohang)
    {
        auto outstanding_records = synch_on_context(context, timeout_ns);
        if(outstanding_records > 0)
        {
            return_status = ROCPROFILER_STATUS_ERROR_TIMEOUT;
        }
        if(result != nullptr)
        {
            // Best-effort estimate of what the wait did not reach.
            result->num_timeout_records = outstanding_records;
        }
    }

    // Flushed even when the wait timed out: the records that did land belong to the caller, and
    // the status already reports the shortfall, so a flush failure must not displace it.
    if(!_noflush)
    {
        auto _flush_status = flush_context_buffers(context);
        if(_flush_status != ROCPROFILER_STATUS_SUCCESS &&
           return_status == ROCPROFILER_STATUS_SUCCESS)
            return_status = _flush_status;
    }
    return return_status;
}
}
