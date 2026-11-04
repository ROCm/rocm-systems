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
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <algorithm>
#include <atomic>
#include <vector>

namespace
{
constexpr auto rocprofiler_context_none = ROCPROFILER_CONTEXT_NONE;
}

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
rocprofiler_synchronize_contexts(const rocprofiler_context_id_t* contexts, size_t num_contexts)
{
    // Arguments are validated before the post-finalization shortcut below, so the contract does
    // not depend on when the call is made: a bogus context is rejected either way.
    for(size_t i = 0; i < num_contexts; ++i)
    {
        if(contexts[i] == rocprofiler_context_none ||
           !rocprofiler::context::get_registered_context(contexts[i]))
            return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
    }

    // Finalization has already run these drains.
    if(rocprofiler::registration::get_fini_status() > 0) return ROCPROFILER_STATUS_SUCCESS;

    // The copy drain reports whether it completed. queue_controller_sync reports only its
    // per-queue waits: its completion-monitor wait has no deadline and no return value, so a
    // monitor that never drains cannot show up in _dispatches_drained.
    bool _copies_drained     = rocprofiler::hsa::async_copy_sync();
    bool _dispatches_drained = rocprofiler::hsa::queue_controller_sync();

    // Every record in the ring copied, published, popped and processed, and every submitted
    // completion finished, with signal-less left running.
    bool _signal_less_drained = rocprofiler::kfd::drain_signal_less_completions();

    // Ordered after the dispatch drains and not beside them: a dispatch's counter sample is
    // handed off while its completion is being handled, so a drained dispatch is one whose
    // sample is already counted. Draining first would wait on a ring that is still filling.
    rocprofiler::counters::callback_thread_drain();

    // Both of the remaining waits are scoped to the named contexts, so a context the caller does
    // not name can never hold this call up. Device counting is per context; the PC sampling flush
    // is per client, so it is deduplicated on client_idx and several contexts owned by one client
    // flush the same agent buffers once rather than once each.
    auto _flushed_clients = rocprofiler::common::container::small_vector<uint32_t, 8>{};
    for(size_t i = 0; i < num_contexts; ++i)
    {
        const auto* _ctx = rocprofiler::context::get_registered_context(contexts[i]);
        if(_ctx == nullptr) continue;

        rocprofiler::counters::drain_device_counting_samples(_ctx);

        const auto _client_idx = _ctx->client_idx;
        if(std::find(_flushed_clients.begin(), _flushed_clients.end(), _client_idx) !=
           _flushed_clients.end())
            continue;
        _flushed_clients.emplace_back(_client_idx);

        rocprofiler::pc_sampling::service_sync(
            rocprofiler_client_id_t{sizeof(rocprofiler_client_id_t), nullptr, _client_idx});
    }

    return (_copies_drained && _dispatches_drained && _signal_less_drained)
               ? ROCPROFILER_STATUS_SUCCESS
               : ROCPROFILER_STATUS_ERROR_RECORDS_OUTSTANDING;
}
}
