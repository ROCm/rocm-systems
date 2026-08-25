// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"

#include <memory>

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace context
{
struct correlation_id;
}
namespace hip
{
namespace event
{
const char*
name_by_id(uint32_t id);

uint32_t
id_by_name(const char* name);

std::vector<const char*>
get_names();

std::vector<uint32_t>
get_ids();

using profiling_time = tracing::profiling_time;

void
barrier_complete(tracing::tracing_data&                        tracing_data_v,
                 rocprofiler_thread_id_t                       tid,
                 uint64_t                                      internal_corr_id,
                 uint64_t                                      ancestor_corr_id,
                 profiling_time                                barrier_time,
                 rocprofiler_hip_event_operation_t             operation,
                 rocprofiler_callback_tracing_hip_event_data_t callback_record);

struct active_event_context_t
{
    rocprofiler_hip_event_operation_t operation        = ROCPROFILER_HIP_EVENT_NONE;
    uint64_t                          hip_event_handle = 0;
    bool                              barrier_captured = false;
};

active_event_context_t*
get_active_event_context();

struct event_record_info_t
{
    rocprofiler_queue_id_t queue_id        = {.handle = 0};
    rocprofiler_agent_id_t agent_id        = {.handle = 0};
    uint64_t               original_signal = 0;
};

void
record_event_info(uint64_t hip_event_handle, event_record_info_t info);

event_record_info_t
lookup_event_info(uint64_t hip_event_handle);

struct coalesce_pending_t
{
    tracing::tracing_data                         tracing_data     = {};
    rocprofiler_callback_tracing_hip_event_data_t callback_record  = {};
    rocprofiler_thread_id_t                       tid              = 0;
    uint64_t                                      internal_corr_id = 0;
    uint64_t                                      ancestor_corr_id = 0;
    context::correlation_id*                      corr_id_ref      = nullptr;
};

struct coalesce_group_t
{
    profiling_time                                         barrier_time = {};
    bool                                                   completed    = false;
    common::container::small_vector<coalesce_pending_t, 4> pending      = {};
};

using coalesce_group_ptr_t = std::shared_ptr<common::Synchronized<coalesce_group_t>>;

void
store_coalesce_group(uint64_t hip_event_handle, coalesce_group_ptr_t group);

coalesce_group_ptr_t
lookup_coalesce_group(uint64_t hip_event_handle);

void
erase_event_info(uint64_t hip_event_handle);

struct pending_wait_t
{
    tracing::tracing_data                         tracing_data     = {};
    rocprofiler_callback_tracing_hip_event_data_t callback_record  = {};
    rocprofiler_thread_id_t                       tid              = 0;
    uint64_t                                      internal_corr_id = 0;
    uint64_t                                      ancestor_corr_id = 0;
    context::correlation_id*                      corr_id_ref      = nullptr;
};

void
register_pending_wait(uint64_t signal_handle, pending_wait_t pw);

bool
has_pending_waits();

pending_wait_t
consume_pending_wait(uint64_t signal_handle);

template <typename TableT>
void
update_table(TableT* table);

}  // namespace event
}  // namespace hip
}  // namespace rocprofiler
