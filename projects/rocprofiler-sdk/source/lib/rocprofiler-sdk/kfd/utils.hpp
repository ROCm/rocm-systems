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
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace kfd
{
enum rocprofiler_kfd_event_id
{
    KFD_EVENT_NONE = -1,
    KFD_EVENT_PAGE_MIGRATE_START,
    KFD_EVENT_PAGE_MIGRATE_END,
    KFD_EVENT_PAGE_FAULT_START,
    KFD_EVENT_PAGE_FAULT_END,
    KFD_EVENT_QUEUE_EVICTION,
    KFD_EVENT_QUEUE_RESTORE,
    KFD_EVENT_UNMAP_FROM_GPU,
    KFD_EVENT_DROPPED_EVENT,
    KFD_EVENT_LAST,
};

struct kfd_event_record
{
    rocprofiler_buffer_tracing_kind_t kind{ROCPROFILER_BUFFER_TRACING_NONE};
    int                               operation{-1};
    union
    {
        rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t   page_migrate_event;
        rocprofiler_buffer_tracing_kfd_event_page_fault_record_t     page_fault_event;
        rocprofiler_buffer_tracing_kfd_event_queue_record_t          queue_event;
        rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t unmap_event;
        rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t dropped_event;

        rocprofiler_buffer_tracing_kfd_page_migrate_record_t page_migrate_record;
        rocprofiler_buffer_tracing_kfd_page_fault_record_t   page_fault_record;
        rocprofiler_buffer_tracing_kfd_queue_record_t        queue_record;
    } data;
};

using agent_id_map_t = std::unordered_map<uint64_t, rocprofiler_agent_id_t>;

template <uint32_t>
struct kfd_event_info;

template <uint32_t>
struct kfd_kind_info;

template <uint32_t, uint32_t>
struct kfd_operation_info;

using namespace rocprofiler::common;

namespace kfd
{
template <typename T>
struct IOC_event;
}  // namespace kfd

constexpr size_t
bitmask(size_t num)
{
    if(num == 0)
        return 0;
    else
        return (1ULL << (num - 1));
}

template <size_t... Args>
constexpr size_t bitmask(std::index_sequence<Args...>)
{
    return (bitmask(Args) | ...);
}

template <size_t... Ints>
constexpr size_t kfd_bitmask(std::index_sequence<Ints...>)
{
    return (kfd_event_info<Ints>::kfd_bitmask | ...);
}

template <size_t OpInx, size_t... OpInxs>
constexpr size_t
kfd_bitmask_impl(size_t rocprof_op, std::index_sequence<OpInx, OpInxs...>)
{
    if(rocprof_op == OpInx) return kfd_event_info<OpInx>::kfd_bitmask;
    if constexpr(sizeof...(OpInxs) > 0)
        return kfd_bitmask_impl(rocprof_op, std::index_sequence<OpInxs...>{});
    else
        return 0;
}

template <size_t... OpInxs>
constexpr auto
kfd_bitmask(const container::small_vector<size_t>& rocprof_event_ids,
            std::index_sequence<OpInxs...>)
{
    uint64_t m{};
    for(const size_t& event_id : rocprof_event_ids)
    {
        m |= kfd_bitmask_impl(event_id, std::index_sequence<OpInxs...>{});
    }
    return m;
}

template <size_t OpInx, size_t... OpInxs>
constexpr size_t
to_rocprofiler_kfd_event_id_func(size_t kfd_id, std::index_sequence<OpInx, OpInxs...>)
{
    if(kfd_id == kfd_event_info<OpInx>::kfd_id) return OpInx;
    if constexpr(sizeof...(OpInxs) > 0)
        return to_rocprofiler_kfd_event_id_func(kfd_id, std::index_sequence<OpInxs...>{});
    else
        return std::numeric_limits<size_t>::max();
}

kfd_event_record
parse_event(size_t event_id, const agent_id_map_t& agents, std::string_view strn);

size_t
get_rocprof_op(const std::string_view event_data);

void
kfd_readlines(const std::string_view str, void(handler)(std::string_view));

using node_fd_t = int;

}  // namespace kfd
}  // namespace rocprofiler
