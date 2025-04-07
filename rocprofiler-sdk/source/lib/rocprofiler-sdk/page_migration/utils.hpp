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

#pragma once

#include "lib/common/container/small_vector.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace page_migration
{
/* serves as an overview of what events we capture and report

struct event_page_fault_start_t
{
    int      kind;
    uint64_t timestamp;
    int      pid;
    int      node_id;
    uint64_t address;
    fault_t  fault;
};

struct event_page_fault_end_t
{
    int      kind;
    uint64_t timestamp;
    uint32_t pid;
    int      node_id;
    uint64_t address;
    bool     migrated;
};

struct event_migrate_start_t
{
    int      kind;
    uint64_t timestamp;
    uint32_t pid;
    uint64_t start;
    uint64_t end_offset;
    uint32_t from;
    uint32_t to;
    uint32_t prefetch_node;   // last prefetch location, 0 for CPU, or GPU id
    uint32_t preferred_node;  // perferred location, 0 for CPU, or GPU id
    uint32_t trigger;
};

struct event_migrate_end_t
{
    int      kind;
    uint64_t timestamp;
    uint32_t pid;
    uint64_t start;
    uint64_t end_offset;
    uint32_t from;
    uint32_t to;
    uint32_t trigger;
};

struct event_queue_eviction_t
{
    int      kind;
    uint64_t timestamp;
    uint32_t pid;
    int      node_id;
    uint32_t trigger;
};

struct event_queue_restore_t
{
    int      kind;
    uint64_t timestamp;
    uint32_t pid;
    int      node_id;
    bool     rescheduled;
};

struct event_unmap_from_gpu_t
{
    int      kind;
    uint64_t timestamp;
    uint32_t pid;
    uint64_t address;
    uint64_t size;
    int      node_id;
    uint32_t trigger;
};
*/

template <size_t>
struct page_migration_info;

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
    return (page_migration_info<Ints>::kfd_bitmask | ...);
}

template <size_t OpInx, size_t... OpInxs>
constexpr size_t
kfd_bitmask_impl(size_t rocprof_op, std::index_sequence<OpInx, OpInxs...>)
{
    if(rocprof_op == OpInx) return page_migration_info<OpInx>::kfd_bitmask;
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
kfd_to_rocprof_op(size_t kfd_id, std::index_sequence<OpInx, OpInxs...>)
{
    if(kfd_id == page_migration_info<OpInx>::kfd_operation) return OpInx;
    if constexpr(sizeof...(OpInxs) > 0)
        return kfd_to_rocprof_op(kfd_id, std::index_sequence<OpInxs...>{});
    else
        return 0;
}

size_t
get_rocprof_op(const std::string_view event_data);

void
kfd_readlines(const std::string_view str, void(handler)(std::string_view));

using rocprof_buffer_op_t = rocprofiler_page_migration_operation_t;

using node_fd_t = int;

}  // namespace page_migration
}  // namespace rocprofiler
