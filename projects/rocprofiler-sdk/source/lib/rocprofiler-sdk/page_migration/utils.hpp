// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/page_migration/details/kfd_ioctl.h"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace page_migration
{
// serves as an overview of what events we capture and report
enum fault_type_t
{
    NONE,
    READ,
    WRITE,
};

struct uvm_event_page_fault_start_t
{
    int          kind;
    uint64_t     start_timestamp;
    int          pid;
    int          node_id;
    uint64_t     address;
    fault_type_t fault;
};

struct uvm_event_page_fault_end_t
{
    int      kind;
    uint64_t end_timestamp;
    uint32_t pid;
    int      node_id;
    uint64_t address;
    bool     migrated;
};

struct uvm_event_migrate_start_t
{
    int      kind;
    uint64_t start_timestamp;
    uint32_t pid;
    uint64_t start;
    uint64_t end_offset;
    uint32_t from;
    uint32_t to;
    uint32_t prefetch_node;   // last prefetch location, 0 for CPU, or GPU id
    uint32_t preferred_node;  // perferred location, 0 for CPU, or GPU id
    uint32_t trigger;
};

struct uvm_event_migrate_end_t
{
    int      kind;
    uint64_t end_timestamp;
    uint32_t pid;
    uint64_t start;
    uint64_t end_offset;
    uint32_t from;
    uint32_t to;
    uint32_t trigger;
};

struct uvm_event_queue_eviction_t
{
    int      kind;
    uint64_t start_timestamp;
    uint32_t pid;
    int      node_id;
    uint32_t trigger;
};

struct uvm_event_queue_restore_t
{
    int      kind;
    uint64_t end_timestamp;
    uint32_t pid;
    int      node_id;
    bool     rescheduled;
};

struct uvm_event_unmap_from_gpu_t
{
    int      kind;
    uint64_t timestamp;
    uint32_t pid;
    uint64_t address;
    uint64_t size;
    int      node_id;
    uint32_t trigger;
};

template <size_t e>
struct uvm_event_info;

template <size_t>
struct page_migration_info;

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

enum uvm_event_id_t
{
    ROCPROFILER_UVM_EVENT_NONE,
    ROCPROFILER_UVM_EVENT_MIGRATE_START,
    ROCPROFILER_UVM_EVENT_MIGRATE_END,
    ROCPROFILER_UVM_EVENT_PAGE_FAULT_START,
    ROCPROFILER_UVM_EVENT_PAGE_FAULT_END,
    ROCPROFILER_UVM_EVENT_QUEUE_EVICTION,
    ROCPROFILER_UVM_EVENT_QUEUE_RESTORE,
    ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU,
    ROCPROFILER_UVM_EVENT_LAST,
};

static_assert(KFD_SMI_EVENT_NONE == 0);
static_assert(KFD_SMI_EVENT_MIGRATE_START == 5);
static_assert(KFD_SMI_EVENT_MIGRATE_END == 6);
static_assert(KFD_SMI_EVENT_PAGE_FAULT_START == 7);
static_assert(KFD_SMI_EVENT_PAGE_FAULT_END == 8);
static_assert(KFD_SMI_EVENT_QUEUE_EVICTION == 9);
static_assert(KFD_SMI_EVENT_QUEUE_RESTORE == 10);
static_assert(KFD_SMI_EVENT_UNMAP_FROM_GPU == 11);
static_assert(KFD_SMI_EVENT_ALL_PROCESS == 64);

using rocprof_buffer_op_t = rocprofiler_page_migration_operation_t;

using node_fd_t = int;

using event_map_t =
    std::unordered_map<uint64_t, rocprofiler_buffer_tracing_page_migration_record_t>;
using events_cache_t = std::array<event_map_t, ROCPROFILER_PAGE_MIGRATION_LAST>;

template <size_t... Ints>
constexpr size_t to_kfd_bitmask(std::index_sequence<Ints...>)
{
    return bitmask(std::index_sequence<uvm_event_info<Ints>::kfd_event...>());
}

template <rocprofiler_page_migration_operation_t... Ops>
constexpr size_t to_uvm_bitmask(std::index_sequence<Ops...>)
{
    return bitmask(std::index_sequence<static_cast<uint32_t>(Ops)...>());
}

template <size_t RocprofOpIdx, size_t UvmOpIdx>
constexpr bool
is_rocprof_uvm_map()
{
    return page_migration_info<RocprofOpIdx>::uvm_bitmask & bitmask(UvmOpIdx);
}

template <size_t RocprofOpIdx, size_t OpInx, size_t... OpInxs>
constexpr bool
_is_rocprof_uvm_map(size_t uvm_event, std::index_sequence<OpInx, OpInxs...>)
{
    if(OpInx == uvm_event)
        return is_rocprof_uvm_map<RocprofOpIdx, OpInx>();
    else if constexpr(sizeof...(OpInxs) > 0)
        return _is_rocprof_uvm_map<RocprofOpIdx>(uvm_event, std::index_sequence<OpInxs...>{});
    else
        return false;
}

template <size_t RocprofOpIdx>
constexpr bool
is_rocprof_uvm_map(size_t uvm_event)
{
    return _is_rocprof_uvm_map<RocprofOpIdx>(
        uvm_event, std::make_index_sequence<ROCPROFILER_UVM_EVENT_LAST>{});
}
}  // namespace page_migration
}  // namespace rocprofiler
