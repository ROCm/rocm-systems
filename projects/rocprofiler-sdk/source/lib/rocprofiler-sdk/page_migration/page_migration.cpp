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

#include "lib/rocprofiler-sdk/page_migration/page_migration.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/page_migration/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/page_migration/utils.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa/api_id.h>
#include <rocprofiler-sdk/hsa/table_id.h>

#include <fmt/core.h>
#include <glog/logging.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>

#include <sys/poll.h>
#include <unistd.h>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>

#define ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL 1
#include "page_migration.def.cpp"
#undef ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL

namespace rocprofiler
{
namespace page_migration
{
template <typename T>
using small_vector = common::container::small_vector<T>;

using context_t               = context::context;
using context_array_t         = common::container::small_vector<const context_t*>;
using kfd_event_id_t          = decltype(KFD_SMI_EVENT_NONE);
using page_migration_record_t = rocprofiler_buffer_tracing_page_migration_record_t;
using migrate_trigger_t       = rocprofiler_page_migration_trigger_t;
using qsuspend_trigger_t      = rocprofiler_page_migration_queue_suspend_trigger_t;
using unmap_trigger_t         = rocprofiler_page_migration_unmap_from_gpu_trigger_t;

// Parsing and utilities
namespace
{
using namespace page_migration;

constexpr auto
page_to_bytes(size_t val)
{
    // each page is 4KB = 4096 bytes
    return val << 12;
}

template <typename EnumT, int ValueE>
struct page_migration_enum_info;

template <typename EnumT>
struct page_migration_bounds;

#define SPECIALIZE_PM_ENUM_INFO(TYPE, TRIGGER_CATEGORY, NAME)                                      \
    template <>                                                                                    \
    struct page_migration_enum_info<TYPE, ROCPROFILER_PAGE_MIGRATION_##TRIGGER_CATEGORY##_##NAME>  \
    {                                                                                              \
        static constexpr auto name = #NAME;                                                        \
    };

#define SPECIALIZE_PM_ENUM_BOUNDS(TYPE, TRIGGER_CATEGORY)                                          \
    template <>                                                                                    \
    struct page_migration_bounds<TYPE>                                                             \
    {                                                                                              \
        static constexpr auto last = ROCPROFILER_PAGE_MIGRATION_##TRIGGER_CATEGORY##_LAST;         \
    };

using queue_suspend_trigger_t  = rocprofiler_page_migration_queue_suspend_trigger_t;
using unmap_from_gpu_trigger_t = rocprofiler_page_migration_unmap_from_gpu_trigger_t;

SPECIALIZE_PM_ENUM_BOUNDS(rocprofiler_page_migration_trigger_t, TRIGGER)
SPECIALIZE_PM_ENUM_BOUNDS(queue_suspend_trigger_t, QUEUE_SUSPEND_TRIGGER)
SPECIALIZE_PM_ENUM_BOUNDS(unmap_from_gpu_trigger_t, UNMAP_FROM_GPU_TRIGGER)

SPECIALIZE_PM_ENUM_INFO(rocprofiler_page_migration_trigger_t, TRIGGER, PREFETCH)
SPECIALIZE_PM_ENUM_INFO(rocprofiler_page_migration_trigger_t, TRIGGER, PAGEFAULT_GPU)
SPECIALIZE_PM_ENUM_INFO(rocprofiler_page_migration_trigger_t, TRIGGER, PAGEFAULT_CPU)
SPECIALIZE_PM_ENUM_INFO(rocprofiler_page_migration_trigger_t, TRIGGER, TTM_EVICTION)

SPECIALIZE_PM_ENUM_INFO(queue_suspend_trigger_t, QUEUE_SUSPEND_TRIGGER, SVM)
SPECIALIZE_PM_ENUM_INFO(queue_suspend_trigger_t, QUEUE_SUSPEND_TRIGGER, USERPTR)
SPECIALIZE_PM_ENUM_INFO(queue_suspend_trigger_t, QUEUE_SUSPEND_TRIGGER, TTM)
SPECIALIZE_PM_ENUM_INFO(queue_suspend_trigger_t, QUEUE_SUSPEND_TRIGGER, SUSPEND)
SPECIALIZE_PM_ENUM_INFO(queue_suspend_trigger_t, QUEUE_SUSPEND_TRIGGER, CRIU_CHECKPOINT)
SPECIALIZE_PM_ENUM_INFO(queue_suspend_trigger_t, QUEUE_SUSPEND_TRIGGER, CRIU_RESTORE)

SPECIALIZE_PM_ENUM_INFO(unmap_from_gpu_trigger_t, UNMAP_FROM_GPU_TRIGGER, MMU_NOTIFY)
SPECIALIZE_PM_ENUM_INFO(unmap_from_gpu_trigger_t, UNMAP_FROM_GPU_TRIGGER, MMU_NOTIFY_MIGRATE)
SPECIALIZE_PM_ENUM_INFO(unmap_from_gpu_trigger_t, UNMAP_FROM_GPU_TRIGGER, UNMAP_FROM_CPU)

using trigger_type_list_t = common::mpl::type_list<rocprofiler_page_migration_trigger_t,
                                                   queue_suspend_trigger_t,
                                                   unmap_from_gpu_trigger_t>;

template <typename EnumT, size_t Idx, size_t... IdxTail>
std::string_view
to_string_impl(EnumT val, std::index_sequence<Idx, IdxTail...>)
{
    if(val == Idx) return page_migration_enum_info<EnumT, Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return to_string_impl(val, std::index_sequence<IdxTail...>{});
    else
        return std::string_view{};
}

template <typename EnumT, typename Up = EnumT>
std::string_view
to_string(EnumT val,
          std::enable_if_t<std::is_enum<Up>::value &&
                               common::mpl::is_one_of<Up, trigger_type_list_t>::value,
                           int> = 0)
{
    constexpr auto last = page_migration_bounds<EnumT>::last;
    return to_string_impl(val, std::make_index_sequence<last>{});
}

template <size_t>
page_migration_record_t parse_uvm_event(std::string_view)
{
    LOG_IF(FATAL, false) << uvm_event_info<ROCPROFILER_UVM_EVENT_NONE>::format_str;
    return {};
}

template <>
page_migration_record_t
parse_uvm_event<ROCPROFILER_UVM_EVENT_PAGE_FAULT_START>(std::string_view str)
{
    page_migration_record_t rec{};
    auto&                   e = rec.page_fault;
    uint32_t                kind{};

    char fault;
    std::sscanf(str.data(),
                uvm_event_info<ROCPROFILER_UVM_EVENT_PAGE_FAULT_START>::format_str.data(),
                &kind,
                &rec.start_timestamp,
                &rec.pid,
                &e.address,
                &e.node_id,
                &fault);

    e.read_fault = (fault == 'R');
    e.address    = page_to_bytes(e.address);

    LOG(INFO) << fmt::format("Page fault start [ ts: {} pid: {} addr: 0x{:X} node: {} ] \n",
                             rec.start_timestamp,
                             rec.pid,
                             e.address,
                             e.node_id);

    return rec;
}

template <>
page_migration_record_t
parse_uvm_event<ROCPROFILER_UVM_EVENT_PAGE_FAULT_END>(std::string_view str)
{
    page_migration_record_t rec{};
    auto&                   e = rec.page_fault;
    uint32_t                kind{};

    char migrated;
    std::sscanf(str.data(),
                uvm_event_info<ROCPROFILER_UVM_EVENT_PAGE_FAULT_END>::format_str.data(),
                &kind,
                &rec.end_timestamp,
                &rec.pid,
                &e.address,
                &e.node_id,
                &migrated);

    // M or U -> migrated / unmigrated?
    if(migrated == 'M')
        e.migrated = true;
    else if(migrated == 'U')
        e.migrated = false;
    // else
    // throw std::runtime_error("Invalid SVM memory migrate type");
    e.address = page_to_bytes(e.address);

    LOG(INFO) << fmt::format(
        "Page fault end [ ts: {} pid: {} addr: 0x{:X} node: {} migrated: {} ] \n",
        rec.end_timestamp,
        rec.pid,
        e.address,
        e.node_id,
        migrated);

    return rec;
}

template <>
page_migration_record_t
parse_uvm_event<ROCPROFILER_UVM_EVENT_MIGRATE_START>(std::string_view str)
{
    page_migration_record_t rec{};
    auto&                   e = rec.page_migrate;
    uint32_t                kind{};
    uint32_t                trigger{};

    std::sscanf(str.data(),
                uvm_event_info<ROCPROFILER_UVM_EVENT_MIGRATE_START>::format_str.data(),
                &kind,
                &rec.start_timestamp,
                &rec.pid,
                &e.start_addr,
                &e.end_addr,
                &e.from_node,
                &e.to_node,
                &e.prefetch_node,
                &e.preferred_node,
                &trigger);

    e.end_addr += e.start_addr;
    e.trigger    = static_cast<migrate_trigger_t>(trigger);
    e.start_addr = page_to_bytes(e.start_addr);
    e.end_addr   = page_to_bytes(e.end_addr) - 1;

    LOG(INFO) << fmt::format(
        "Page migrate start [ ts: {} pid: {} addr s: 0x{:X} addr "
        "e: 0x{:X} size: {}B from node: {} to node: {} prefetch node: {} preferred node: {} "
        "trigger: {} ] \n",
        rec.start_timestamp,
        rec.pid,
        e.start_addr,
        e.end_addr,
        (e.end_addr - e.start_addr),
        e.from_node,
        e.to_node,
        e.prefetch_node,
        e.preferred_node,
        to_string(e.trigger));

    return rec;
}

template <>
page_migration_record_t
parse_uvm_event<ROCPROFILER_UVM_EVENT_MIGRATE_END>(std::string_view str)
{
    page_migration_record_t rec{};
    auto&                   e = rec.page_migrate;
    uint32_t                kind{};
    uint32_t                trigger{};

    std::sscanf(str.data(),
                uvm_event_info<ROCPROFILER_UVM_EVENT_MIGRATE_END>::format_str.data(),
                &kind,
                &rec.end_timestamp,
                &rec.pid,
                &e.start_addr,
                &e.end_addr,
                &e.from_node,
                &e.to_node,
                &trigger);

    e.end_addr += e.start_addr;
    e.trigger    = static_cast<migrate_trigger_t>(trigger);
    e.start_addr = page_to_bytes(e.start_addr);
    e.end_addr   = page_to_bytes(e.end_addr) - 1;

    LOG(INFO) << fmt::format("Page migrate end [ ts: {} pid: {} addr s: 0x{:X} addr e: "
                             "0x{:X} from node: {} to node: {} trigger: {} ] \n",
                             rec.end_timestamp,
                             rec.pid,
                             e.start_addr,
                             e.end_addr,
                             e.from_node,
                             e.to_node,
                             to_string(e.trigger));

    return rec;
}

template <>
page_migration_record_t
parse_uvm_event<ROCPROFILER_UVM_EVENT_QUEUE_EVICTION>(std::string_view str)
{
    page_migration_record_t rec{};
    auto&                   e = rec.queue_suspend;
    uint32_t                kind{};
    uint32_t                trigger{};

    std::sscanf(str.data(),
                uvm_event_info<ROCPROFILER_UVM_EVENT_QUEUE_EVICTION>::format_str.data(),
                &kind,
                &rec.start_timestamp,
                &rec.pid,
                &e.node_id,
                &trigger);

    rec.queue_suspend.trigger = static_cast<qsuspend_trigger_t>(trigger);

    LOG(INFO) << fmt::format("Queue evict [ ts: {} pid: {} node: {} trigger: {} ] \n",
                             rec.start_timestamp,
                             rec.pid,
                             e.node_id,
                             to_string(e.trigger));

    return rec;
}

template <>
page_migration_record_t
parse_uvm_event<ROCPROFILER_UVM_EVENT_QUEUE_RESTORE>(std::string_view str)
{
    page_migration_record_t rec{};
    auto&                   e = rec.queue_suspend;
    uint32_t                kind{};

    std::sscanf(str.data(),
                uvm_event_info<ROCPROFILER_UVM_EVENT_QUEUE_RESTORE>::format_str.data(),
                &kind,
                &rec.end_timestamp,
                &rec.pid,
                &e.node_id);
    // check if we have a valid char at the end. -1 has \0
    if(str[str.size() - 2] == 'R')
        e.rescheduled = true;
    else
        e.rescheduled = false;

    LOG(INFO) << fmt::format(
        "Queue restore [ ts: {} pid: {} node: {} ] \n", rec.end_timestamp, rec.pid, e.node_id);

    return rec;
}

template <>
page_migration_record_t
parse_uvm_event<ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU>(std::string_view str)
{
    page_migration_record_t rec{};
    auto&                   e = rec.unmap_from_gpu;
    uint32_t                kind{};
    uint32_t                trigger{};

    std::sscanf(str.data(),
                uvm_event_info<ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU>::format_str.data(),
                &kind,
                &rec.start_timestamp,
                &rec.pid,
                &e.start_addr,
                &e.end_addr,
                &e.node_id,
                &trigger);

    e.end_addr += e.start_addr;
    rec.end_timestamp          = rec.start_timestamp;
    rec.unmap_from_gpu.trigger = static_cast<unmap_trigger_t>(trigger);
    e.start_addr               = page_to_bytes(e.start_addr);
    e.end_addr                 = page_to_bytes(e.end_addr);

    LOG(INFO) << fmt::format("Unmap from GPU [ ts: {} pid: {} start addr: 0x{:X} end addr: 0x{:X}  "
                             "node: {} trigger {} ] \n",
                             rec.start_timestamp,
                             rec.pid,
                             e.start_addr,
                             e.end_addr,
                             e.node_id,
                             to_string(e.trigger));

    return rec;
}

template <size_t OpInx, size_t... OpInxs>
page_migration_record_t
parse_uvm_event(uvm_event_id_t   event_id,
                std::string_view strn,
                std::index_sequence<OpInx, OpInxs...>)
{
    if(OpInx == static_cast<uint32_t>(event_id))
    {
        auto rec      = parse_uvm_event<OpInx>(strn);
        rec.size      = sizeof(page_migration_record_t);
        rec.kind      = ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION;
        rec.operation = to_rocprof_op(OpInx);
        return rec;
    }
    else if constexpr(sizeof...(OpInxs) > 0)
        return parse_uvm_event(event_id, strn, std::index_sequence<OpInxs...>{});
    else
        return page_migration_record_t{};
}

/* -----------------------------------------------------------------------------------*/

template <size_t OpInx>
void
update_end(const page_migration_record_t& start, page_migration_record_t& end);

template <>
void
update_end<ROCPROFILER_UVM_EVENT_PAGE_FAULT_END>(const page_migration_record_t& start,
                                                 page_migration_record_t&       end)
{
    CHECK(start.pid == end.pid);
    CHECK(start.page_fault.address == end.page_fault.address);
    CHECK(start.page_fault.node_id == end.page_fault.node_id);
    COPY_FROM_START_1(start_timestamp);
    COPY_FROM_START_2(page_fault, migrated);
}

template <>
void
update_end<ROCPROFILER_UVM_EVENT_MIGRATE_END>(const page_migration_record_t& start,
                                              page_migration_record_t&       end)
{
    CHECK(start.pid == end.pid);
    CHECK(start.page_migrate.start_addr == end.page_migrate.start_addr);
    CHECK(start.page_migrate.end_addr == end.page_migrate.end_addr);
    CHECK(start.page_migrate.from_node == end.page_migrate.from_node);
    CHECK(start.page_migrate.to_node == end.page_migrate.to_node);
    CHECK(start.page_migrate.trigger == end.page_migrate.trigger);
    COPY_FROM_START_1(start_timestamp);
    COPY_FROM_START_2(page_migrate, prefetch_node);
    COPY_FROM_START_2(page_migrate, preferred_node);
}

template <>
void
update_end<ROCPROFILER_UVM_EVENT_QUEUE_RESTORE>(const page_migration_record_t& start,
                                                page_migration_record_t&       end)
{
    CHECK(start.pid == end.pid);
    CHECK(start.queue_suspend.node_id == end.queue_suspend.node_id);
    COPY_FROM_START_1(start_timestamp);
    COPY_FROM_START_2(queue_suspend, trigger);
}

/* -----------------------------------------------------------------------------------*/

template <rocprofiler_page_migration_operation_t>
uint64_t
get_key(const rocprofiler_buffer_tracing_page_migration_record_t& rec) = delete;

template <>
uint64_t
get_key<ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE>(
    const rocprofiler_buffer_tracing_page_migration_record_t& rec)
{
    // page migrate, use address as identifier
    return rec.page_migrate.start_addr;
}

template <>
uint64_t
get_key<ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT>(
    const rocprofiler_buffer_tracing_page_migration_record_t& rec)
{
    // page fault, use address as identifier
    return rec.page_fault.address;
}

template <>
uint64_t
get_key<ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND>(
    const rocprofiler_buffer_tracing_page_migration_record_t& rec)
{
    // Queue suspend/evict. Node ID and pid are sufficient as in kfd,
    // eviction is reference-counted per process-device.
    uint64_t node_id = rec.queue_suspend.node_id;
    return (node_id << 32) | rec.pid;
}

/* -----------------------------------------------------------------------------------*/

template <>
page_migration_record_t parse_uvm_event<0>(std::string_view)
{
    throw std::runtime_error("None Op for parsing UVM events should not happen");
}

template <>
void
update_end<ROCPROFILER_UVM_EVENT_NONE>(const page_migration_record_t&, page_migration_record_t&)
{
    throw std::runtime_error("None Op for parsing UVM events should not happen");
}

template <>
uint64_t
get_key<ROCPROFILER_PAGE_MIGRATION_NONE>(const page_migration_record_t&)
{
    throw std::runtime_error("None Op for parsing UVM events should not happen");
}

/* -----------------------------------------------------------------------------------*/

template <size_t OpInx, size_t... OpInxs>
void
update_end(uvm_event_id_t                 event_id,
           const page_migration_record_t& start,
           page_migration_record_t&       end,
           std::index_sequence<OpInx, OpInxs...>)
{
    if(OpInx == static_cast<uint32_t>(event_id))
        update_end<OpInx>(start, end);
    else if constexpr(sizeof...(OpInxs) > 0)
        update_end(event_id, start, end, std::index_sequence<OpInxs...>{});
    else
        return;
}

template <size_t OpInx, size_t... OpInxs>
uint64_t
get_key(uvm_event_id_t                 event_id,
        const page_migration_record_t& record,
        std::index_sequence<OpInx, OpInxs...>)
{
    if constexpr(OpInx == ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU)
        return {};
    else if(is_rocprof_uvm_map<OpInx>(event_id))
        return get_key<page_migration_info<OpInx>::operation_idx>(record);
    else if constexpr(sizeof...(OpInxs) > 0)
        return get_key(event_id, record, std::index_sequence<OpInxs...>{});
    else
        return {};
}

void
update_end(uvm_event_id_t                 event_id,
           const page_migration_record_t& start,
           page_migration_record_t&       end)
{
    update_end(event_id,
               start,
               end,
               std::index_sequence<ROCPROFILER_UVM_EVENT_NONE,
                                   ROCPROFILER_UVM_EVENT_MIGRATE_END,
                                   ROCPROFILER_UVM_EVENT_PAGE_FAULT_END,
                                   ROCPROFILER_UVM_EVENT_QUEUE_RESTORE>{});
}
}  // namespace

// Event capture and reporting
namespace
{
// Support seems to have been added in kfdv > 1.10
static_assert(KFD_IOCTL_MAJOR_VERSION == 1, "KFD API major version changed");
static_assert(KFD_IOCTL_MINOR_VERSION >= 10, "KFD SMI support missing in kfd_ioctl.h");

// Convert from public events to KFD enum config

template <size_t OpInx, size_t... OpInxs>
constexpr size_t
kfd_bitmask_impl(size_t uvm_event_id, std::index_sequence<OpInx, OpInxs...>)
{
    if(uvm_event_id == OpInx) return page_migration_info<OpInx>::kfd_bitmask;
    if constexpr(sizeof...(OpInxs) > 0)
        return kfd_bitmask_impl(uvm_event_id, std::index_sequence<OpInxs...>{});
    else
        return 0;
}

template <size_t... OpInxs>
constexpr auto
kfd_bitmask(const small_vector<size_t>& rocprof_event_ids, std::index_sequence<OpInxs...>)
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
to_uvm_op_impl(size_t kfd_id, std::index_sequence<OpInx, OpInxs...>)
{
    // if(kfd_id == uvm_event_info<OpInx>::kfd_event) return uvm_event_info<OpInx>::uvm_event;
    if(kfd_id == uvm_event_info<OpInx>::kfd_event) return OpInx;
    if constexpr(sizeof...(OpInxs) > 0)
        return to_uvm_op_impl(kfd_id, std::index_sequence<OpInxs...>{});
    else
        return 0;
}

constexpr uvm_event_id_t
kfd_to_uvm_op(kfd_event_id_t kfd_id)
{
    return static_cast<uvm_event_id_t>(
        to_uvm_op_impl(kfd_id, std::make_index_sequence<ROCPROFILER_UVM_EVENT_LAST>{}));
}

struct buffered_context_data
{
    const context::context* ctx = nullptr;
};

void
populate_contexts(int operation_idx, std::vector<buffered_context_data>& buffered_contexts)
{
    buffered_contexts.clear();

    auto active_contexts = context::context_array_t{};
    for(const auto* itr : context::get_active_contexts(active_contexts))
    {
        if(itr->buffered_tracer)
        {
            // if the given domain + op is not enabled, skip this context
            if(itr->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION,
                                             operation_idx))
                buffered_contexts.emplace_back(buffered_context_data{itr});
        }
    }
}

void
remove_events(events_cache_t& events, size_t timestamp)
{
    for(auto map : events)
    {
        for(auto i = map.begin(); i != map.end(); ++i)
        {
            if(i->second.start_timestamp < timestamp) map.erase(i);
        }
    }
}

bool
report_event(uvm_event_id_t                                      event_id,
             rocprofiler_buffer_tracing_page_migration_record_t& end_record)
{
    using rocprofiler_page_migr_seq = std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>;
    static thread_local events_cache_t EVENTS_CACHE{};

    auto& events_map = EVENTS_CACHE[to_rocprof_op(event_id)];

    switch(static_cast<uint32_t>(event_id))
    {
        case ROCPROFILER_UVM_EVENT_MIGRATE_START: [[fallthrough]];
        case ROCPROFILER_UVM_EVENT_PAGE_FAULT_START: [[fallthrough]];
        case ROCPROFILER_UVM_EVENT_QUEUE_EVICTION:
        {
            // insert into map
            auto key        = get_key(event_id, end_record, rocprofiler_page_migr_seq{});
            events_map[key] = end_record;
            return false;
        }
        // End events. Pair up and report
        case ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU:
        {
            return true;
        }
        case ROCPROFILER_UVM_EVENT_MIGRATE_END: [[fallthrough]];
        case ROCPROFILER_UVM_EVENT_PAGE_FAULT_END: [[fallthrough]];
        case ROCPROFILER_UVM_EVENT_QUEUE_RESTORE:
        {
            auto key = get_key(event_id, end_record, rocprofiler_page_migr_seq{});
            if(auto start_rec = events_map.find(key); start_rec != events_map.end())
            {
                update_end(event_id, start_rec->second, end_record);
            }
            else
            {
                // we got an end record and can't find the start record
                // drop everything in the map before this timestamp
                remove_events(EVENTS_CACHE, end_record.end_timestamp);
            }
            return true;
        }
        default: throw std::runtime_error("Invalid page migration event");
    }
}

void
handle_reporting(std::string_view event_data)
{
    uint32_t kfd_event_id;
    std::sscanf(event_data.data(), "%x ", &kfd_event_id);
    std::vector<buffered_context_data> buffered_contexts{};

    auto uvm_event_op = kfd_to_uvm_op(static_cast<kfd_event_id_t>(kfd_event_id));

    populate_contexts(uvm_event_op, buffered_contexts);
    if(buffered_contexts.empty()) return;

    // Parse and process the event
    auto record = parse_uvm_event(
        uvm_event_op, event_data, std::make_index_sequence<ROCPROFILER_UVM_EVENT_LAST>{});

    // pair up start and end and only then insert it into the buffer
    if(report_event(uvm_event_op, record))
    {
        for(const auto& itr : buffered_contexts)
        {
            auto* _buffer = buffer::get_buffer(itr.ctx->buffered_tracer->buffer_data.at(
                ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION));
            CHECK_NOTNULL(_buffer)->emplace(ROCPROFILER_BUFFER_CATEGORY_TRACING,
                                            ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION,
                                            record);
        }
    }
}

}  // namespace

// KFD utils
namespace kfd
{
void
poll_events(small_vector<pollfd>, bool);

using fd_flags_t = decltype(EFD_NONBLOCK);
using fd_t       = decltype(pollfd::fd);
constexpr auto KFD_DEVICE_PATH{"/dev/kfd"};

SPECIALIZE_KFD_IOC_IOCTL(kfd_ioctl_get_version_args, AMDKFD_IOC_GET_VERSION);
SPECIALIZE_KFD_IOC_IOCTL(kfd_ioctl_smi_events_args, AMDKFD_IOC_SMI_EVENTS);

namespace
{
template <typename T>
auto
ioctl(int kfd_fd, T& args)
{
    // from hsaKmt library (hsakmt/src/libhsakmt.c)
    int exit_code{};

    do
    {
        exit_code = ::ioctl(kfd_fd, IOC_event<T>::value, static_cast<void*>(&args));
    } while(exit_code == -1 && (errno == EINTR || errno == EAGAIN));

    if(exit_code == -1 && errno == EBADF)
    {
        /* In case pthread_atfork didn't catch it, this will
         * make any subsequent hsaKmt calls fail in CHECK_KFD_OPEN.
         */
        CHECK(true && "KFD file descriptor not valid in this process\n");
    }
    return exit_code;
}

struct kfd_device_fd
{
    fd_t fd{-1};

    kfd_device_fd()
    {
        fd = ::open(KFD_DEVICE_PATH, O_RDWR | O_CLOEXEC);
        LOG_IF(FATAL, fd == -1) << "Error opening KFD handle @ " << KFD_DEVICE_PATH;
    }

    ~kfd_device_fd()
    {
        if(fd >= 0) close(fd);
    }
};

const kfd_ioctl_get_version_args
get_version()
{
    static kfd_ioctl_get_version_args version = [&]() {
        auto          args = kfd_ioctl_get_version_args{0, 0};
        kfd_device_fd kfd_fd{};

        if(ioctl(kfd_fd.fd, args) != -1)
            LOG(INFO) << fmt::format("KFD v{}.{}", args.major_version, args.minor_version);
        else
            LOG(ERROR) << fmt::format("Could not determine KFD version");
        return args;
    }();

    return version;
}

struct poll_kfd_t
{
    static constexpr auto DEFAULT_FLAGS{EFD_CLOEXEC};

    struct gpu_fd_t
    {
        unsigned int               node_id{};
        fd_t                       fd{};
        const rocprofiler_agent_t* agent{};
    };

    kfd_device_fd        kfd_fd{};
    small_vector<pollfd> file_handles{};
    pollfd               thread_notify{};
    std::thread          bg_thread;
    bool                 active{false};

    poll_kfd_t() = default;

    poll_kfd_t(const small_vector<size_t>& rprof_ev, bool non_blocking)
    : kfd_fd{kfd_device_fd{}}
    {
        const auto kfd_flags =
            kfd_bitmask(rprof_ev, std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});

        LOG(INFO) << fmt::format("Setting KFD flags to [0b{:b}] \n", kfd_flags);

        // Create fd for notifying thread when we want to wake it up, and an eventfd for any events
        // to this thread
        file_handles.emplace_back(pollfd{
            .fd = eventfd(0, DEFAULT_FLAGS),
        });

        fd_t thread_pipes[2]{};

        [&]() {
            const auto retcode = pipe2(&thread_pipes[0], DEFAULT_FLAGS);

            if(retcode != 0)
                throw std::runtime_error{
                    fmt::format("Pipe creation for thread notify failed with {} code\n", retcode)};
        }();

        thread_notify = pollfd{
            .fd      = thread_pipes[1],
            .events  = POLLIN,
            .revents = 0,
        };

        // add pipe listening end to fds to watch
        file_handles.emplace_back(pollfd{thread_pipes[0], POLLIN, 0});

        // get FD, start thread, and then enable events
        for(const auto& agent : agent::get_agents())
        {
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                auto gpu_event_fd = get_node_fd(agent->gpu_id);
                file_handles.emplace_back(pollfd{gpu_event_fd, POLLIN, 0});
                LOG(INFO) << fmt::format(
                    "GPU node {} with fd {} added\n", agent->gpu_id, gpu_event_fd);
            }
        }

        // Enable KFD masked events by writing flags to kfd fd
        for(size_t i = 2; i < file_handles.size(); ++i)
        {
            auto& fd         = file_handles[i];
            auto  write_size = write(fd.fd, &kfd_flags, sizeof(kfd_flags));
            LOG(INFO) << fmt::format(
                "Writing {} to GPU fd {} ({} bytes)\n", kfd_flags, fd.fd, write_size);
            CHECK(write_size == sizeof(kfd_flags));
        }

        // start bg thread
        internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
        bg_thread = std::thread{poll_events, file_handles, non_blocking};
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

        active = true;
    }

    inline static auto get_event_id(const std::string_view& strn)
    {
        uint32_t event_id{std::numeric_limits<uint32_t>::max()};
        std::sscanf(strn.data(), "%x ", &event_id);

        CHECK(event_id <= KFD_SMI_EVENT_ALL_PROCESS);
    }

    poll_kfd_t(const poll_kfd_t&) = delete;
    poll_kfd_t& operator=(const poll_kfd_t&) = delete;

    poll_kfd_t(poll_kfd_t&&) = default;
    poll_kfd_t& operator=(poll_kfd_t&&) = default;

    ~poll_kfd_t();

    node_fd_t get_node_fd(int gpu_node_id) const
    {
        kfd_ioctl_smi_events_args args{};
        args.gpuid = gpu_node_id;

        if(auto ret = ioctl(kfd_fd.fd, args); ret == -1)
            LOG(ERROR) << fmt::format(
                "Could not get GPU node {} file descriptor (exit code: {})", gpu_node_id, ret);
        return args.anon_fd;
    }
};

// for all contexts
struct page_migration_config
{
    bool should_exit() const { return m_should_exit.load(); }
    void set_exit(bool val) { m_should_exit.store(val); }

    uint64_t         enabled_events = 0;
    kfd::poll_kfd_t* kfd_handle     = nullptr;

private:
    std::atomic<bool> m_should_exit = false;
};

page_migration_config&
get_config()
{
    static auto& state = *common::static_object<page_migration_config>::construct();
    return state;
}

kfd::poll_kfd_t::~poll_kfd_t()
{
    LOG(INFO) << fmt::format("Terminating poll_kfd\n");
    if(!active) return;

    // wake thread up
    kfd::get_config().set_exit(true);
    auto bytes_written{-1};
    do
    {
        bytes_written = write(thread_notify.fd, "E", 1);
    } while(bytes_written == -1 && (errno == EINTR || errno == EAGAIN));

    if(bg_thread.joinable()) bg_thread.join();
    LOG(INFO) << fmt::format("Background thread terminated\n");

    for(const auto& f : file_handles)
        close(f.fd);
}
}  // namespace

void
poll_events(small_vector<pollfd> file_handles, bool non_blocking)
{
    // storage to write records to, 1MB
    constexpr size_t PREALLOCATE_ELEMENT_COUNT{1024 * 128};
    std::string      scratch_buffer(PREALLOCATE_ELEMENT_COUNT, '\0');
    auto&            exitfd      = file_handles[1];
    const auto       timeout_val = non_blocking == true ? 0 : -1;

    // Wait or spin on events.
    //  0 -> return immediately even if no events
    // -1 -> wait indefinitely

    LOG(INFO) << fmt::format("{} polling = {}, polling with timeout = {}",
                             non_blocking ? "Non-blocking" : "Blocking",
                             non_blocking,
                             timeout_val);

    pthread_setname_np(pthread_self(), "bg:pagemigr");

    for(auto& fd : file_handles)
    {
        LOG(INFO) << fmt::format(
            "Handle = {}, events = {}, revents = {}\n", fd.fd, fd.events, fd.revents);
    }

    while(!kfd::get_config().should_exit())
    {
        auto poll_ret = poll(file_handles.data(), file_handles.size(), timeout_val);

        if(poll_ret == -1)
            throw std::runtime_error{"Background thread file descriptors are invalid"};

        if((exitfd.revents & POLLIN) != 0)
        {
            LOG(INFO) << "Terminating background thread\n";
            return;
        }

        using namespace std::chrono_literals;

        for(size_t i = 2; i < file_handles.size(); ++i)
        {
            auto& fd = file_handles[i];

            // We have data to read, perhaps multiple events
            if((fd.revents & POLLIN) != 0)
            {
                size_t status_size = read(fd.fd, scratch_buffer.data(), scratch_buffer.size());

                // LOG(INFO) << fmt::format(
                //     "status_size: {} size {}\n", status_size, scratch_buffer.size());
                std::string_view event_strings{scratch_buffer.data(), status_size};

                // LOG(INFO) << fmt::format("Raw KFD string [({})]\n",
                // event_strings.data());
                KFD_EVENT_PARSE_EVENTS(event_strings, handle_reporting);
            }
            fd.revents = 0;
        }
    }
}
}  // namespace kfd

template <size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return page_migration_info<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST)) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, page_migration_info<Idx>::operation_idx), ...);
}

bool
context_filter(const context::context* ctx)
{
    return (ctx->buffered_tracer &&
            (ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION)));
}

template <size_t... Idx>
void
to_bitmask(small_vector<size_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST)) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, page_migration_info<Idx>::operation_idx), ...);
}

namespace
{
rocprofiler_status_t
init(const small_vector<size_t>& event_ids, bool non_blocking)
{
    // Check if version is more than 1.11
    auto ver = kfd::get_version();
    if(ver.major_version * 1000 + ver.minor_version > 1011)
    {
        if(!context::get_registered_contexts(context_filter).empty())
        {
            if(!kfd::get_config().kfd_handle)
                kfd::get_config().kfd_handle = new kfd::poll_kfd_t{event_ids, non_blocking};
        }
        return ROCPROFILER_STATUS_SUCCESS;
    }
    else
    {
        // Add a buffer record with this info
        LOG(ERROR) << fmt::format(
            "KFD does not support SVM event reporting in v{}.{} (requires v1.11)",
            ver.major_version,
            ver.minor_version);
        return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
    }
}
}  // namespace

rocprofiler_status_t
init()
{
    // Testing page migration
    return init({ROCPROFILER_PAGE_MIGRATION_NONE,
                 ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT,
                 ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE,
                 ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND,
                 ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU},
                rocprofiler::common::get_env("ROCPROF_PAGE_MIGRATION_NON_BLOCKING", false));
}

void
finalize()
{
    if(kfd::get_config().kfd_handle)
    {
        kfd::poll_kfd_t* _handle = nullptr;
        std::swap(kfd::get_config().kfd_handle, _handle);
        delete _handle;
    }
}

const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_PAGE_MIGRATION_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});
    return _data;
}
}  // namespace page_migration
}  // namespace rocprofiler
