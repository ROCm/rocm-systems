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

#include "lib/rocprofiler-sdk/kfd/kfd.hpp"
#include "include/rocprofiler-sdk/kfd/kfd_id.h"
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/kfd/defines.hpp"
#include "lib/rocprofiler-sdk/kfd/utils.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa/api_id.h>
#include <rocprofiler-sdk/hsa/table_id.h>

#include <fmt/core.h>

#include <sys/poll.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <ratio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>

namespace rocprofiler
{
namespace kfd
{
template <typename T>
using small_vector = common::container::small_vector<T>;

using context_t       = context::context;
using context_array_t = common::container::small_vector<const context_t*>;

using page_migrate_event_record_t = rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t;
using page_fault_event_record_t   = rocprofiler_buffer_tracing_kfd_event_page_fault_record_t;
using queue_event_record_t        = rocprofiler_buffer_tracing_kfd_event_queue_record_t;
using page_migrate_record_t       = rocprofiler_buffer_tracing_kfd_page_migrate_record_t;
using page_fault_record_t         = rocprofiler_buffer_tracing_kfd_page_fault_record_t;
using queue_record_t              = rocprofiler_buffer_tracing_kfd_queue_record_t;

#define ROCPROFILER_LIB_ROCPROFILER_SDK_KFD_CPP_IMPL 1
#include "kfd.def.cpp"
#undef ROCPROFILER_LIB_ROCPROFILER_SDK_KFD_CPP_IMPL

// enum / info checks
namespace
{
using kfd_seq_t = std::make_index_sequence<KFD_EVENT_LAST>;

static_assert(kfd_bitmask(std::index_sequence<KFD_EVENT_PAGE_FAULT_START,
                                              KFD_EVENT_PAGE_MIGRATE_END,
                                              KFD_EVENT_UNMAP_FROM_GPU>()) ==
              (KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_PAGE_FAULT_START) |
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_MIGRATE_END) |
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_UNMAP_FROM_GPU)));

static_assert((kfd_event_info<KFD_EVENT_PAGE_MIGRATE_END>::kfd_bitmask |
               kfd_event_info<KFD_EVENT_QUEUE_EVICTION>::kfd_bitmask |
               kfd_event_info<KFD_EVENT_UNMAP_FROM_GPU>::kfd_bitmask) ==
              (KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_MIGRATE_END) |
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_QUEUE_EVICTION) |
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_UNMAP_FROM_GPU)));

static_assert(to_rocprofiler_kfd_event_id_func(KFD_SMI_EVENT_MIGRATE_START, kfd_seq_t{}) ==
              KFD_EVENT_PAGE_MIGRATE_START);
static_assert(to_rocprofiler_kfd_event_id_func(KFD_SMI_EVENT_MIGRATE_END, kfd_seq_t{}) ==
              KFD_EVENT_PAGE_MIGRATE_END);
static_assert(to_rocprofiler_kfd_event_id_func(KFD_SMI_EVENT_PAGE_FAULT_START, kfd_seq_t{}) ==
              KFD_EVENT_PAGE_FAULT_START);
static_assert(to_rocprofiler_kfd_event_id_func(KFD_SMI_EVENT_PAGE_FAULT_END, kfd_seq_t{}) ==
              KFD_EVENT_PAGE_FAULT_END);
static_assert(to_rocprofiler_kfd_event_id_func(KFD_SMI_EVENT_QUEUE_EVICTION, kfd_seq_t{}) ==
              KFD_EVENT_QUEUE_EVICTION);
static_assert(to_rocprofiler_kfd_event_id_func(KFD_SMI_EVENT_QUEUE_RESTORE, kfd_seq_t{}) ==
              KFD_EVENT_QUEUE_RESTORE);
static_assert(to_rocprofiler_kfd_event_id_func(KFD_SMI_EVENT_UNMAP_FROM_GPU, kfd_seq_t{}) ==
              KFD_EVENT_UNMAP_FROM_GPU);
}  // namespace

// Parsing and utilities
namespace
{
using page_migrate_start_ops_t =
    std::index_sequence<ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PREFETCH,
                        ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_GPU,
                        ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_CPU,
                        ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_TTM_EVICTION>;
using page_migrate_end_ops_t = std::index_sequence<ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_END>;
using page_fault_start_ops_t =
    std::index_sequence<ROCPROFILER_KFD_EVENT_PAGE_FAULT_START,
                        ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_READ_FAULT,
                        ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_WRITE_FAULT>;
using page_fault_end_ops_t = std::index_sequence<ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_MIGRATED,
                                                 ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_UPDATED>;
using queue_evict_ops_t    = std::index_sequence<ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SVM,
                                              ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR,
                                              ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM,
                                              ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND,
                                              ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT,
                                              ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE>;

using queue_restore_ops_t = std::index_sequence<ROCPROFILER_KFD_EVENT_QUEUE_RESTORE>;

constexpr auto
page_to_bytes(size_t val)
{
    // each page is 4KB = 4096 bytes
    return val << 12;
}

template <size_t>
kfd_event_record
parse_event(const agent_id_map_t&, std::string_view)
{
    ROCP_FATAL_IF(false) << "Invalid KFD event";
    return {};
}

auto
get_node_map()
{
    static auto*& _data = static_object<agent_id_map_t>::construct([]() {
        auto _v = agent_id_map_t{};
        for(const auto* agent : agent::get_agents())
            _v.emplace(agent->gpu_id, agent->id);
        return _v;
    }());

    return *_data;
}

auto
get_node_agent_id(const agent_id_map_t& agents, uint32_t _node_id)
{
    ROCP_FATAL_IF(agents.count(_node_id) == 0) << "kfd_events: unknown node id: " << _node_id;
    return agents.at(_node_id);
}

constexpr char READ_FAULT_CHAR    = 'R';
constexpr char WRITE_FAULT_CHAR   = 'W';
constexpr char FAULT_MIGRATE_CHAR = 'M';  // Fault resolved with a migration
constexpr char FAULT_UPDATE_CHAR  = 'U';  // Fault resolved with an update
// Queue was not restored, will be restored later
constexpr char QUEUE_RESTORE_RESCHEDULED_CHAR = 'R';

template <>
kfd_event_record
parse_event<KFD_EVENT_PAGE_MIGRATE_START>(const agent_id_map_t& agents, std::string_view str)
{
    auto  rec = kfd_event_record{};
    auto& e   = rec.data.page_migrate_event;

    common::init_public_api_struct(e);
    e.kind = ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE;

    uint32_t _kind           = 0;
    uint32_t _operation      = 0;
    uint64_t _start_address  = 0;
    uint64_t _size           = 0;
    uint32_t _from_node      = 0;
    uint32_t _to_node        = 0;
    uint32_t _prefetch_node  = 0;
    uint32_t _preferred_node = 0;

    const auto scan_count =
        std::sscanf(str.data(),
                    kfd_event_info<KFD_EVENT_PAGE_MIGRATE_START>::format_str.data(),
                    &_kind,
                    &e.timestamp,
                    &e.pid,
                    &_start_address,
                    &_size,
                    &_from_node,
                    &_to_node,
                    &_prefetch_node,
                    &_preferred_node,
                    &_operation);

    if(scan_count != 10)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "kfd: parse_event: Expected {}, scanned {}", 10, scan_count);
        return {};
    }

    e.operation           = static_cast<rocprofiler_kfd_event_page_migrate_operation_t>(_operation);
    e.start_address.value = page_to_bytes(_start_address);
    e.end_address.value   = page_to_bytes(_start_address + _size);
    e.src_agent           = get_node_agent_id(agents, _from_node);
    e.dst_agent           = get_node_agent_id(agents, _to_node);
    e.prefetch_agent      = get_node_agent_id(agents, _prefetch_node);
    e.preferred_agent     = get_node_agent_id(agents, _preferred_node);
    e.error_code          = 0;

    ROCP_INFO << fmt::format(
        "Page migrate start [ ts: {} pid: {} addr s: 0x{:X} addr "
        "e: 0x{:X} size: {}B from node: {} to node: {} prefetch node: {} preferred node: {} "
        "trigger: {} ] \n",
        e.timestamp,
        e.pid,
        e.start_address.value,
        e.end_address.value,
        (e.end_address.value - e.start_address.value),
        e.src_agent.handle,
        e.dst_agent.handle,
        e.prefetch_agent.handle,
        e.preferred_agent.handle,
        _operation);

    rec.kind      = e.kind;
    rec.operation = e.operation;
    return rec;
}

template <>
kfd_event_record
parse_event<KFD_EVENT_PAGE_MIGRATE_END>(const agent_id_map_t& agents, std::string_view str)
{
    auto  rec = kfd_event_record{};
    auto& e   = rec.data.page_migrate_event;

    common::init_public_api_struct(e);
    e.kind = ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE;

    uint32_t _kind          = 0;
    uint32_t _operation     = 0;
    uint64_t _start_address = 0;
    uint64_t _size          = 0;
    uint32_t _from_node     = 0;
    uint32_t _to_node       = 0;

    const auto scan_count =
        std::sscanf(str.data(),
                    kfd_event_info<KFD_EVENT_PAGE_MIGRATE_END>::format_str.data(),
                    &_kind,
                    &e.timestamp,
                    &e.pid,
                    &_start_address,
                    &_size,
                    &_from_node,
                    &_to_node,
                    &_operation,
                    &e.error_code);

    if(scan_count == 8)
    {
        // KFD version was not bumped when this value was added,
        // so older versions may not output an error code
        e.error_code = 0;
    }
    else if(scan_count != 9)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "kfd: parse_event: Expected {}, scanned {}", 9, scan_count);
        return {};
    }

    // e.operation = static_cast<rocprofiler_kfd_event_page_migrate_operation_t>(operation);
    e.operation           = ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_END;
    e.start_address.value = page_to_bytes(_start_address);
    e.end_address.value   = page_to_bytes(_start_address + _size);
    e.src_agent           = get_node_agent_id(agents, _from_node);
    e.dst_agent           = get_node_agent_id(agents, _to_node);

    ROCP_INFO << fmt::format("Page migrate end [ ts: {} pid: {} addr s: 0x{:X} addr e: "
                             "0x{:X} from node: {} to node: {} trigger: {} error code: {}] \n",
                             e.timestamp,
                             e.pid,
                             e.start_address.value,
                             e.end_address.value,
                             e.src_agent.handle,
                             e.dst_agent.handle,
                             _operation,
                             e.error_code);

    rec.kind      = e.kind;
    rec.operation = e.operation;
    return rec;
}

template <>
kfd_event_record
parse_event<KFD_EVENT_PAGE_FAULT_START>(const agent_id_map_t& agents, std::string_view str)
{
    auto  rec = kfd_event_record{};
    auto& e   = rec.data.page_fault_event;

    common::init_public_api_struct(e);
    e.kind = ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT;

    uint32_t _kind    = 0;
    uint32_t _node_id = 0;
    uint64_t _address = 0;

    char       _fault;
    const auto scan_count =
        std::sscanf(str.data(),
                    kfd_event_info<KFD_EVENT_PAGE_FAULT_START>::format_str.data(),
                    &_kind,
                    &e.timestamp,
                    &e.pid,
                    &_address,
                    &_node_id,
                    &_fault);

    if(scan_count != 6)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "kfd: parse_event: Expected {}, scanned {}", 6, scan_count);
        return {};
    }

    e.address.value = page_to_bytes(_address);
    e.agent_id      = get_node_agent_id(agents, _node_id);

    if(_fault == READ_FAULT_CHAR)
    {
        e.operation = ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_READ_FAULT;
    }
    else if(_fault == WRITE_FAULT_CHAR)
    {
        e.operation = ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_WRITE_FAULT;
    }
    else
    {
        ROCP_CI_LOG(WARNING) << "Unknown PAGE_FAULT_START fault type. Expected read or write fault";
    }

    ROCP_INFO << fmt::format("Page fault start [ ts: {} pid: {} addr: 0x{:X} node: {} ] \n",
                             e.timestamp,
                             e.pid,
                             e.address.value,
                             e.agent_id.handle,
                             _fault);

    rec.kind      = e.kind;
    rec.operation = e.operation;
    return rec;
}

template <>
kfd_event_record
parse_event<KFD_EVENT_PAGE_FAULT_END>(const agent_id_map_t& agents, std::string_view str)
{
    auto  rec = kfd_event_record{};
    auto& e   = rec.data.page_fault_event;

    common::init_public_api_struct(e);
    e.kind = ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT;

    uint32_t _kind    = 0;
    uint32_t _node_id = 0;
    uint64_t _address = 0;

    // How the fault was resolved: 'M'igrate / 'U'pdate
    char       _resolve_kind;
    const auto scan_count = std::sscanf(str.data(),
                                        kfd_event_info<KFD_EVENT_PAGE_FAULT_END>::format_str.data(),
                                        &_kind,
                                        &e.timestamp,
                                        &e.pid,
                                        &_address,
                                        &_node_id,
                                        &_resolve_kind);
    if(scan_count != 6)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "kfd: parse_event: Expected {}, scanned {}", 6, scan_count);
        return {};
    }

    e.address.value = page_to_bytes(_address);
    e.agent_id      = get_node_agent_id(agents, _node_id);

    if(_resolve_kind == FAULT_MIGRATE_CHAR)
    {
        e.operation = ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_MIGRATED;
    }
    else if(_resolve_kind == FAULT_UPDATE_CHAR)
    {
        e.operation = ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_UPDATED;
    }
    else
    {
        ROCP_CI_LOG(WARNING) << "Unknown PAGE_FAULT_END migrated/updated state";
    }

    ROCP_INFO << fmt::format(
        "Page fault end [ ts: {} pid: {} addr: 0x{:X} node: {} resolution: {} ] \n",
        e.timestamp,
        e.pid,
        e.address.value,
        e.agent_id.handle,
        _resolve_kind);

    rec.kind      = e.kind;
    rec.operation = e.operation;
    return rec;
}

template <>
kfd_event_record
parse_event<KFD_EVENT_QUEUE_EVICTION>(const agent_id_map_t& agents, std::string_view str)
{
    auto  rec = kfd_event_record{};
    auto& e   = rec.data.queue_event;

    common::init_public_api_struct(e);
    e.kind = ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE;

    uint32_t _kind      = 0;
    uint32_t _operation = 0;
    uint32_t _node_id   = 0;

    const auto scan_count = std::sscanf(str.data(),
                                        kfd_event_info<KFD_EVENT_QUEUE_EVICTION>::format_str.data(),
                                        &_kind,
                                        &e.timestamp,
                                        &e.pid,
                                        &_node_id,
                                        &_operation);

    if(scan_count != 5)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "kfd: parse_event: Expected {}, scanned {}", 5, scan_count);
        return {};
    }

    e.operation = static_cast<rocprofiler_kfd_event_queue_operation_t>(_operation);
    e.agent_id  = get_node_agent_id(agents, _node_id);

    ROCP_INFO << fmt::format("Queue evict [ ts: {} pid: {} node: {} trigger: {} ] \n",
                             e.timestamp,
                             e.pid,
                             e.agent_id.handle,
                             _operation);

    rec.kind      = e.kind;
    rec.operation = e.operation;
    return rec;
}

template <>
kfd_event_record
parse_event<KFD_EVENT_QUEUE_RESTORE>(const agent_id_map_t& agents, std::string_view str)
{
    auto  rec = kfd_event_record{};
    auto& e   = rec.data.queue_event;

    common::init_public_api_struct(e);
    e.kind = ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE;

    uint32_t _kind        = 0;
    uint32_t _node_id     = 0;
    char     _rescheduled = 0;

    const auto scan_count = std::sscanf(str.data(),
                                        kfd_event_info<KFD_EVENT_QUEUE_RESTORE>::format_str.data(),
                                        &_kind,
                                        &e.timestamp,
                                        &e.pid,
                                        &_node_id,
                                        &_rescheduled);

    e.agent_id = get_node_agent_id(agents, _node_id);

    if(scan_count == 5 && _rescheduled == QUEUE_RESTORE_RESCHEDULED_CHAR)
    {
        e.operation = ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED;
    }
    else if(scan_count == 5 && _rescheduled != QUEUE_RESTORE_RESCHEDULED_CHAR)
    {
        ROCP_CI_LOG(WARNING) << "kfd: parse_event: Expected rescheduled with 5 items parsed";
        return {};
    }
    else if(scan_count == 4)
    {
        e.operation = ROCPROFILER_KFD_EVENT_QUEUE_RESTORE;
    }
    else
    {
        ROCP_CI_LOG(WARNING) << fmt::format("kfd: parse_event: Expected 4 or 5, scanned {}",
                                            scan_count);
        return {};
    }

    ROCP_INFO << fmt::format("Queue restore [ ts: {} pid: {} node: {} rescheduled: {} ] \n",
                             e.timestamp,
                             e.pid,
                             e.agent_id.handle,
                             e.operation == ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED);

    rec.kind      = e.kind;
    rec.operation = e.operation;
    return rec;
}

template <>
kfd_event_record
parse_event<KFD_EVENT_UNMAP_FROM_GPU>(const agent_id_map_t& agents, std::string_view str)
{
    auto  rec = kfd_event_record{};
    auto& e   = rec.data.unmap_event;

    common::init_public_api_struct(e);
    e.kind = ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU;

    uint32_t _kind          = 0;
    uint32_t _operation     = 0;
    uint64_t _start_address = 0;
    uint64_t _size          = 0;
    uint32_t _node_id       = 0;

    const auto scan_count = std::sscanf(str.data(),
                                        kfd_event_info<KFD_EVENT_UNMAP_FROM_GPU>::format_str.data(),
                                        &_kind,
                                        &e.timestamp,
                                        &e.pid,
                                        &_start_address,
                                        &_size,
                                        &_node_id,
                                        &_operation);
    if(scan_count != 7)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "kfd: parse_event: Expected {}, scanned {}", 7, scan_count);
        return {};
    }

    e.operation = static_cast<rocprofiler_kfd_event_unmap_from_gpu_operation_t>(_operation);
    e.start_address.value = page_to_bytes(_start_address);
    e.end_address.value   = page_to_bytes(_start_address + _size);
    e.agent_id            = get_node_agent_id(agents, _node_id);

    ROCP_INFO << fmt::format("Unmap from GPU [ ts: {} pid: {} start addr: 0x{:X} end addr: 0x{:X}  "
                             "node: {} trigger {} ] \n",
                             e.timestamp,
                             e.pid,
                             e.start_address.value,
                             e.end_address.value,
                             e.agent_id.handle,
                             _operation);

    rec.kind      = e.kind;
    rec.operation = e.operation;
    return rec;
}

template <>
kfd_event_record
parse_event<KFD_EVENT_DROPPED_EVENT>(const agent_id_map_t&, std::string_view str)
{
    auto  rec = kfd_event_record{};
    auto& e   = rec.data.dropped_event;

    common::init_public_api_struct(e);
    e.kind      = ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS;
    e.operation = ROCPROFILER_KFD_EVENT_DROPPED_EVENTS;

    uint32_t _kind = 0;

    const auto scan_count = std::sscanf(str.data(),
                                        kfd_event_info<KFD_EVENT_DROPPED_EVENT>::format_str.data(),
                                        &_kind,
                                        &e.timestamp,
                                        &e.pid,
                                        &e.count);

    if(scan_count != 4)
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "kfd: parse_event: Expected {}, scanned {}", 4, scan_count);
        return {};
    }

    ROCP_TRACE << fmt::format(
        "Dropped events [ ts: {} pid: {} dropped count: {} ] \n", e.timestamp, e.pid, e.count);

    rec.kind      = e.kind;
    rec.operation = e.operation;
    return rec;
}

template <size_t OpInx, size_t... OpInxs>
kfd_event_record
parse_event(size_t                event_id,
            const agent_id_map_t& agents,
            std::string_view      strn,
            std::index_sequence<OpInx, OpInxs...>)
{
    if(OpInx == static_cast<uint32_t>(event_id))
    {
        return parse_event<OpInx>(agents, strn);
    }

    if constexpr(sizeof...(OpInxs) > 0)
        return parse_event(event_id, agents, strn, std::index_sequence<OpInxs...>{});

    return kfd_event_record{};
}

size_t
to_rocprofiler_kfd_event_id(const std::string_view event_data)
{
    size_t     kfd_id{std::numeric_limits<size_t>::max()};
    const auto scan_count = std::sscanf(event_data.data(), "%lx ", &kfd_id);

    ROCP_CI_LOG_IF(WARNING, scan_count != 1)
        << fmt::format("kfd: parse_event: Expected {}, scanned {}", 1, scan_count);

    auto event_id =
        to_rocprofiler_kfd_event_id_func(kfd_id, std::make_index_sequence<KFD_EVENT_LAST>{});

    ROCP_CI_LOG_IF(WARNING, event_id == std::numeric_limits<size_t>::max())
        << fmt::format("Failed to parse KFD event ID {}. Parsed ID: {}, kfd_event_id ID: {}\n",
                       event_data[0],
                       kfd_id,
                       event_id);

    return event_id;
}

}  // namespace

// For use in tests
kfd_event_record
parse_event(size_t event_id, const agent_id_map_t& agents, std::string_view strn)
{
    return parse_event(event_id, agents, strn, std::make_index_sequence<KFD_EVENT_LAST>{});
}

void
kfd_readlines(const std::string_view str, void(handler)(std::string_view))
{
    const auto  find_newline = [&](auto b) { return std::find(b, str.cend(), '\n'); };
    const auto* cursor       = str.cbegin();

    for(const auto* pos = find_newline(cursor); pos != str.cend(); pos = find_newline(cursor))
    {
        size_t char_count = pos - cursor;
        assert(char_count > 0);
        std::string_view event_str{cursor, char_count};

        ROCP_INFO << fmt::format("KFD event: [{}]", event_str);
        handler(event_str);

        cursor = pos + 1;
    }
}

// Event capture and reporting
namespace
{
constexpr auto kfd_ioctl_version = (1000 * KFD_IOCTL_MAJOR_VERSION) + KFD_IOCTL_MINOR_VERSION;
// Support has been added in kfdv >= 1.10+
static_assert(kfd_ioctl_version >= 1010, "KFD SMI support missing in kfd_ioctl.h");

bool
kfd_context_kinds(const context::context* ctx)
{
    return ctx->is_tracing_one_of(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE,
                                  ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT,
                                  ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE,
                                  ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                                  ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
                                  ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE,
                                  ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT,
                                  ROCPROFILER_BUFFER_TRACING_KFD_QUEUE);
}

auto
get_contexts(rocprofiler_buffer_tracing_kind_t kind, int operation)
{
    auto active_contexts = context::get_active_contexts(
        [](const auto* ctx) { return (ctx->buffered_tracer && kfd_context_kinds(ctx)); });
    auto operation_ctxs = context::context_array_t{};

    for(const auto* itr : active_contexts)
    {
        // if the given domain + op is not enabled, skip this context
        if(itr->buffered_tracer->domains(kind, operation))
        {
            operation_ctxs.emplace_back(itr);
        }
    }

    return operation_ctxs;
}

void poll_events(small_vector<pollfd>);

}  // namespace

// KFD utils
namespace kfd
{
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
        ROCP_FATAL_IF(fd == -1) << "Error opening KFD handle @ " << KFD_DEVICE_PATH;
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
            ROCP_INFO << fmt::format("KFD v{}.{}", args.major_version, args.minor_version);
        else
            ROCP_ERROR << fmt::format("Could not determine KFD version");
        return args;
    }();

    return version;
}

struct poll_kfd_t
{
    static constexpr auto DEFAULT_FLAGS{EFD_CLOEXEC};

    struct gpu_fd_t
    {
        unsigned int               node_id = 0;
        fd_t                       fd      = {};
        const rocprofiler_agent_t* agent   = nullptr;
    };

    kfd_device_fd kfd_fd        = {};
    pollfd        thread_notify = {};
    std::thread   bg_thread     = {};
    bool          active        = {false};

    poll_kfd_t() = default;

    poll_kfd_t(const small_vector<size_t>& rprof_ev)
    : kfd_fd{kfd_device_fd{}}
    {
        small_vector<pollfd> file_handles = {};

        const auto kfd_flags = kfd_bitmask(rprof_ev, std::make_index_sequence<KFD_EVENT_LAST>{});

        ROCP_INFO << fmt::format("Setting KFD flags to [0b{:b}] \n", kfd_flags);

        // Create fd for notifying thread when we want to wake it up, and an eventfd for any events
        // to this thread
        file_handles.emplace_back(
            pollfd{.fd = eventfd(0, DEFAULT_FLAGS), .events = 0, .revents = 0});

        fd_t thread_pipes[2]{};

        [&]() {
            const auto retcode = pipe2(&thread_pipes[0], DEFAULT_FLAGS);
            const auto _err    = errno;
            ROCP_FATAL_IF(retcode != 0)
                << fmt::format("Pipe creation for page-migration thread notify returned {} :: {}\n",
                               retcode,
                               strerror(_err));
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
                ROCP_INFO << fmt::format(
                    "GPU node {} with fd {} added\n", agent->gpu_id, gpu_event_fd);
            }
        }

        // Enable KFD masked events by writing flags to kfd fd
        for(size_t i = 2; i < file_handles.size(); ++i)
        {
            auto& fd         = file_handles[i];
            auto  write_size = write(fd.fd, &kfd_flags, sizeof(kfd_flags));
            ROCP_INFO << fmt::format(
                "Writing {} to GPU fd {} ({} bytes)\n", kfd_flags, fd.fd, write_size);
            CHECK(write_size == sizeof(kfd_flags));
        }

        // start bg thread
        internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
        bg_thread = std::thread{poll_events, file_handles};
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

        active = true;
    }

    poll_kfd_t(const poll_kfd_t&) = delete;
    poll_kfd_t& operator=(const poll_kfd_t&) = delete;

    poll_kfd_t(poll_kfd_t&&) noexcept = default;
    poll_kfd_t& operator=(poll_kfd_t&&) noexcept = default;

    ~poll_kfd_t()
    {
        ROCP_INFO << fmt::format("Terminating poll_kfd\n");
        if(!active) return;

        // wake thread up
        auto bytes_written{-1};
        do
        {
            bytes_written = write(thread_notify.fd, "E", 1);
        } while(bytes_written == -1 && (errno == EINTR || errno == EAGAIN));

        ROCP_INFO << fmt::format("Background thread signalled\n");

        bg_thread.join();

        close(thread_notify.fd);
    }

    node_fd_t get_node_fd(int gpu_node_id) const
    {
        kfd_ioctl_smi_events_args args{};
        args.gpuid = gpu_node_id;

        if(auto ret = ioctl(kfd_fd.fd, args); ret == -1)
            ROCP_ERROR << fmt::format(
                "Could not get GPU node {} file descriptor (exit code: {})", gpu_node_id, ret);
        return args.anon_fd;
    }
};
}  // namespace
}  // namespace kfd

// for all contexts
struct config
{
private:
    kfd::poll_kfd_t kfd_handle{};

    static inline config* _config{nullptr};

    config(const small_vector<size_t>& _event_ids)
    : kfd_handle{_event_ids}
    {}

public:
    static void init(const small_vector<size_t>& event_ids) { _config = new config{event_ids}; }

    static void reset()
    {
        config* ptr = nullptr;
        std::swap(ptr, _config);
        delete ptr;
    }

    static void reset_on_fork() { _config = nullptr; }
};

namespace
{
rocprofiler_kfd_page_migrate_operation_t
get_page_migrate_record_op(const page_migrate_event_record_t& start,
                           const page_migrate_event_record_t& end)
{
    ROCP_ERROR_IF(end.operation != ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_END)
        << fmt::format("Expected end to be operation {}, got vs {}",
                       static_cast<int>(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_END),
                       static_cast<int>(end.operation));

    if(start.operation == ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PREFETCH)
    {
        return ROCPROFILER_KFD_PAGE_MIGRATE_PREFETCH;
    }
    else if(start.operation == ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_GPU)
    {
        return ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_GPU;
    }
    else if(start.operation == ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_CPU)
    {
        return ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_CPU;
    }
    else if(start.operation == ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_TTM_EVICTION)
    {
        return ROCPROFILER_KFD_PAGE_MIGRATE_TTM_EVICTION;
    }
    else
    {
        ROCP_ERROR << fmt::format("Invalid operation for pairing page_migrate (start {}, end {})",
                                  static_cast<int>(start.operation),
                                  static_cast<int>(end.operation));
        return ROCPROFILER_KFD_PAGE_MIGRATE_NONE;
    }
}

rocprofiler_kfd_page_fault_operation_t
get_page_fault_record_op(const page_fault_event_record_t& start,
                         const page_fault_event_record_t& end)
{
    if(start.operation == ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_READ_FAULT &&
       end.operation == ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_MIGRATED)
    {
        return ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_MIGRATED;
    }
    else if(start.operation == ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_READ_FAULT &&
            end.operation == ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_UPDATED)
    {
        return ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_UPDATED;
    }
    else if(start.operation == ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_WRITE_FAULT &&
            end.operation == ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_MIGRATED)
    {
        return ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_MIGRATED;
    }
    else if(start.operation == ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_WRITE_FAULT &&
            end.operation == ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_UPDATED)
    {
        return ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_UPDATED;
    }
    else
    {
        ROCP_ERROR << fmt::format("Invalid operation for pairing page_fault (start {}, end {})",
                                  static_cast<int>(start.operation),
                                  static_cast<int>(end.operation));
        return ROCPROFILER_KFD_PAGE_FAULT_NONE;
    }
}

rocprofiler_kfd_queue_operation_t
get_queue_record_op(const queue_event_record_t& start, const queue_event_record_t& end)
{
    ROCP_ERROR_IF(end.operation != ROCPROFILER_KFD_EVENT_QUEUE_RESTORE &&
                  end.operation != ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED)
        << "Expected end operation for queue end event";

    if(start.operation == ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SVM)
        return ROCPROFILER_KFD_QUEUE_EVICT_SVM;
    else if(start.operation == ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR)
        return ROCPROFILER_KFD_QUEUE_EVICT_USERPTR;
    else if(start.operation == ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM)
        return ROCPROFILER_KFD_QUEUE_EVICT_TTM;
    else if(start.operation == ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND)
        return ROCPROFILER_KFD_QUEUE_EVICT_SUSPEND;
    else if(start.operation == ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT)
        return ROCPROFILER_KFD_QUEUE_EVICT_CRIU_CHECKPOINT;
    else if(start.operation == ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE)
        return ROCPROFILER_KFD_QUEUE_EVICT_CRIU_RESTORE;
    else
    {
        ROCP_ERROR << fmt::format("Invalid operation for pairing queue_suspend (start {}, end {})",
                                  static_cast<int>(start.operation),
                                  static_cast<int>(end.operation));
        return ROCPROFILER_KFD_QUEUE_NONE;
    }
}

template <typename T>
struct kfd_event_hash_t;

template <typename T>
struct kfd_event_compare_t;

template <typename T>
uint64_t
bitshift(T&& val, uint32_t lshift)
{
    return static_cast<uint64_t>(val) << lshift;
}

template <>
struct kfd_event_hash_t<page_migrate_event_record_t>
{
    std::size_t operator()(const page_migrate_event_record_t& data) const noexcept
    {
        return data.start_address.handle ^ bitshift(data.src_agent.handle, 32) ^
               bitshift(data.dst_agent.handle, 32);
    }
};

template <>
struct kfd_event_hash_t<page_fault_event_record_t>
{
    std::size_t operator()(const page_fault_event_record_t& data) const noexcept
    {
        return data.address.handle ^ bitshift(data.agent_id.handle, 32);
    }
};

template <>
struct kfd_event_hash_t<queue_event_record_t>
{
    std::size_t operator()(const queue_event_record_t& data) const noexcept
    {
        return bitshift(data.pid, 32) ^ data.agent_id.handle;
    }
};

template <>
struct kfd_event_compare_t<page_migrate_event_record_t>
{
    bool operator()(const page_migrate_event_record_t& lhs,
                    const page_migrate_event_record_t& rhs) const noexcept
    {
        return std::tie(lhs.start_address.handle,
                        lhs.end_address.handle,
                        lhs.src_agent.handle,
                        lhs.dst_agent.handle) == std::tie(rhs.start_address.handle,
                                                          rhs.end_address.handle,
                                                          rhs.src_agent.handle,
                                                          rhs.dst_agent.handle);
    }
};

template <>
struct kfd_event_compare_t<page_fault_event_record_t>
{
    bool operator()(const page_fault_event_record_t& lhs,
                    const page_fault_event_record_t& rhs) const noexcept
    {
        return std::tie(lhs.address.handle, lhs.agent_id.handle) ==
               std::tie(rhs.address.handle, rhs.agent_id.handle);
    }
};

template <>
struct kfd_event_compare_t<queue_event_record_t>
{
    bool operator()(const queue_event_record_t& lhs, const queue_event_record_t& rhs) const noexcept
    {
        return std::tie(lhs.pid, lhs.agent_id.handle) == std::tie(rhs.pid, rhs.agent_id.handle);
    }
};

template <typename T>
using events_unordered_set = std::unordered_set<T, kfd_event_hash_t<T>, kfd_event_compare_t<T>>;

template <size_t... Ops>
bool
is_one_of(int op, std::index_sequence<Ops...>)
{
    return ((op == Ops) || ...);
}

void
check_paired_events(buffer::instance* buffer, const kfd_event_record& rec)
{
    thread_local static events_unordered_set<page_migrate_event_record_t> page_migrate_events{};
    thread_local static events_unordered_set<page_fault_event_record_t>   page_fault_events{};
    thread_local static events_unordered_set<queue_event_record_t>        queue_events{};

    if(rec.kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE)
    {
        const auto& end = rec.data.page_migrate_event;

        bool is_start_event = is_one_of(rec.operation, page_migrate_start_ops_t{});
        bool is_end_event   = is_one_of(rec.operation, page_migrate_end_ops_t{});

        if(is_start_event)
        {
            // start event, insert
            page_migrate_events.insert(rec.data.page_migrate_event);
            return;
        }
        else if(is_end_event)
        {
            // end event: pair and emplace into buffer
            auto ret = common::init_public_api_struct(page_migrate_record_t{});
            if(auto found = page_migrate_events.find(end); found != page_migrate_events.end())
            {
                const auto& start   = *found;
                ret.kind            = ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE;
                ret.operation       = get_page_migrate_record_op(start, end);
                ret.start_timestamp = start.timestamp;
                ret.end_timestamp   = end.timestamp;
                ASSERT_SAME_AND_COPY(pid);
                ASSERT_SAME_AND_COPY(start_address.handle);
                ASSERT_SAME_AND_COPY(end_address.handle);
                ASSERT_SAME_AND_COPY(src_agent.handle);
                ASSERT_SAME_AND_COPY(dst_agent.handle);
                ret.prefetch_agent  = start.prefetch_agent;
                ret.preferred_agent = start.preferred_agent;
                ASSERT_SAME_AND_COPY(error_code);
                // Create a paired record and insert into buffer
                CHECK_NOTNULL(buffer)->emplace(ROCPROFILER_BUFFER_CATEGORY_TRACING, ret.kind, ret);
                // Remove the item from the buffer
                page_migrate_events.erase(found);
            }
        }
        else
        {
            // This is not a valid operation
            ROCP_ERROR << fmt::format(
                "kfd_events: Invalid operation {} for paring page_migrate events", rec.operation);
        }
    }
    else if(rec.kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT)
    {
        const auto& end = rec.data.page_fault_event;

        bool is_start_event = is_one_of(rec.operation, page_fault_start_ops_t{});
        bool is_end_event   = is_one_of(rec.operation, page_fault_end_ops_t{});

        if(is_start_event)
        {
            // start event, insert
            page_fault_events.insert(rec.data.page_fault_event);
            return;
        }
        else if(is_end_event)
        {
            // end event: pair and emplace into buffer
            auto ret = common::init_public_api_struct(page_fault_record_t{});
            if(auto found = page_fault_events.find(end); found != page_fault_events.end())
            {
                const auto& start   = *found;
                ret.kind            = ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT;
                ret.operation       = get_page_fault_record_op(start, end);
                ret.start_timestamp = start.timestamp;
                ret.end_timestamp   = end.timestamp;
                ASSERT_SAME_AND_COPY(pid);
                ASSERT_SAME_AND_COPY(agent_id.handle);
                ASSERT_SAME_AND_COPY(address.handle);
                // Create a paired record and insert into buffer
                CHECK_NOTNULL(buffer)->emplace(ROCPROFILER_BUFFER_CATEGORY_TRACING, ret.kind, ret);
                // Remove the item from the buffer
                page_fault_events.erase(found);
            }
        }
        else
        {
            // This is not a valid operation
            ROCP_ERROR << fmt::format(
                "kfd_events: Invalid operation {} for paring page_fault events", rec.operation);
        }
    }
    else if(rec.kind == ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE)
    {
        const auto& end = rec.data.queue_event;

        bool is_start_event = is_one_of(rec.operation, queue_evict_ops_t{});
        bool is_end_event   = is_one_of(rec.operation, queue_restore_ops_t{});

        if(is_start_event)
        {
            // start event, insert
            queue_events.insert(rec.data.queue_event);
            return;
        }
        else if(is_end_event)
        {
            // end event: pair and emplace into buffer
            auto ret = common::init_public_api_struct(queue_record_t{});
            if(auto found = queue_events.find(end); found != queue_events.end())
            {
                const auto& start   = *found;
                ret.kind            = ROCPROFILER_BUFFER_TRACING_KFD_QUEUE;
                ret.operation       = get_queue_record_op(start, end);
                ret.start_timestamp = start.timestamp;
                ret.end_timestamp   = end.timestamp;
                ASSERT_SAME_AND_COPY(pid);
                ASSERT_SAME_AND_COPY(agent_id.handle);
                // Create a paired record and insert into buffer
                CHECK_NOTNULL(buffer)->emplace(ROCPROFILER_BUFFER_CATEGORY_TRACING, ret.kind, ret);
                // Remove the item from the buffer
                queue_events.erase(found);
            }
        }
        else if(rec.operation == ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED)
        {
            // If event is ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED we should not attempt to
            // pair it. It is an instantaneous event.
            // It is handled in handle_reporting -> emplace_buffer_record.
        }
        else
        {
            // Else, it is an error.
            ROCP_ERROR << fmt::format("kfd_events: Invalid operation {} for paring events",
                                      rec.operation);
        }
    }
}

void
emplace_buffer_record(buffer::instance* buffer, const kfd_event_record& rec)
{
    switch(rec.kind)
    {
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE:
        {
            CHECK_NOTNULL(buffer)->emplace(
                ROCPROFILER_BUFFER_CATEGORY_TRACING, rec.kind, rec.data.page_migrate_event);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT:
        {
            CHECK_NOTNULL(buffer)->emplace(
                ROCPROFILER_BUFFER_CATEGORY_TRACING, rec.kind, rec.data.page_fault_event);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE:
        {
            CHECK_NOTNULL(buffer)->emplace(
                ROCPROFILER_BUFFER_CATEGORY_TRACING, rec.kind, rec.data.queue_event);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU:
        {
            CHECK_NOTNULL(buffer)->emplace(
                ROCPROFILER_BUFFER_CATEGORY_TRACING, rec.kind, rec.data.unmap_event);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS:
        {
            CHECK_NOTNULL(buffer)->emplace(
                ROCPROFILER_BUFFER_CATEGORY_TRACING, rec.kind, rec.data.dropped_event);
            break;
        }
        default:
        {
            ROCP_ERROR << fmt::format("Invalid Kind {} for record", static_cast<int>(rec.kind));
        }
    }
}

void
handle_reporting(std::string_view event_data)
{
    // We can check the operation only after parsing the event
    const auto kfd_event = to_rocprofiler_kfd_event_id(event_data);
    auto       event     = parse_event(
        kfd_event, get_node_map(), event_data, std::make_index_sequence<KFD_EVENT_LAST>{});

    ROCP_ERROR_IF(event.kind < ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE ||
                  event.kind > ROCPROFILER_BUFFER_TRACING_KFD_QUEUE)
        << fmt::format("kfd_events: Invalid record kind {}", static_cast<int>(event.kind));

    ROCP_ERROR_IF(event.operation == -1)
        << fmt::format("kfd_events: Invalid record operation: ({}, {})",
                       static_cast<int>(event.kind),
                       event.operation);

    auto buffered_contexts = get_contexts(event.kind, event.operation);
    if(buffered_contexts.empty()) return;

    for(const auto& itr : buffered_contexts)
    {
        auto* buffer = buffer::get_buffer(itr->buffered_tracer->buffer_data.at(event.kind));

        check_paired_events(buffer, event);
        emplace_buffer_record(buffer, event);
    }
}

void
poll_events(small_vector<pollfd> file_handles)
{
    // storage to write records to, 1MB
    constexpr size_t PREALLOCATE_ELEMENT_COUNT{1024 * 128};
    std::string      scratch_buffer(PREALLOCATE_ELEMENT_COUNT, '\0');
    auto&            exitfd = file_handles[1];

    // Wait or spin on events.
    //  0 -> return immediately even if no events
    // -1 -> wait indefinitely

    pthread_setname_np(pthread_self(), "bg:poll-kfd");

    for(auto& fd : file_handles)
    {
        ROCP_INFO << fmt::format(
            "Handle = {}, events = {}, revents = {}\n", fd.fd, fd.events, fd.revents);
    }

    while(true)
    {
        auto poll_ret = poll(file_handles.data(), file_handles.size(), -1);

        if(poll_ret == -1)
        {
            ROCP_CI_LOG(WARNING)
                << "Background thread file descriptors for page-migration are invalid";
            return;
        }

        if((exitfd.revents & POLLIN) != 0)
        {
            for(const auto& f : file_handles)
            {
                close(f.fd);
            }
            ROCP_INFO << "Terminating background thread\n";
            return;
        }

        using namespace std::chrono_literals;

        // 0 and 1 are for generic and pipe-notify handles
        for(size_t i = 2; i < file_handles.size(); ++i)
        {
            auto& fd = file_handles[i];

            // We have data to read, perhaps multiple events
            if((fd.revents & POLLIN) != 0)
            {
                size_t status_size   = read(fd.fd, scratch_buffer.data(), scratch_buffer.size());
                auto   event_strings = std::string_view{scratch_buffer.data(), status_size};
                kfd_readlines(event_strings, handle_reporting);
            }
            fd.revents = 0;
        }
    }
}

template <size_t Kind, size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return kfd_operation_info<Kind, Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id<Kind>(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t Kind, size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(kfd_kind_info<Kind>::last)) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, kfd_operation_info<Kind, Idx>::operation), ...);
}

bool
context_filter(const context::context* ctx)
{
    return (ctx->buffered_tracer && kfd_context_kinds(ctx));
}

template <size_t... Inxs>
rocprofiler_status_t init(std::index_sequence<Inxs...>)
{
    static const small_vector<size_t> event_ids{Inxs...};
    // Check if version is more than 1.11
    auto ver = kfd::get_version();
    if(ver.major_version * 1000 + ver.minor_version > 1011)
    {
        if(!context::get_registered_contexts(context_filter).empty())
        {
            config::init(event_ids);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    }
    else
    {
        // Add a buffer record with this info
        ROCP_ERROR << fmt::format(
            "KFD does not support SVM event reporting in v{}.{} (requires v1.11)",
            ver.major_version,
            ver.minor_version);
        return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
    }
}

using kfd_buffer_tracing_ids_t =
    std::index_sequence<ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE,
                        ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT,
                        ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE,
                        ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                        ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
                        ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE,
                        ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT,
                        ROCPROFILER_BUFFER_TRACING_KFD_QUEUE>;

template <size_t Kind, size_t... Kinds>
const char*
name_by_id(uint32_t kind, uint32_t op, std::index_sequence<Kind, Kinds...>)
{
    if(kind == Kind)
    {
        return name_by_id<Kind>(op, std::make_index_sequence<kfd_kind_info<Kind>::last>{});
    }
    else if constexpr(sizeof...(Kinds) > 0)
        return name_by_id(kind, op, std::index_sequence<Kinds...>{});

    ROCP_CI_LOG(WARNING) << fmt::format("KFD events name_by_id: Unknown Kind {} {}", kind, op);
    return "KFD events: Unknown Kind";
}

template <size_t Kind, size_t... Kinds>
std::vector<uint32_t>
get_ids(int kind, std::index_sequence<Kind, Kinds...>)
{
    if(kind == Kind)
    {
        auto _data = std::vector<uint32_t>{};
        _data.reserve(kfd_kind_info<Kind>::last);
        get_ids<Kind>(_data, std::make_index_sequence<kfd_kind_info<Kind>::last>{});
        return _data;
    }
    else if constexpr(sizeof...(Kinds) > 0)
        return get_ids(kind, std::index_sequence<Kinds...>{});

    ROCP_CI_LOG(WARNING) << fmt::format("KFD events get_ids: Unknown Kind {}", kind);
}

}  // namespace

}  // namespace kfd
}  // namespace rocprofiler

namespace rocprofiler
{
namespace kfd
{
rocprofiler_status_t
init()
{
    // Prevent re-init for different buffer op kinds
    static bool                 init_done{false};
    static rocprofiler_status_t retcode{ROCPROFILER_STATUS_ERROR};

    if(!init_done)
    {
        pthread_atfork(nullptr, nullptr, []() {
            // null out child's copy on fork and reinitialize
            // otherwise all children wait on the same thread to join
            config::reset_on_fork();
            init(std::make_index_sequence<KFD_EVENT_LAST>{});
        });

        retcode   = init(std::make_index_sequence<KFD_EVENT_LAST>{});
        init_done = true;
    }

    return retcode;
}

void
finalize()
{
    config::reset();
}

const char*
name_by_id(uint32_t kind, uint32_t id)
{
    return name_by_id(kind, id, kfd_buffer_tracing_ids_t{});
}

std::vector<uint32_t>
get_ids(uint32_t kind)
{
    return get_ids(kind, kfd_buffer_tracing_ids_t{});
}
}  // namespace kfd
}  // namespace rocprofiler
