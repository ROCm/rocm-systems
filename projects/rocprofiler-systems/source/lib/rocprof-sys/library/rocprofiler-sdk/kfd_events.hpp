// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/fwd.hpp"

#include <cstdint>

#if ROCPROFILER_VERSION >= 10000

#    include "core/categories.hpp"
#    include "core/trace_cache/buffer_storage.hpp"
#    include "core/trace_cache/cache_manager.hpp"
#    include "core/trace_cache/metadata_registry.hpp"
#    include "logger/debug.hpp"

#    include <spdlog/fmt/fmt.h>
#    include <unistd.h>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

// ─── Public template functions ────────────────────────────────────────────────

template <typename Wrapper>
void
kfd_event_metadata_initialize(const client_data<Wrapper>* tool_data);

template <typename Wrapper>
void
tool_kfd_page_fault_callback(const client_data<Wrapper>*           tool_data,
                             const backend::kfd_page_fault_record* record);

template <typename Wrapper>
void
tool_kfd_page_migrate_callback(const client_data<Wrapper>*             tool_data,
                               const backend::kfd_page_migrate_record* record);

template <typename Wrapper>
void
tool_kfd_queue_callback(const client_data<Wrapper>*      tool_data,
                        const backend::kfd_queue_record* record);

template <typename Wrapper>
void
tool_kfd_event_queue_callback(const client_data<Wrapper>*            tool_data,
                              const backend::kfd_event_queue_record* record);

template <typename Wrapper>
void
tool_kfd_event_unmap_from_gpu_callback(const client_data<Wrapper>*            tool_data,
                                       const backend::kfd_event_unmap_record* record);

template <typename Wrapper>
void
tool_kfd_event_dropped_events_callback(const client_data<Wrapper>*              tool_data,
                                       const backend::kfd_event_dropped_record* record);

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

// ─── Template implementations ────────────────────────────────────────────────

namespace rocprofsys::rocprofiler_sdk
{

// ─── kfd_events_detail: helpers that specialize on concrete record types ──────

namespace kfd_events_detail
{

template <typename RecordT>
const char*
get_kfd_operation_name([[maybe_unused]] const RecordT* record)
{
    return "KFD Event";
}

template <>
inline const char*
get_kfd_operation_name(const backend::kfd_page_fault_record* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_MIGRATED:
            return "PAGE_FAULT_READ_FAULT_MIGRATED";
        case ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_UPDATED:
            return "PAGE_FAULT_READ_FAULT_UPDATED";
        case ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_MIGRATED:
            return "PAGE_FAULT_WRITE_FAULT_MIGRATED";
        case ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_UPDATED:
            return "PAGE_FAULT_WRITE_FAULT_UPDATED";
        default: return "PAGE_FAULT";
    }
}

template <>
inline const char*
get_kfd_operation_name(const backend::kfd_page_migrate_record* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_PAGE_MIGRATE_PREFETCH: return "PAGE_MIGRATE_PREFETCH";
        case ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_GPU:
            return "PAGE_MIGRATE_PAGEFAULT_GPU";
        case ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_CPU:
            return "PAGE_MIGRATE_PAGEFAULT_CPU";
        case ROCPROFILER_KFD_PAGE_MIGRATE_TTM_EVICTION:
            return "PAGE_MIGRATE_TTM_EVICTION";
        default: return "PAGE_MIGRATE";
    }
}

template <>
inline const char*
get_kfd_operation_name(const backend::kfd_queue_record* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_QUEUE_EVICT_SVM: return "QUEUE_EVICT_SVM";
        case ROCPROFILER_KFD_QUEUE_EVICT_USERPTR: return "QUEUE_EVICT_USERPTR";
        case ROCPROFILER_KFD_QUEUE_EVICT_TTM: return "QUEUE_EVICT_TTM";
        case ROCPROFILER_KFD_QUEUE_EVICT_SUSPEND: return "QUEUE_EVICT_SUSPEND";
        case ROCPROFILER_KFD_QUEUE_EVICT_CRIU_CHECKPOINT:
            return "QUEUE_EVICT_CRIU_CHECKPOINT";
        case ROCPROFILER_KFD_QUEUE_EVICT_CRIU_RESTORE: return "QUEUE_EVICT_CRIU_RESTORE";
        default: return "QUEUE_EVICT";
    }
}

template <>
inline const char*
get_kfd_operation_name(const backend::kfd_event_queue_record* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SVM: return "QUEUE_EVICT_SVM_EVENT";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR:
            return "QUEUE_EVICT_USERPTR_EVENT";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM: return "QUEUE_EVICT_TTM_EVENT";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND:
            return "QUEUE_EVICT_SUSPEND_EVENT";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT:
            return "QUEUE_EVICT_CRIU_CHECKPOINT_EVENT";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE:
            return "QUEUE_EVICT_CRIU_RESTORE_EVENT";
        case ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED:
            return "QUEUE_RESTORE_RESCHEDULED_EVENT";
        case ROCPROFILER_KFD_EVENT_QUEUE_RESTORE: return "QUEUE_RESTORE_EVENT";
        default: return "QUEUE_EVICT_EVENT";
    }
}

template <>
inline const char*
get_kfd_operation_name(const backend::kfd_event_unmap_record* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY:
            return "UNMAP_FROM_GPU_MMU_NOTIFY";
        case ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE:
            return "UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE";
        case ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_UNMAP_FROM_CPU:
            return "UNMAP_FROM_GPU_UNMAP_FROM_CPU";
        default: return "UNMAP_FROM_GPU";
    }
}

template <>
inline const char*
get_kfd_operation_name(const backend::kfd_event_dropped_record* /*record*/)
{
    return "DROPPED_EVENTS";
}

// ─── get_kfd_pmc_value ────────────────────────────────────────────────────────

template <typename RecordT>
std::uint64_t
get_kfd_pmc_value([[maybe_unused]] const RecordT* record)
{
    return 1;
}

template <>
inline std::uint64_t
get_kfd_pmc_value(const backend::kfd_page_fault_record* record)
{
    return record->address.value;
}

template <>
inline std::uint64_t
get_kfd_pmc_value(const backend::kfd_page_migrate_record* record)
{
    if(record->end_address.value < record->start_address.value)
    {
        LOG_WARNING("KFD page migrate: end_address ({:#x}) < start_address ({:#x})",
                    record->end_address.value, record->start_address.value);
        return 0;
    }
    return record->end_address.value - record->start_address.value;
}

template <>
inline std::uint64_t
get_kfd_pmc_value(const backend::kfd_queue_record* /*record*/)
{
    return 1;
}

template <>
inline std::uint64_t
get_kfd_pmc_value(const backend::kfd_event_dropped_record* record)
{
    return record->count;
}

template <>
inline std::uint64_t
get_kfd_pmc_value(const backend::kfd_event_unmap_record* record)
{
    if(record->end_address.value < record->start_address.value)
    {
        LOG_WARNING("KFD unmap_from_gpu: end_address ({:#x}) < start_address ({:#x})",
                    record->end_address.value, record->start_address.value);
        return 0;
    }
    return record->end_address.value - record->start_address.value;
}

// ─── Metadata / track helpers (no SDK types beyond what's in fwd.hpp) ─────────

template <typename CategoryT>
void
cache_category()
{
    trace_cache::get_metadata_registry().add_string(trait::name<CategoryT>::value);
}

inline void
cache_add_thread_info(std::uint64_t tid)
{
    trace_cache::get_metadata_registry().add_thread_info(
        { getppid(), getpid(), tid, 0, 0, "{}" });
}

inline void
cache_add_track(const char* track_name, std::uint64_t tid)
{
    trace_cache::get_metadata_registry().add_track({ track_name, tid, "{}" });
}

template <typename Wrapper>
inline const tool_agent*
get_tool_agent(const client_data<Wrapper>* tool_data, typename Wrapper::agent_id agent_id)
{
    const auto* ag = tool_data->get_gpu_tool_agent(agent_id);
    if(ag) return ag;
    for(const auto& itr : tool_data->cpu_agents)
        if(agent_id.handle == itr.agent->handle) return &itr;
    return nullptr;
}

inline std::string
agent_label(const tool_agent* _agent)
{
    if(!_agent || !_agent->agent) return "?";
    auto type = _agent->agent->type;
    auto idx  = _agent->device_id;
    return fmt::format("{} {}", type == agent_type::GPU ? "GPU" : "CPU", idx);
}

inline std::string
agent_node_id_str(const tool_agent* _agent)
{
    if(_agent && _agent->agent) return std::to_string(_agent->agent->node_id);
    return "null";
}

}  // namespace kfd_events_detail

// ─── kfd_event_metadata_initialize<Wrapper> ───────────────────────────────────

template <typename Wrapper>
void
kfd_event_metadata_initialize(const client_data<Wrapper>* tool_data)
{
    using namespace kfd_events_detail;

    cache_category<category::rocm_kfd_page_fault>();
    cache_category<category::rocm_kfd_page_migrate>();
    cache_category<category::rocm_kfd_queue>();
    cache_category<category::rocm_kfd_event_queue>();
    cache_category<category::rocm_kfd_event_unmap_from_gpu>();
    cache_category<category::rocm_kfd_event_dropped_events>();

    constexpr size_t EVENT_CODE  = 0;
    constexpr size_t INSTANCE_ID = 0;
    constexpr auto*  COMPONENT   = "rocm";
    constexpr auto*  BLOCK       = "KFD";
    constexpr auto*  EXPRESSION  = "";

    if(!tool_data->gpu_agents.empty())
    {
        const auto dropped_dev_id =
            static_cast<std::uint32_t>(tool_data->gpu_agents.front().device_id);
        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, dropped_dev_id, "GPU", EVENT_CODE, INSTANCE_ID,
              trait::name<category::rocm_kfd_event_dropped_events>::value,
              "KFD Dropped Events",
              trait::name<category::rocm_kfd_event_dropped_events>::description,
              "KFD dropped_events events", COMPONENT, "count", trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });
    }

    for(const auto& gpu : tool_data->gpu_agents)
    {
        auto dev_idx = static_cast<std::uint32_t>(gpu.device_id);

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, dev_idx, "GPU", EVENT_CODE, INSTANCE_ID,
              trait::name<category::rocm_kfd_page_fault>::value, "KFD Page Fault Events",
              trait::name<category::rocm_kfd_page_fault>::description,
              "KFD page fault paired records", COMPONENT, "events", trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, dev_idx, "GPU", EVENT_CODE, INSTANCE_ID,
              trait::name<category::rocm_kfd_page_migrate>::value,
              "KFD Page Migration Events",
              trait::name<category::rocm_kfd_page_migrate>::description,
              "KFD page migration paired records", COMPONENT, "events",
              trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, dev_idx, "GPU", EVENT_CODE, INSTANCE_ID,
              trait::name<category::rocm_kfd_queue>::value, "KFD Queue Events",
              trait::name<category::rocm_kfd_queue>::description,
              "KFD queue eviction/restore paired records", COMPONENT, "events",
              trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, dev_idx, "GPU", EVENT_CODE, INSTANCE_ID,
              trait::name<category::rocm_kfd_event_queue>::value,
              "KFD Event Queue Operations",
              trait::name<category::rocm_kfd_event_queue>::description,
              "KFD queue eviction/restore events", COMPONENT, "events",
              trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0, "{}" });

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, dev_idx, "GPU", EVENT_CODE, INSTANCE_ID,
              trait::name<category::rocm_kfd_event_unmap_from_gpu>::value,
              "KFD Unmap from GPU Events",
              trait::name<category::rocm_kfd_event_unmap_from_gpu>::description,
              "KFD unmap from GPU events", COMPONENT, "events", trace_cache::ABSOLUTE,
              BLOCK, EXPRESSION, 0, 0, "{}" });
    }

    for(const auto& cpu : tool_data->cpu_agents)
    {
        auto dev_idx = static_cast<std::uint32_t>(cpu.device_id);
        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::CPU, dev_idx, "CPU", EVENT_CODE, INSTANCE_ID,
              trait::name<category::rocm_kfd_page_migrate>::value,
              "KFD Page Migration Events",
              trait::name<category::rocm_kfd_page_migrate>::description,
              "KFD page migration paired records", COMPONENT, "events",
              trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0, "{}" });
    }
}

// ─── tool_kfd_page_fault_callback<Wrapper> ────────────────────────────────────

template <typename Wrapper>
void
tool_kfd_page_fault_callback(const client_data<Wrapper>*           tool_data,
                             const backend::kfd_page_fault_record* record)
{
    using namespace kfd_events_detail;
    if(!record) return;

    auto        _beg_ns   = record->start_timestamp;
    auto        _end_ns   = record->end_timestamp;
    const auto* _name     = get_kfd_operation_name(record);
    auto        _pid      = record->pid;
    auto        _agent_id = record->agent_id;
    auto        _address  = record->address.value;

    const auto* _agent = get_tool_agent(tool_data, _agent_id);

    auto tid = static_cast<std::uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = fmt::format("KFD Page Fault [{}]", agent_label(_agent));
    cache_add_track(track_name.c_str(), tid);

    auto _agent_nid = agent_node_id_str(_agent);
    auto args_str   = fmt::format("0;;std::uint64_t;;address;;{:#x};;"
                                    "1;;string;;agent;;{};;",
                                  _address, _agent_nid);

    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));
    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid, _name, _beg_ns, _end_ns, std::move(args_str),
        trait::name<category::rocm_kfd_page_fault>::value, std::move(track_name), "{}",
        static_cast<std::uint32_t>(_agent ? _agent->device_id : 0),
        static_cast<std::uint8_t>(agent_type::GPU),
        trait::name<category::rocm_kfd_page_fault>::value, pmc_value,
        std::optional<std::int64_t>(_pid) });
}

// ─── tool_kfd_page_migrate_callback<Wrapper> ─────────────────────────────────

template <typename Wrapper>
void
tool_kfd_page_migrate_callback(const client_data<Wrapper>*             tool_data,
                               const backend::kfd_page_migrate_record* record)
{
    using namespace kfd_events_detail;
    if(!record) return;

    auto        _beg_ns          = record->start_timestamp;
    auto        _end_ns          = record->end_timestamp;
    const auto* _name            = get_kfd_operation_name(record);
    auto        _pid             = record->pid;
    auto        _start_addr      = record->start_address.value;
    auto        _end_addr        = record->end_address.value;
    auto        _src_agent       = record->src_agent;
    auto        _dst_agent       = record->dst_agent;
    auto        _prefetch_agent  = record->prefetch_agent;
    auto        _preferred_agent = record->preferred_agent;
    auto        _error_code      = record->error_code;

    const auto* _src_tool_agent       = get_tool_agent(tool_data, _src_agent);
    const auto* _dst_tool_agent       = get_tool_agent(tool_data, _dst_agent);
    const auto* _prefetch_tool_agent  = get_tool_agent(tool_data, _prefetch_agent);
    const auto* _preferred_tool_agent = get_tool_agent(tool_data, _preferred_agent);

    auto tid = static_cast<std::uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name =
        fmt::format("KFD Page Migrate [{}->{}]", agent_label(_src_tool_agent),
                    agent_label(_dst_tool_agent));
    cache_add_track(track_name.c_str(), tid);

    auto args_str = fmt::format(
        "0;;std::uint64_t;;start_address;;{:#x};;"
        "1;;std::uint64_t;;end_address;;{:#x};;"
        "2;;string;;src_agent;;{};;"
        "3;;string;;dst_agent;;{};;"
        "4;;string;;prefetch_agent;;{};;"
        "5;;string;;preferred_agent;;{};;"
        "6;;int;;error_code;;{};;",
        _start_addr, _end_addr, agent_node_id_str(_src_tool_agent),
        agent_node_id_str(_dst_tool_agent), agent_node_id_str(_prefetch_tool_agent),
        agent_node_id_str(_preferred_tool_agent), _error_code);

    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));

    if(!_src_tool_agent || !_src_tool_agent->agent)
        LOG_WARNING("KFD page migrate: source agent not found for agent_id={}",
                    _src_agent.handle);

    auto src_dev_id   = _src_tool_agent ? _src_tool_agent->device_id : 0;
    auto src_dev_type = (_src_tool_agent && _src_tool_agent->agent)
                            ? _src_tool_agent->agent->type
                            : agent_type::CPU;

    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid, _name, _beg_ns, _end_ns, std::move(args_str),
        trait::name<category::rocm_kfd_page_migrate>::value, std::move(track_name), "{}",
        static_cast<std::uint32_t>(src_dev_id), static_cast<std::uint8_t>(src_dev_type),
        trait::name<category::rocm_kfd_page_migrate>::value, pmc_value,
        std::optional<std::int64_t>(_pid) });
}

// ─── tool_kfd_queue_callback<Wrapper> ────────────────────────────────────────

template <typename Wrapper>
void
tool_kfd_queue_callback(const client_data<Wrapper>*      tool_data,
                        const backend::kfd_queue_record* record)
{
    using namespace kfd_events_detail;
    if(!record) return;

    auto        _beg_ns   = record->start_timestamp;
    auto        _end_ns   = record->end_timestamp;
    const auto* _name     = get_kfd_operation_name(record);
    auto        _pid      = record->pid;
    auto        _agent_id = record->agent_id;

    const auto* _agent = get_tool_agent(tool_data, _agent_id);

    auto tid = static_cast<std::uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = fmt::format("KFD Queue [{}]", agent_label(_agent));
    cache_add_track(track_name.c_str(), tid);

    auto args_str  = fmt::format("0;;string;;agent;;{};;", agent_node_id_str(_agent));
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));

    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid, _name, _beg_ns, _end_ns, std::move(args_str),
        trait::name<category::rocm_kfd_queue>::value, std::move(track_name), "{}",
        static_cast<std::uint32_t>(_agent ? _agent->device_id : 0),
        static_cast<std::uint8_t>(agent_type::GPU),
        trait::name<category::rocm_kfd_queue>::value, pmc_value,
        std::optional<std::int64_t>(_pid) });
}

// ─── tool_kfd_event_queue_callback<Wrapper> ───────────────────────────────────

template <typename Wrapper>
void
tool_kfd_event_queue_callback(const client_data<Wrapper>*            tool_data,
                              const backend::kfd_event_queue_record* record)
{
    using namespace kfd_events_detail;
    if(!record) return;
    if(record->operation != ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED) return;

    auto        _timestamp = record->timestamp;
    const auto* _name      = get_kfd_operation_name(record);
    auto        _pid       = record->pid;
    auto        _agent_id  = record->agent_id;

    const auto* _agent = get_tool_agent(tool_data, _agent_id);

    auto tid = static_cast<std::uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = fmt::format("KFD Event Queue [{}]", agent_label(_agent));
    cache_add_track(track_name.c_str(), tid);

    auto args_str  = fmt::format("0;;string;;agent;;{};;", agent_node_id_str(_agent));
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));

    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid, _name, _timestamp, _timestamp, std::move(args_str),
        trait::name<category::rocm_kfd_event_queue>::value, std::move(track_name), "{}",
        static_cast<std::uint32_t>(_agent ? _agent->device_id : 0),
        static_cast<std::uint8_t>(agent_type::GPU),
        trait::name<category::rocm_kfd_event_queue>::value, pmc_value,
        std::optional<std::int64_t>(_pid) });
}

// ─── tool_kfd_event_unmap_from_gpu_callback<Wrapper> ─────────────────────────

template <typename Wrapper>
void
tool_kfd_event_unmap_from_gpu_callback(const client_data<Wrapper>*            tool_data,
                                       const backend::kfd_event_unmap_record* record)
{
    using namespace kfd_events_detail;
    if(!record) return;

    auto        _timestamp  = record->timestamp;
    const auto* _name       = get_kfd_operation_name(record);
    auto        _pid        = record->pid;
    auto        _agent_id   = record->agent_id;
    auto        _start_addr = record->start_address.value;
    auto        _end_addr   = record->end_address.value;

    const auto* _agent = get_tool_agent(tool_data, _agent_id);

    auto tid = static_cast<std::uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = fmt::format("KFD Unmap from GPU [{}]", agent_label(_agent));
    cache_add_track(track_name.c_str(), tid);

    auto args_str  = fmt::format("0;;string;;agent;;{};;"
                                  "1;;std::uint64_t;;start_address;;{:#x};;"
                                  "2;;std::uint64_t;;end_address;;{:#x};;",
                                 agent_node_id_str(_agent), _start_addr, _end_addr);
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));

    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid, _name, _timestamp, _timestamp, std::move(args_str),
        trait::name<category::rocm_kfd_event_unmap_from_gpu>::value,
        std::move(track_name), "{}",
        static_cast<std::uint32_t>(_agent ? _agent->device_id : 0),
        static_cast<std::uint8_t>(agent_type::GPU),
        trait::name<category::rocm_kfd_event_unmap_from_gpu>::value, pmc_value,
        std::optional<std::int64_t>(_pid) });
}

// ─── tool_kfd_event_dropped_events_callback<Wrapper> ─────────────────────────

template <typename Wrapper>
void
tool_kfd_event_dropped_events_callback(const client_data<Wrapper>* /*tool_data*/,
                                       const backend::kfd_event_dropped_record* record)
{
    using namespace kfd_events_detail;
    if(!record) return;

    auto        _timestamp = record->timestamp;
    const auto* _name      = get_kfd_operation_name(record);
    auto        _pid       = record->pid;
    auto        _count     = record->count;

    auto tid = static_cast<std::uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = std::string{ "KFD Dropped Events" };
    cache_add_track(track_name.c_str(), tid);

    auto args_str  = fmt::format("0;;std::uint64_t;;count;;{};;", _count);
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));

    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid, _name, _timestamp, _timestamp, std::move(args_str),
        trait::name<category::rocm_kfd_event_dropped_events>::value,
        std::move(track_name), "{}", 0, static_cast<std::uint8_t>(agent_type::GPU),
        trait::name<category::rocm_kfd_event_dropped_events>::value, pmc_value,
        std::optional<std::int64_t>(_pid) });
}

}  // namespace rocprofsys::rocprofiler_sdk

#endif  // ROCPROFILER_VERSION >= 10000
