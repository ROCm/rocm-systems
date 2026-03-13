// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/rocprofiler-sdk/kfd_events.hpp"

#if ROCPROFSYS_HAS_KFD_EVENTS

#    include "core/categories.hpp"
#    include "core/trace_cache/buffer_storage.hpp"
#    include "core/trace_cache/cache_manager.hpp"
#    include "core/trace_cache/metadata_registry.hpp"
#    include "library/rocprofiler-sdk.hpp"
#    include "library/rocprofiler-sdk/fwd.hpp"

#    include "logger/debug.hpp"

#    include <rocprofiler-sdk/buffer_tracing.h>
#    include <rocprofiler-sdk/kfd/kfd_id.h>

#    include <fmt/format.h>
#    include <unistd.h>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

extern client_data* tool_data;

namespace
{
// Helper to get operation name for KFD events
template <typename RecordT>
std::string
get_kfd_operation_name([[maybe_unused]] const RecordT* record)
{
    return "KFD Event";
}

template <>
std::string
get_kfd_operation_name(const rocprofiler_buffer_tracing_kfd_page_fault_record_t* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_MIGRATED: return "ReadFaultMigrated";
        case ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_UPDATED: return "ReadFaultUpdated";
        case ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_MIGRATED: return "WriteFaultMigrated";
        case ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_UPDATED: return "WriteFaultUpdated";
        default: return "PageFault";
    }
}

template <>
std::string
get_kfd_operation_name(const rocprofiler_buffer_tracing_kfd_page_migrate_record_t* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_PAGE_MIGRATE_PREFETCH: return "Prefetch";
        case ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_GPU: return "PageFaultGPU";
        case ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_CPU: return "PageFaultCPU";
        case ROCPROFILER_KFD_PAGE_MIGRATE_TTM_EVICTION: return "TTMEviction";
        default: return "PageMigrate";
    }
}

template <>
std::string
get_kfd_operation_name(const rocprofiler_buffer_tracing_kfd_queue_record_t* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_QUEUE_EVICT_SVM: return "EvictSVM";
        case ROCPROFILER_KFD_QUEUE_EVICT_USERPTR: return "EvictUserPtr";
        case ROCPROFILER_KFD_QUEUE_EVICT_TTM: return "EvictTTM";
        case ROCPROFILER_KFD_QUEUE_EVICT_SUSPEND: return "EvictSuspend";
        case ROCPROFILER_KFD_QUEUE_EVICT_CRIU_CHECKPOINT: return "EvictCRIUCheckpoint";
        case ROCPROFILER_KFD_QUEUE_EVICT_CRIU_RESTORE: return "EvictCRIURestore";
        default: return "Queue";
    }
}

template <>
std::string
get_kfd_operation_name(const rocprofiler_buffer_tracing_kfd_event_queue_record_t* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SVM: return "EventQueueEvictSVM";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR: return "EventQueueEvictUserPtr";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM: return "EventQueueEvictTTM";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND: return "EventQueueEvictSuspend";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT:
            return "EventQueueEvictCRIUCheckpoint";
        case ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE:
            return "EventQueueEvictCRIURestore";
        case ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED:
            return "EventQueueRestoreRescheduled";
        case ROCPROFILER_KFD_EVENT_QUEUE_RESTORE: return "EventQueueRestore";
        default: return "EventQueue";
    }
}

template <>
std::string
get_kfd_operation_name(
    const rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t* record)
{
    switch(record->operation)
    {
        case ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY: return "UnmapMMUNotify";
        case ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE:
            return "UnmapMMUNotifyMigrate";
        case ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_UNMAP_FROM_CPU: return "UnmapFromCPU";
        default: return "UnmapFromGPU";
    }
}

template <>
std::string
get_kfd_operation_name(
    const rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t* /*record*/)
{
    return "DroppedEvents";
}

// Helper to cache category
template <typename CategoryT>
void
cache_category()
{
    trace_cache::get_metadata_registry().add_string(trait::name<CategoryT>::value);
}

// Helper to add thread info to metadata
void
cache_add_thread_info(uint64_t tid)
{
    trace_cache::get_metadata_registry().add_thread_info(
        { getppid(), getpid(), tid, 0, 0, "{}" });
}

// Helper to add track info to metadata
void
cache_add_track(const char* track_name, uint64_t tid)
{
    trace_cache::get_metadata_registry().add_track({ track_name, tid, "{}" });
}

// Get agent helper — searches GPU agents first, then CPU agents
const tool_agent*
get_tool_agent(rocprofiler_agent_id_t agent_id)
{
    const auto* agent = tool_data->get_gpu_tool_agent(agent_id);
    if(agent) return agent;
    for(const auto& itr : tool_data->cpu_agents)
        if(agent_id.handle == itr.agent->handle) return &itr;
    return nullptr;
}

// Format agent as "TYPE INDEX" for track names (e.g., "GPU 0", "CPU 1")
std::string
agent_label(const tool_agent* _agent)
{
    if(!_agent || !_agent->agent) return "?";
    auto type = _agent->agent->type;
    auto idx  = _agent->device_id;
    return fmt::format("{} {}", type == agent_type::GPU ? "GPU" : "CPU", idx);
}

// Resolve agent_id to node_id, matching rocprofiler-sdk's agent_node_id() behavior.
// Falls back to the global agent manager if tool_agent is null.
std::string
agent_node_id_str(const tool_agent* _agent)
{
    if(_agent && _agent->agent) return std::to_string(_agent->agent->node_id);
    return "null";
}

std::string
agent_node_id_str(rocprofiler_agent_id_t agent_id)
{
    const auto* _agent = get_tool_agent(agent_id);
    if(_agent && _agent->agent) return std::to_string(_agent->agent->node_id);
    return "null";
}

// Helper to compute PMC value for KFD events
// For paired events: returns address range size
// For instant events: returns 1
template <typename RecordT>
uint64_t
get_kfd_pmc_value([[maybe_unused]] const RecordT* record)
{
    return 1;  // Default for instant events
}

template <>
uint64_t
get_kfd_pmc_value(const rocprofiler_buffer_tracing_kfd_page_fault_record_t* record)
{
    return record->address.value;
}

template <>
uint64_t
get_kfd_pmc_value(const rocprofiler_buffer_tracing_kfd_page_migrate_record_t* record)
{
    // Return the size of the address range being migrated
    return record->end_address.value - record->start_address.value;
}

template <>
uint64_t
get_kfd_pmc_value(const rocprofiler_buffer_tracing_kfd_queue_record_t* /*record*/)
{
    return 1;  // Queue suspend is counted as 1 event
}

template <>
uint64_t
get_kfd_pmc_value(
    const rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t* record)
{
    return record->count;
}

template <>
uint64_t
get_kfd_pmc_value(
    const rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t* record)
{
    return record->end_address.value - record->start_address.value;
}

}  // namespace

void
kfd_event_metadata_initialize()
{
    // Initialize category strings in metadata registry
    cache_category<category::kfd_page_fault>();
    cache_category<category::kfd_page_migrate>();
    cache_category<category::kfd_queue>();
    cache_category<category::kfd_event_queue>();
    cache_category<category::kfd_event_unmap_from_gpu>();
    cache_category<category::kfd_event_dropped_events>();

    // Register PMC info for each KFD event type to match rocprofiler-sdk schema
    // This ensures data is stored in rocpd_info_pmc and rocpd_pmc_event tables
    constexpr size_t EVENT_CODE  = 0;
    constexpr size_t INSTANCE_ID = 0;
    constexpr auto*  COMPONENT   = "rocm";
    constexpr auto*  BLOCK       = "KFD";
    constexpr auto*  EXPRESSION  = "";
    constexpr auto*  TARGET_ARCH = "GPU";

    // KFD events are not tied to a specific GPU device, so we use device_id = 0
    // The actual agent info is stored in the JSON extdata
    constexpr uint32_t DEVICE_ID = 0;

    // Register PMC info for each KFD buffer tracing kind
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::kfd_page_fault>::value, "KFD Page Fault Events",
          trait::name<category::kfd_page_fault>::description,
          "KFD page fault paired records", COMPONENT, "events", trace_cache::ABSOLUTE,
          BLOCK, EXPRESSION, 0, 0, "{}" });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::kfd_page_migrate>::value, "KFD Page Migration Events",
          trait::name<category::kfd_page_migrate>::description,
          "KFD page migration paired records", COMPONENT, "events", trace_cache::ABSOLUTE,
          BLOCK, EXPRESSION, 0, 0, "{}" });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::kfd_queue>::value, "KFD Queue Events",
          trait::name<category::kfd_queue>::description,
          "KFD queue eviction/restore paired records", COMPONENT, "events",
          trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0, "{}" });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::kfd_event_queue>::value, "KFD Event Queue Operations",
          trait::name<category::kfd_event_queue>::description,
          "KFD queue eviction/restore events", COMPONENT, "events", trace_cache::ABSOLUTE,
          BLOCK, EXPRESSION, 0, 0, "{}" });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::kfd_event_unmap_from_gpu>::value,
          "KFD Unmap from GPU Events",
          trait::name<category::kfd_event_unmap_from_gpu>::description,
          "KFD unmap from GPU events", COMPONENT, "events", trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0, "{}" });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, DEVICE_ID, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::kfd_event_dropped_events>::value, "KFD Dropped Events",
          trait::name<category::kfd_event_dropped_events>::description,
          "KFD dropped_events events", COMPONENT, "count", trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0, "{}" });
}

void
tool_kfd_page_fault_callback(
    const rocprofiler_buffer_tracing_kfd_page_fault_record_t* record)
{
    if(!record) return;

    auto _beg_ns   = record->start_timestamp;
    auto _end_ns   = record->end_timestamp;
    auto _name     = get_kfd_operation_name(record);
    auto _pid      = record->pid;
    auto _agent_id = record->agent_id;
    auto _address  = record->address.value;

    const auto* _agent = get_tool_agent(_agent_id);

    // Cache the data
    cache_category<category::kfd_page_fault>();

    // Use PID as pseudo thread ID for KFD events since they're kernel-level
    auto tid = static_cast<uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = fmt::format("KFD Page Fault [{}]", agent_label(_agent));
    cache_add_track(track_name.c_str(), tid);

    // Create JSON extdata matching rocprofiler-sdk format (agent resolved to node_id)
    auto _agent_nid   = agent_node_id_str(_agent);
    auto extdata_json = fmt::format(
        "{{\"kfd\":{{\"agent_id\":{},\"address\":\"{:#x}\",\"operation\":{}}}}}",
        _agent_nid, _address, static_cast<int>(record->operation));

    // Create region with additional info in args
    auto args_str =
        fmt::format("{{\"address\":\"{:#x}\",\"agent\":{}}}", _address, _agent_nid);

    // Store as kfd_sample which combines region and PMC event data
    // This matches rocprofiler-sdk logic where both share the same event_id
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));
    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid,                                                    // thread_id
        _name.c_str(),                                          // name
        _beg_ns,                                                // start_timestamp
        _end_ns,                                                // end_timestamp
        args_str.c_str(),                                       // args_str
        trait::name<category::kfd_page_fault>::value,           // category
        track_name,                                             // track_name
        extdata_json,                                           // event_metadata (JSON)
        static_cast<uint32_t>(_agent ? _agent->device_id : 0),  // device_id
        static_cast<uint8_t>(agent_type::GPU),                  // device_type
        trait::name<category::kfd_page_fault>::value,           // pmc_info_name
        pmc_value,                                              // value
        std::optional<int64_t>(_pid)                            // system_tid
    });
}

void
tool_kfd_page_migrate_callback(
    const rocprofiler_buffer_tracing_kfd_page_migrate_record_t* record)
{
    if(!record) return;

    auto _beg_ns          = record->start_timestamp;
    auto _end_ns          = record->end_timestamp;
    auto _name            = get_kfd_operation_name(record);
    auto _pid             = record->pid;
    auto _start_addr      = record->start_address.value;
    auto _end_addr        = record->end_address.value;
    auto _src_agent       = record->src_agent;
    auto _dst_agent       = record->dst_agent;
    auto _prefetch_agent  = record->prefetch_agent;
    auto _preferred_agent = record->preferred_agent;
    auto _error_code      = record->error_code;

    const auto* _src_tool_agent       = get_tool_agent(_src_agent);
    const auto* _dst_tool_agent       = get_tool_agent(_dst_agent);
    const auto* _prefetch_tool_agent  = get_tool_agent(_prefetch_agent);
    const auto* _preferred_tool_agent = get_tool_agent(_preferred_agent);

    // Cache the data
    cache_category<category::kfd_page_migrate>();

    auto tid = static_cast<uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name =
        fmt::format("KFD Page Migrate [{}->{}]", agent_label(_src_tool_agent),
                    agent_label(_dst_tool_agent));
    cache_add_track(track_name.c_str(), tid);

    // Resolve agent IDs to node IDs matching rocprofiler-sdk format
    auto _src_nid       = agent_node_id_str(_src_tool_agent);
    auto _dst_nid       = agent_node_id_str(_dst_tool_agent);
    auto _prefetch_nid  = agent_node_id_str(_prefetch_tool_agent);
    auto _preferred_nid = agent_node_id_str(_preferred_tool_agent);

    // Create JSON extdata matching rocprofiler-sdk format (agents resolved to node_ids)
    auto extdata_json =
        fmt::format("{{\"kfd\":{{\"start_address\":\"{:#x}\",\"end_address\":\"{:#x}\","
                    "\"src_agent_id\":{},\"dst_agent_id\":{},\"prefetch_agent_id\":{},"
                    "\"preferred_agent_id\":{},\"error_code\":{}}}}}",
                    _start_addr, _end_addr, _src_nid, _dst_nid, _prefetch_nid,
                    _preferred_nid, _error_code);

    // Create region with additional info in args
    auto args_str = fmt::format("{{\"start_address\":\"{:#x}\",\"end_address\":\"{:#x}\","
                                "\"src_agent\":{},\"dst_agent\":{},\"prefetch_agent\":{},"
                                "\"preferred_agent\":{},\"error_code\":{}}}",
                                _start_addr, _end_addr, _src_nid, _dst_nid, _prefetch_nid,
                                _preferred_nid, _error_code);

    // Store as kfd_sample which combines region and PMC event data
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));
    auto device_id = _dst_tool_agent ? _dst_tool_agent->device_id : 0;
    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid,                                             // thread_id
        _name.c_str(),                                   // name
        _beg_ns,                                         // start_timestamp
        _end_ns,                                         // end_timestamp
        args_str.c_str(),                                // args_str
        trait::name<category::kfd_page_migrate>::value,  // category
        track_name,                                      // track_name
        extdata_json,                                    // event_metadata (JSON)
        static_cast<uint32_t>(device_id),                // device_id
        static_cast<uint8_t>(agent_type::GPU),           // device_type
        trait::name<category::kfd_page_migrate>::value,  // pmc_info_name
        pmc_value,                                       // value
        std::optional<int64_t>(_pid)                     // system_tid
    });
}

void
tool_kfd_queue_callback(const rocprofiler_buffer_tracing_kfd_queue_record_t* record)
{
    if(!record) return;

    auto _beg_ns   = record->start_timestamp;
    auto _end_ns   = record->end_timestamp;
    auto _name     = get_kfd_operation_name(record);
    auto _pid      = record->pid;
    auto _agent_id = record->agent_id;

    const auto* _agent = get_tool_agent(_agent_id);

    // Cache the data
    cache_category<category::kfd_queue>();

    auto tid = static_cast<uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = fmt::format("KFD Queue [{}]", agent_label(_agent));
    cache_add_track(track_name.c_str(), tid);

    // Create JSON extdata matching rocprofiler-sdk format (agent resolved to node_id)
    auto _agent_nid   = agent_node_id_str(_agent);
    auto extdata_json = fmt::format("{{\"kfd\":{{\"agent_id\":{}}}}}", _agent_nid);

    // Create region with additional info in args
    auto args_str = fmt::format("{{\"agent\":{}}}", _agent_nid);

    // Store as kfd_sample which combines region and PMC event data
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));
    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid,                                                    // thread_id
        _name.c_str(),                                          // name
        _beg_ns,                                                // start_timestamp
        _end_ns,                                                // end_timestamp
        args_str.c_str(),                                       // args_str
        trait::name<category::kfd_queue>::value,                // category
        track_name,                                             // track_name
        extdata_json,                                           // event_metadata (JSON)
        static_cast<uint32_t>(_agent ? _agent->device_id : 0),  // device_id
        static_cast<uint8_t>(agent_type::GPU),                  // device_type
        trait::name<category::kfd_queue>::value,                // pmc_info_name
        pmc_value,                                              // value
        std::optional<int64_t>(_pid)                            // system_tid
    });
}

void
tool_kfd_event_queue_callback(
    const rocprofiler_buffer_tracing_kfd_event_queue_record_t* record)
{
    if(!record) return;

    // Only process RESTORE_RESCHEDULED operations
    if(record->operation != ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED) return;

    auto _timestamp = record->timestamp;
    auto _name      = get_kfd_operation_name(record);
    auto _pid       = record->pid;
    auto _agent_id  = record->agent_id;

    const auto* _agent = get_tool_agent(_agent_id);

    // Cache the data
    cache_category<category::kfd_event_queue>();

    auto tid = static_cast<uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = fmt::format("KFD Event Queue [{}]", agent_label(_agent));
    cache_add_track(track_name.c_str(), tid);

    // Create JSON extdata matching rocprofiler-sdk format (agent resolved to node_id)
    auto _agent_nid   = agent_node_id_str(_agent);
    auto extdata_json = fmt::format("{{\"kfd\":{{\"agent_id\":{}}}}}", _agent_nid);

    // Create region with additional info in args
    auto args_str = fmt::format("{{\"agent\":{}}}", _agent_nid);

    // Event queue records are instant events (begin == end)
    // Store as kfd_sample which combines region and PMC event data
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));
    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid,                                            // thread_id
        _name.c_str(),                                  // name
        _timestamp,                                     // start_timestamp
        _timestamp,                                     // end_timestamp (instant event)
        args_str.c_str(),                               // args_str
        trait::name<category::kfd_event_queue>::value,  // category
        track_name,                                     // track_name
        extdata_json,                                   // event_metadata (JSON)
        static_cast<uint32_t>(_agent ? _agent->device_id : 0),  // device_id
        static_cast<uint8_t>(agent_type::GPU),                  // device_type
        trait::name<category::kfd_event_queue>::value,          // pmc_info_name
        pmc_value,                                              // value
        std::optional<int64_t>(_pid)                            // system_tid
    });
}

void
tool_kfd_event_unmap_from_gpu_callback(
    const rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t* record)
{
    if(!record) return;

    auto _timestamp  = record->timestamp;
    auto _name       = get_kfd_operation_name(record);
    auto _pid        = record->pid;
    auto _agent_id   = record->agent_id;
    auto _start_addr = record->start_address.value;
    auto _end_addr   = record->end_address.value;

    const auto* _agent = get_tool_agent(_agent_id);

    // Cache the data
    cache_category<category::kfd_event_unmap_from_gpu>();

    auto tid = static_cast<uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = fmt::format("KFD Unmap from GPU [{}]", agent_label(_agent));
    cache_add_track(track_name.c_str(), tid);

    // Create JSON extdata matching rocprofiler-sdk format (agent resolved to node_id)
    auto _agent_nid   = agent_node_id_str(_agent);
    auto extdata_json = fmt::format("{{\"kfd\":{{\"agent_id\":{},\"start_address\":\"{:#"
                                    "x}\",\"end_address\":\"{:#x}\"}}}}",
                                    _agent_nid, _start_addr, _end_addr);

    // Create region with additional info in args
    auto args_str = fmt::format(
        "{{\"agent\":{},\"start_address\":\"{:#x}\",\"end_address\":\"{:#x}\"}}",
        _agent_nid, _start_addr, _end_addr);

    // Unmap events are instant events (begin == end)
    // Store as kfd_sample which combines region and PMC event data
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));
    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid,               // thread_id
        _name.c_str(),     // name
        _timestamp,        // start_timestamp
        _timestamp,        // end_timestamp (instant event)
        args_str.c_str(),  // args_str
        trait::name<category::kfd_event_unmap_from_gpu>::value,  // category
        track_name,                                              // track_name
        extdata_json,                                            // event_metadata (JSON)
        static_cast<uint32_t>(_agent ? _agent->device_id : 0),   // device_id
        static_cast<uint8_t>(agent_type::GPU),                   // device_type
        trait::name<category::kfd_event_unmap_from_gpu>::value,  // pmc_info_name
        pmc_value,                                               // value
        std::optional<int64_t>(_pid)                             // system_tid
    });
}

void
tool_kfd_event_dropped_events_callback(
    const rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t* record)
{
    if(!record) return;

    auto _timestamp = record->timestamp;
    auto _name      = get_kfd_operation_name(record);
    auto _pid       = record->pid;
    auto _count     = record->count;

    // Cache the data
    cache_category<category::kfd_event_dropped_events>();

    auto tid = static_cast<uint64_t>(_pid);
    cache_add_thread_info(tid);

    auto track_name = std::string{ "KFD Dropped Events" };
    cache_add_track(track_name.c_str(), tid);

    // Create JSON extdata matching rocprofiler-sdk format
    auto extdata_json = fmt::format("{{\"kfd\":{{\"count\":{}}}}}", _count);

    // Create region with additional info in args
    auto args_str = fmt::format("{{\"count\":{}}}", _count);

    // Dropped events are instant events (begin == end)
    // Store as kfd_sample which combines region and PMC event data
    auto pmc_value = static_cast<double>(get_kfd_pmc_value(record));
    trace_cache::get_buffer_storage().store(trace_cache::kfd_sample{
        tid,               // thread_id
        _name.c_str(),     // name
        _timestamp,        // start_timestamp
        _timestamp,        // end_timestamp (instant event)
        args_str.c_str(),  // args_str
        trait::name<category::kfd_event_dropped_events>::value,  // category
        track_name,                                              // track_name
        extdata_json,                                            // event_metadata (JSON)
        0,                                      // device_id = 0 (no specific device)
        static_cast<uint8_t>(agent_type::GPU),  // device_type
        trait::name<category::kfd_event_dropped_events>::value,  // pmc_info_name
        pmc_value,                                               // value
        std::optional<int64_t>(_pid)                             // system_tid
    });
}

}  // namespace rocprofiler_sdk

}  // namespace rocprofsys

#endif
