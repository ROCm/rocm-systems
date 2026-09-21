// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include <profiler-hub/shared_types.hpp>

namespace profiler_hub::writer_types
{

/**
 * @brief OWNERSHIP MODEL
 *
 * All string fields in writer_types are NON-OWNING views (std::string_view).
 * The caller MUST ensure the underlying string data remains valid for the
 * duration of the register_*() or insert_*() call.
 *
 * SAFE patterns:
 *   - String literals: Always valid
 *   - std::string member variables: Valid while object exists
 *   - Function-local std::string: Valid within function scope
 *
 * UNSAFE patterns (will cause undefined behavior):
 *   - ss.str() temporaries: Destroyed immediately after statement
 *   - std::string(x).data(): Destroyed immediately after statement
 *
 * After register_*() returns, the writer owns internal copies and the
 * original string data may be freed.
 *
 * Fields use:
 *   - std::string_view: For NOT NULL fields (must have a value)
 *   - std::optional<std::string_view>: For nullable fields (can be NULL in DB)
 */

/**
 * @brief Node id
 * @note This is a unique value which will be used to identify the node
 */
using node_id_t = size_t;
/**
 * @brief Process id
 * @note This is a unique value which will be used to identify the process
 */
using process_id_t = size_t;
/**
 * @brief Thread id
 * @note This is a unique value which will be used to identify the thread
 */
using thread_id_t = size_t;

/**
 * @brief Code object id
 * @note This is a unique value which will be used to identify the code object
 */
using code_object_id_t = size_t;
/**
 * @brief Kernel symbol id
 * @note This is a unique value which will be used to identify the kernel symbol
 */
using kernel_symbol_id_t = size_t;
/**
 * @brief Pmc description name
 * @note This is a unique value which will be used to identify the pmc description
 */
using pmc_description_name_t = std::string_view;
/**
 * @brief Stream id
 * @note This is a unique value which will be used to identify the stream
 */
using stream_id_t = size_t;
/**
 * @brief Queue id
 * @note This is a unique value which will be used to identify the queue
 */
using queue_id_t = size_t;
/**
 * @brief Track name
 * @note This is a unique value which will be used to identify the track
 */
using track_name_t = std::string_view;

using timestamp_ns_t = size_t;

constexpr std::string_view empty_json = "{}";

/**
 * @brief Agent unique id
 * @note This is a struct which will be used to identify the agent uniquely.
 * @param logical_index Logical index which will uniquely identify the agent.
 * @param agent_type Agent type which will uniquely identify the agent.
 */
struct agent_unique_id_t
{
    std::optional<std::string_view> agent_type;
    size_t                          type_index;

    bool operator==(const agent_unique_id_t& other) const noexcept
    {
        return agent_type == other.agent_type && type_index == other.type_index;
    }
};

/**
 * @brief Trace environment
 * @note This is a struct which will be used to identify the trace environment.
 * Put whatever is available about trace environment. If not available, leave it empty.
 */
struct trace_environment_t
{
    std::optional<node_id_t>    node_id;
    std::optional<process_id_t> process_id;
    std::optional<thread_id_t>  thread_id;

    std::optional<agent_unique_id_t> agent_id;
    std::optional<stream_id_t>       stream_id;
    std::optional<queue_id_t>        queue_id;

    std::optional<track_name_t> track_name;
};

// --------------------- Info Tables ---------------------

/**
 * @brief Node info
 * @note This is a struct which will be used to identify the node.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 */
struct node_info_t
{
    node_id_t        node_id;
    size_t           hash;
    std::string_view machine_id;

    std::optional<std::string_view> system_name;
    std::optional<std::string_view> hostname;
    std::optional<std::string_view> release;
    std::optional<std::string_view> version;
    std::optional<std::string_view> hardware_name;
    std::optional<std::string_view> domain_name;
};

/**
 * @brief Process info
 * @note This is a struct which will be used to identify the process.
 * @param pid Process id which will uniquely identify the process. Use this
 * value to refer to process_info.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 */
struct process_info_t
{
    size_t       ppid{};
    process_id_t pid{};
    size_t       init{};
    size_t       fini{};
    size_t       start{};
    size_t       end{};

    std::optional<std::string_view> command;
    std::string_view                environment = empty_json;
    std::string_view                extdata     = empty_json;

    node_id_t node_id{};
};

/**
 * @brief Agent info
 * @note This is a struct which will be used to identify the agent.
 * @param unique_id Unique id which will uniquely identify the agent.
 * @param node_id Node id which will uniquely identify the node. Use this value to
 * refer to node_info.
 * @param process_id Process id which will uniquely identify the process. Use this
 * value to refer to process_info.
 */
struct agent_info_t
{
    agent_unique_id_t unique_id{};

    size_t absolute_index{};
    size_t logical_index{};
    size_t uuid{};

    std::optional<std::string_view> name;
    std::optional<std::string_view> model_name;
    std::optional<std::string_view> vendor_name;
    std::optional<std::string_view> product_name;
    std::optional<std::string_view> user_name;
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

struct pmc_info_unique_id_t
{
    pmc_description_name_t           name;
    std::optional<agent_unique_id_t> agent_id;

    bool operator==(const pmc_info_unique_id_t& other) const noexcept
    {
        const bool are_names_same = name == other.name;
        if(agent_id.has_value() && other.agent_id.has_value())
        {
            return are_names_same && (agent_id.value() == other.agent_id.value());
        }
        return are_names_same;
    }
};

/**
 * @brief Pmc info
 * @note This is a struct which will be used to identify the pmc.
 * @param unique_id Unique id which will uniquely identify the pmc.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 * @param process_id Process id which will uniquely identify the process. Use this
 * value to refer to process_info.
 */
struct pmc_info_t
{
    pmc_info_unique_id_t unique_id;

    std::optional<std::string_view> target_arch;
    size_t                          event_code{};
    size_t                          instance_id{};
    std::string_view                symbol;
    std::optional<std::string_view> description;
    std::optional<std::string_view> long_description;
    std::optional<std::string_view> component;
    std::optional<std::string_view> units;
    std::optional<std::string_view> value_type;
    std::optional<std::string_view> block;
    std::optional<std::string_view> expression;
    size_t                          is_constant{};
    size_t                          is_derived{};
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

/**
 * @brief Thread info
 * @note This is a struct which will be used to identify the thread.
 * @param thread_id Thread id which will uniquely identify the thread.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 * @param process_id Process id which will uniquely identify the process. Use this
 * value to refer to process_info.
 */
struct thread_info_t
{
    size_t      parent_process_id{};
    thread_id_t thread_id{};

    std::optional<std::string_view> name;
    size_t                          start{};
    size_t                          end{};
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

/**
 * @brief Stream info
 * @note This is a struct which will be used to identify the stream.
 * @param stream_id Stream id which will uniquely identify the stream.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 * @param process_id Process id which will uniquely identify the process. Use this
 * value to refer to process_info.
 */
struct stream_info_t
{
    stream_id_t stream_id{};

    std::optional<std::string_view> name;
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

/**
 * @brief Queue info
 * @note This is a struct which will be used to identify the queue.
 * @param queue_id Queue id which will uniquely identify the queue.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 * @param process_id Process id which will uniquely identify the process. Use this
 * value to refer to process_info.
 */
struct queue_info_t
{
    queue_id_t queue_id{};

    std::optional<std::string_view> name;
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

/**
 * @brief Code object info
 * @note This is a struct which will be used to identify the code object.
 * @param id Code object id which will uniquely identify the code object.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 * @param process_id Process id which will uniquely identify the process. Use this
 * value to refer to process_info.
 * @param agent_id Agent id which will uniquely identify the agent. Use this value to
 * refer to agent_info.
 */
struct code_object_info_t
{
    code_object_id_t id{};

    std::optional<std::string_view> uri;
    size_t                          load_base{};
    size_t                          load_size{};
    size_t                          load_delta{};
    std::optional<std::string_view> storage_type;
    std::string_view                extdata = empty_json;

    node_id_t                        node_id{};
    process_id_t                     process_id{};
    std::optional<agent_unique_id_t> agent_id;
};

/**
 * @brief Kernel symbol info
 * @note This is a struct which will be used to identify the kernel symbol.
 * @param id Kernel symbol id which will uniquely identify the kernel symbol.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 * @param process_id Process id which will uniquely identify the process. Use this
 * value to refer to process_info.
 * @param code_obj_id Code object id which will uniquely identify the code object.
 */
struct kernel_symbol_info_t
{
    kernel_symbol_id_t id{};

    std::optional<std::string_view> name;
    std::optional<std::string_view> display_name;
    size_t                          kernel_object{};
    size_t                          kernarg_segment_size{};
    size_t                          kernarg_segment_alignment{};
    size_t                          group_segment_size{};
    size_t                          private_segment_size{};
    size_t                          sgpr_count{};
    size_t                          arch_vgpr_count{};
    size_t                          accum_vgpr_count{};
    std::string_view                extdata = empty_json;

    node_id_t        node_id{};
    process_id_t     process_id{};
    code_object_id_t code_obj_id{};
};

/**
 * @brief Track info
 * @note This is a struct which will be used to identify the track.
 * @param name Track name which will uniquely identify the track.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 * @param process_id Process id which will uniquely identify the process. Use this
 * value to refer to process_info.
 * @param thread_id Thread id which will uniquely identify the thread.
 */
struct track_info_t
{
    std::optional<track_name_t> name;
    std::string_view            extdata = empty_json;

    node_id_t                   node_id{};
    std::optional<process_id_t> process_id;
    std::optional<thread_id_t>  thread_id;

    bool operator==(const track_info_t& other) const noexcept
    {
        return name == other.name && node_id == other.node_id &&
               process_id == other.process_id && thread_id == other.thread_id;
    }
};

// --------------------- Data Tables ---------------------

/**
 * @brief Function argument data for API tracing.
 */
struct arg_data_t
{
    size_t           position{};  ///< Argument position (0-indexed)
    std::string_view type;        ///< Argument type name
    std::string_view name;        ///< Argument parameter name

    std::optional<std::string_view> value;  ///< Serialized argument value
    std::string_view                extdata = empty_json;
};

/**
 * @brief Common event metadata shared by all profiling events.
 * @note Maps to rocpd_event table. This is the base event data embedded in all
 * data records (regions, kernel dispatches, memory operations). Contains call
 * stack and source context information for debugging and analysis.
 * In schema v3, call_stack and line_info_list are serialized to JSON.
 * In schema v4, they map to separate rocpd_call_stack and rocpd_line_info tables.
 * The event_category maps to rocpd_string in v3 and rocpd_info_category in v4.
 */
struct event_data_t
{
    std::optional<size_t> stack_id;  ///< Unique identifier for this call stack instance
    std::optional<size_t> parent_stack_id;  ///< Parent stack ID for nested events
    std::optional<size_t> correlation_id;   ///< Correlation ID linking related events

    shared_types::call_stack_t          call_stack;      ///< Call stack at event time
    shared_types::source_context_list_t line_info_list;  ///< Source context information

    std::optional<std::string_view>
        event_category;  ///< Event category name (e.g., "HIP_API", "HSA_API")
    std::string_view extdata = empty_json;
};

/**
 * @brief A named time region representing a span of execution.
 * @note Maps to rocpd_region table. Represents user-annotated regions, API calls,
 * or any named time span.
 */
struct region_data_t
{
    std::optional<event_data_t> event;  ///< Common event metadata

    timestamp_ns_t   start_timestamp;  ///< Region start time (nanoseconds)
    timestamp_ns_t   end_timestamp;    ///< Region end time (nanoseconds)
    std::string_view name;             ///< Region name (e.g., function name, annotation)
    std::string_view extdata = empty_json;

    std::vector<arg_data_t> args;  ///< Optional function arguments
};

/**
 * @brief A point-in-time sample (instantaneous event).
 * @note Maps to rocpd_sample table. Used for counter samples, markers, or any
 * instantaneous event. Associated with a track for timeline visualization.
 */
struct sample_data_t
{
    timestamp_ns_t   timestamp{};  ///< Sample time (nanoseconds)
    track_info_t     track;
    std::string_view extdata = empty_json;
};

/**
 * @brief Performance counter (PMC) event data.
 * @note Maps to rocpd_pmc_event table. Records a hardware performance counter
 * sample with its value. The sample provides the timestamp, and the event
 * provides correlation and context information.
 */
struct pmc_event_data_t
{
    std::optional<event_data_t> event;    ///< Common event metadata
    double                      value{};  ///< Counter value
    std::string_view            extdata = empty_json;
    sample_data_t               sample;  ///< Timestamp information
};

/**
 * @brief GPU kernel dispatch event data.
 * @note Maps to rocpd_kernel_dispatch table. Records a GPU kernel execution
 * including launch configuration (grid/workgroup sizes), timing, and kernel
 * identification.
 */
struct kernel_dispatch_data_t
{
    std::optional<event_data_t> event;               ///< Common event metadata
    size_t                      dispatch_id{};       ///< Unique dispatch identifier
    timestamp_ns_t              start_timestamp{};   ///< Kernel start time (nanoseconds)
    timestamp_ns_t              end_timestamp{};     ///< Kernel end time (nanoseconds)
    kernel_symbol_id_t          kernel_symbol_id{};  ///< Kernel symbol id
    code_object_id_t            code_object_id{};    ///< Code object id
    size_t private_segment_size{};  ///< Private memory per work-item (bytes)
    size_t group_segment_size{};    ///< LDS memory per workgroup (bytes)
    size_t workgroup_size_x{};      ///< Workgroup size in X dimension
    size_t workgroup_size_y{};      ///< Workgroup size in Y dimension
    size_t workgroup_size_z{};      ///< Workgroup size in Z dimension
    size_t grid_size_x{};           ///< Grid size in X dimension
    size_t grid_size_y{};           ///< Grid size in Y dimension
    size_t grid_size_z{};           ///< Grid size in Z dimension

    std::optional<std::string_view> name;  ///< Kernel name (region_name_id nullable)
    std::string_view                extdata = empty_json;
};

/**
 * @brief Memory copy operation event data.
 * @note Maps to rocpd_memory_copy table. Records a memory transfer operation
 * including source/destination addresses, size, and timing. Used for tracking
 * host-to-device, device-to-host, and device-to-device copies.
 */
struct memory_copy_data_t
{
    std::optional<event_data_t> event;              ///< Common event metadata
    timestamp_ns_t              start_timestamp{};  ///< Copy start time (nanoseconds)
    timestamp_ns_t              end_timestamp{};    ///< Copy end time (nanoseconds)
    std::optional<agent_unique_id_t> dst_agent_id;  ///< Destination agent id
    std::optional<size_t>            dst_address;   ///< Destination memory address
    std::optional<agent_unique_id_t> src_agent_id;  ///< Source agent id
    std::optional<size_t>            src_address;   ///< Source memory address
    size_t                           size{};        ///< Transfer size (bytes)

    std::string_view                name;         ///< Operation name
    std::optional<std::string_view> region_name;  ///< Region name
    std::string_view                extdata = empty_json;
};

/**
 * @brief Memory allocation event data.
 * @note Maps to rocpd_memory_allocate table. Records memory allocation and
 * deallocation operations including address, size, allocation type, and timing.
 */
struct memory_alloc_data_t
{
    std::optional<event_data_t> event;  ///< Common event metadata

    std::optional<std::string_view>
        type;  ///< Allocation type (e.g., "hipMalloc", "hipHostMalloc")
    std::optional<std::string_view>
                          level;  ///< Memory level (e.g., "device", "host", "managed")
    timestamp_ns_t        start_timestamp{};  ///< Allocation start time (nanoseconds)
    timestamp_ns_t        end_timestamp{};    ///< Allocation end time (nanoseconds)
    std::optional<size_t> address;            ///< Allocated memory address
    size_t                size{};             ///< Allocation size (bytes)
    std::string_view      extdata = empty_json;
};

}  // namespace profiler_hub::writer_types
