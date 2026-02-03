// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rocstorage::data_types
{

/***
 * @brief Node id
 * @note This is a unique value which will be used to identify the node
 */
using node_id_t = size_t;
/***
 * @brief Process id
 * @note This is a unique value which will be used to identify the process
 */
using process_id_t = size_t;
/***
 * @brief Thread id
 * @note This is a unique value which will be used to identify the thread
 */
using thread_id_t = size_t;

/***
 * @brief Code object id
 * @note This is a unique value which will be used to identify the code object
 */
using code_object_id_t = size_t;
/***
 * @brief Kernel symbol id
 * @note This is a unique value which will be used to identify the kernel symbol
 */
using kernel_symbol_id_t = size_t;
/***
 * @brief Pmc description name
 * @note This is a unique value which will be used to identify the pmc description
 */
using pmc_description_name_t = const char*;
/***
 * @brief Stream id
 * @note This is a unique value which will be used to identify the stream
 */
using stream_id_t = size_t;
/***
 * @brief Queue id
 * @note This is a unique value which will be used to identify the queue
 */
using queue_id_t = size_t;
/***
 * @brief Track name
 * @note This is a unique value which will be used to identify the track
 */
using track_name_t = const char*;

using timestamp_ns_t = size_t;

/***
 * @brief Agent unique id
 * @note This is a struct which will be used to identify the agent uniquely.
 * @param logical_index Logical index which will uniquely identify the agent.
 * @param agent_type Agent type which will uniquely identify the agent.
 */
struct agent_unique_id_t
{
    const char* agent_type;
    size_t      type_index;

    bool operator==(const agent_unique_id_t& other) const noexcept
    {
        return agent_type == other.agent_type && type_index == other.type_index;
    }
};

/***
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

/***
 * @brief Node info
 * @note This is a struct which will be used to identify the node.
 * @param node_id Node id which will uniquely identify the node. Use this value to refer
 * to node_info.
 */
struct node_info_t
{
    node_id_t   node_id;
    size_t      hash;
    const char* machine_id;
    const char* system_name;
    const char* hostname;
    const char* release;
    const char* version;
    const char* hardware_name;
    const char* domain_name;
};

using node_info_ptr_t  = std::shared_ptr<node_info_t>;
using node_info_list_t = std::vector<node_info_ptr_t>;

/***
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
    const char*  command{};
    const char*  environment{};
    const char*  extdata = "{}";

    node_id_t node_id{};
};

using process_info_ptr_t  = std::shared_ptr<process_info_t>;
using process_info_list_t = std::vector<process_info_ptr_t>;

/***
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

    size_t      absolute_index{};
    size_t      logical_index{};
    size_t      uuid{};
    const char* name{};
    const char* model_name{};
    const char* vendor_name{};
    const char* product_name{};
    const char* user_name{};
    const char* extdata = "{}";

    node_id_t    node_id{};
    process_id_t process_id{};
};

using agent_info_ptr_t  = std::shared_ptr<agent_info_t>;
using agent_info_list_t = std::vector<agent_info_ptr_t>;

struct pmc_info_unique_id_t
{
    pmc_description_name_t           name{};
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

/***
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
    const char*          target_arch{};
    size_t               event_code{};
    size_t               instance_id{};
    const char*          symbol{};
    const char*          description{};
    const char*          long_description{};
    const char*          component{};
    const char*          units{};
    const char*          value_type{};
    const char*          block{};
    const char*          expression{};
    size_t               is_constant{};
    size_t               is_derived{};
    const char*          extdata = "{}";

    node_id_t    node_id{};
    process_id_t process_id{};
};

using pmc_info_ptr_t  = std::shared_ptr<pmc_info_t>;
using pmc_info_list_t = std::vector<pmc_info_ptr_t>;

/***
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
    const char* name{};
    size_t      start{};
    size_t      end{};
    const char* extdata = "{}";

    node_id_t    node_id{};
    process_id_t process_id{};
};

using thread_info_ptr_t  = std::shared_ptr<thread_info_t>;
using thread_info_list_t = std::vector<thread_info_ptr_t>;

/***
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
    const char* name{};
    const char* extdata = "{}";

    node_id_t    node_id{};
    process_id_t process_id{};
};

using stream_info_ptr_t  = std::shared_ptr<stream_info_t>;
using stream_info_list_t = std::vector<stream_info_ptr_t>;

/***
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
    queue_id_t  queue_id{};
    const char* name{};
    const char* extdata = "{}";

    node_id_t    node_id{};
    process_id_t process_id{};
};

using queue_info_ptr_t  = std::shared_ptr<queue_info_t>;
using queue_info_list_t = std::vector<queue_info_ptr_t>;

/***
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
    const char*      uri{};
    size_t           load_base{};
    size_t           load_size{};
    size_t           load_delta{};
    const char*      storage_type{};
    const char*      extdata = "{}";

    node_id_t                        node_id{};
    process_id_t                     process_id{};
    std::optional<agent_unique_id_t> agent_id;
};

using code_object_info_ptr_t  = std::shared_ptr<code_object_info_t>;
using code_object_info_list_t = std::vector<code_object_info_ptr_t>;

/***
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
    const char*        name{};
    const char*        display_name{};
    size_t             kernel_object{};
    size_t             kernarg_segment_size{};
    size_t             kernarg_segment_alignment{};
    size_t             group_segment_size{};
    size_t             private_segment_size{};
    size_t             sgpr_count{};
    size_t             arch_vgpr_count{};
    size_t             accum_vgpr_count{};
    const char*        extdata = "{}";

    node_id_t        node_id{};
    process_id_t     process_id{};
    code_object_id_t code_obj_id{};
};

using kernel_symbol_info_ptr_t  = std::shared_ptr<kernel_symbol_info_t>;
using kernel_symbol_info_list_t = std::vector<kernel_symbol_info_ptr_t>;

/***
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
    const char*                 extdata = "{}";

    node_id_t                   node_id{};
    std::optional<process_id_t> process_id;
    std::optional<thread_id_t>  thread_id;

    bool operator==(const track_info_t& other) const noexcept
    {
        return name == other.name && node_id == other.node_id &&
               process_id == other.process_id && thread_id == other.thread_id;
    }
};

using track_info_ptr_t  = std::shared_ptr<track_info_t>;
using track_info_list_t = std::vector<track_info_ptr_t>;

// --------------------- Call Stack & Line Info Abstract Data Types ------------------

/***
 * @brief Memory address range representing a loaded code object region.
 */
struct address_range_info_t
{
    size_t      address_base;  ///< Base load address of the code object
    size_t      address_low;   ///< Lower bound of the address range (>= address_base)
    size_t      address_high;  ///< Upper bound of the address range (>= address_low)
    const char* extdata = "{}";
};

using address_range_info_ptr_t  = std::shared_ptr<address_range_info_t>;
using address_range_info_list_t = std::vector<address_range_info_ptr_t>;

/***
 * @brief Program counter information representing a location in executable code.
 */
struct program_counter_info_t
{
    const char*           function;  ///< Function or symbol name at this program counter
    const char*           filename;  ///< Source file path (if available)
    std::optional<size_t> line_number;  ///< Line number in source file (if available)
    const char*           extdata = "{}";
};

using program_counter_info_ptr_t  = std::shared_ptr<program_counter_info_t>;
using program_counter_info_list_t = std::vector<program_counter_info_ptr_t>;

/***
 * @brief A single frame in a call stack.
 */
struct stack_frame_t
{
    std::optional<program_counter_info_t>
        program_counter;                                ///< Location info for this frame
    std::optional<address_range_info_t> address_range;  ///< Code object memory range
    const char*                         extdata = "{}";
};

using stack_frame_ptr_t  = std::shared_ptr<stack_frame_t>;
using stack_frame_list_t = std::vector<stack_frame_ptr_t>;

/***
 * @brief Complete call stack as an ordered collection of stack frames.
 * @note Front element (index 0) is the top of the stack (most recent call).
 * Back element is the bottom of the stack (e.g., main). Depth is implicit
 * from position in the deque. Maps to multiple rocpd_call_stack rows in schema v4.
 */
using call_stack_t = std::deque<stack_frame_t>;

/***
 * @brief Source code context containing actual source lines and disassembly.
 */
struct source_code_info_t
{
    std::optional<const char*> filename;  ///< Source file path
    std::optional<size_t>
        starting_line_number;  ///< First line number in source_code_lines
    std::vector<const char*> source_code_lines;           ///< Actual source code lines
    std::vector<const char*> assembly_instruction_lines;  ///< Disassembled instructions
    const char*              extdata = "{}";
};

using source_code_info_ptr_t  = std::shared_ptr<source_code_info_t>;
using source_code_info_list_t = std::vector<source_code_info_ptr_t>;

/***
 * @brief Line info entry linking source code context with program counter location.
 */
struct line_info_entry_t
{
    std::optional<source_code_info_t>     source_code;      ///< Source code context
    std::optional<program_counter_info_t> program_counter;  ///< Code location
    std::optional<address_range_info_t>
        address_range;  ///< Code object for program_counter
};

using line_info_entry_ptr_t  = std::shared_ptr<line_info_entry_t>;
using line_info_entry_list_t = std::vector<line_info_entry_ptr_t>;

/***
 * @brief Collection of line info entries for an event.
 * @note An event may have multiple source contexts (e.g., inlined functions,
 * multiple relevant source locations). Each entry provides source code and
 * location information. Maps to multiple rocpd_line_info rows in schema v4.
 */
using source_context_list_t = std::vector<line_info_entry_t>;

// --------------------- Data Tables ---------------------

/***
 * @brief Function argument data for API tracing.
 */
struct arg_data_t
{
    size_t      position{};  ///< Argument position (0-indexed)
    const char* type{};      ///< Argument type name
    const char* name{};      ///< Argument parameter name
    const char* value{};     ///< Serialized argument value
    const char* extdata = "{}";
};

using arg_data_ptr_t  = std::shared_ptr<arg_data_t>;
using arg_data_list_t = std::vector<arg_data_ptr_t>;

/***
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
    size_t stack_id;         ///< Unique identifier for this call stack instance
    size_t parent_stack_id;  ///< Parent stack ID for nested events
    size_t correlation_id;   ///< Correlation ID linking related events

    call_stack_t                   call_stack;      ///< Call stack at event time
    std::vector<line_info_entry_t> line_info_list;  ///< Source context information

    const char* event_category;  ///< Event category name (e.g., "HIP_API", "HSA_API")
    const char* extdata = "{}";
};

using event_data_ptr_t  = std::shared_ptr<event_data_t>;
using event_data_list_t = std::vector<event_data_ptr_t>;

/***
 * @brief A named time region representing a span of execution.
 * @note Maps to rocpd_region table. Represents user-annotated regions, API calls,
 * or any named time span. In schema v4, timestamps are stored in rocpd_timestamp
 * table and referenced by ID.
 */
struct region_data_t
{
    std::optional<event_data_t> event;  ///< Common event metadata

    timestamp_ns_t start_timestamp;  ///< Region start time (nanoseconds)
    timestamp_ns_t end_timestamp;    ///< Region end time (nanoseconds)
    const char*    name;             ///< Region name (e.g., function name, annotation)
    const char*    extdata = "{}";

    std::vector<arg_data_t> args;  ///< Optional function arguments
};

using region_data_ptr_t  = std::shared_ptr<region_data_t>;
using region_data_list_t = std::vector<region_data_ptr_t>;

/***
 * @brief A point-in-time sample (instantaneous event).
 * @note Maps to rocpd_sample table. Used for counter samples, markers, or any
 * instantaneous event. Associated with a track for timeline visualization.
 * In schema v4, timestamp is stored in rocpd_timestamp table.
 */
struct sample_data_t
{
    timestamp_ns_t timestamp{};  ///< Sample time (nanoseconds)
    track_info_t   track;
    const char*    extdata = "{}";
};

using sample_data_ptr_t  = std::shared_ptr<sample_data_t>;
using sample_data_list_t = std::vector<sample_data_ptr_t>;

/***
 * @brief Performance counter (PMC) event data.
 * @note Maps to rocpd_pmc_event table. Records a hardware performance counter
 * sample with its value. The sample provides the timestamp, and the event
 * provides correlation and context information.
 */
struct pmc_event_data_t
{
    std::optional<event_data_t> event;    ///< Common event metadata
    double                      value{};  ///< Counter value
    const char*                 extdata = "{}";
    sample_data_t               sample;  ///< Timestamp information
};

using pmc_event_data_ptr_t  = std::shared_ptr<pmc_event_data_t>;
using pmc_event_data_list_t = std::vector<pmc_event_data_ptr_t>;

/***
 * @brief GPU kernel dispatch event data.
 * @note Maps to rocpd_kernel_dispatch table. Records a GPU kernel execution
 * including launch configuration (grid/workgroup sizes), timing, and kernel
 * identification. In schema v4, timestamps are stored in rocpd_timestamp table
 * and context (agent, queue, stream) is stored via track_id.
 */
struct kernel_dispatch_data_t
{
    std::optional<event_data_t> event;               ///< Common event metadata
    size_t                      dispatch_id{};       ///< Unique dispatch identifier
    timestamp_ns_t              start_timestamp{};   ///< Kernel start time (nanoseconds)
    timestamp_ns_t              end_timestamp{};     ///< Kernel end time (nanoseconds)
    kernel_symbol_id_t          kernel_symbol_id{};  ///< Kernel symbol id
    code_object_id_t            code_object_id{};    ///< Code object id
    size_t      private_segment_size{};  ///< Private memory per work-item (bytes)
    size_t      group_segment_size{};    ///< LDS memory per workgroup (bytes)
    size_t      workgroup_size_x{};      ///< Workgroup size in X dimension
    size_t      workgroup_size_y{};      ///< Workgroup size in Y dimension
    size_t      workgroup_size_z{};      ///< Workgroup size in Z dimension
    size_t      grid_size_x{};           ///< Grid size in X dimension
    size_t      grid_size_y{};           ///< Grid size in Y dimension
    size_t      grid_size_z{};           ///< Grid size in Z dimension
    const char* name{};                  ///< Kernel name
    const char* extdata = "{}";
};

using kernel_dispatch_data_ptr_t  = std::shared_ptr<kernel_dispatch_data_t>;
using kernel_dispatch_data_list_t = std::vector<kernel_dispatch_data_ptr_t>;

/***
 * @brief Memory copy operation event data.
 * @note Maps to rocpd_memory_copy table. Records a memory transfer operation
 * including source/destination addresses, size, and timing. Used for tracking
 * host-to-device, device-to-host, and device-to-device copies. In schema v4,
 * timestamps are stored in rocpd_timestamp table.
 */
struct memory_copy_data_t
{
    std::optional<event_data_t> event;               ///< Common event metadata
    timestamp_ns_t              start_timestamp{};   ///< Copy start time (nanoseconds)
    timestamp_ns_t              end_timestamp{};     ///< Copy end time (nanoseconds)
    std::optional<agent_unique_id_t> dst_agent_id;   ///< Destination agent id
    std::optional<size_t>            dst_address;    ///< Destination memory address
    std::optional<agent_unique_id_t> src_agent_id;   ///< Source agent id
    std::optional<size_t>            src_address;    ///< Source memory address
    size_t                           size{};         ///< Transfer size (bytes)
    const char*                      name{};         ///< Operation name
    const char*                      region_name{};  ///< Region name
    const char*                      extdata = "{}";
};

using memory_copy_data_ptr_t  = std::shared_ptr<memory_copy_data_t>;
using memory_copy_data_list_t = std::vector<memory_copy_data_ptr_t>;

/***
 * @brief Memory allocation event data.
 * @note Maps to rocpd_memory_allocate table. Records memory allocation and
 * deallocation operations including address, size, allocation type, and timing.
 * In schema v4, timestamps are stored in rocpd_timestamp table.
 */
struct memory_alloc_data_t
{
    std::optional<event_data_t> event;  ///< Common event metadata
    const char*    type{};   ///< Allocation type (e.g., "hipMalloc", "hipHostMalloc")
    const char*    level{};  ///< Memory level (e.g., "device", "host", "managed")
    timestamp_ns_t start_timestamp{};  ///< Allocation start time (nanoseconds)
    timestamp_ns_t end_timestamp{};    ///< Allocation end time (nanoseconds)
    std::optional<size_t> address;     ///< Allocated memory address
    size_t                size{};      ///< Allocation size (bytes)
    const char*           extdata = "{}";
};

using memory_alloc_data_ptr_t  = std::shared_ptr<memory_alloc_data_t>;
using memory_alloc_data_list_t = std::vector<memory_alloc_data_ptr_t>;

// ============================================================================
// Types for reader API
// ============================================================================

/// Fundamental event kind - determines timeline rendering
enum class event_kind_t
{
    region,  ///< Has start and end, displayed as bar/span
    instant  ///< Single point in time, displayed as marker/dot
};

/// Specific event type - determines which data object to fetch
enum class event_type_t
{
    region,           ///< API calls, user markers
    kernel_dispatch,  ///< GPU kernel execution
    memory_copy,      ///< Memory transfer
    memory_allocate,  ///< Memory operation
    sample,           ///< Instantaneous sample
    pmc_event         ///< Performance counter
};

struct unique_event_id_t
{
    size_t       id;
    event_type_t type;
};

// ============================================================================
// Timeline Event - Lightweight View for Display
// ============================================================================

/// Lightweight event representation for timeline display.
/// Contains only display-necessary fields read from initial query.
/// Full details fetched on-demand via get_event_details() or type-specific accessors.
struct timeline_event_t
{
    // Identity (for fetching full details)
    unique_event_id_t unique_id;

    event_kind_t kind;  ///< region vs instant (for rendering)

    timestamp_ns_t start_timestamp;  ///< Start time (or timestamp for instant)
    timestamp_ns_t end_timestamp;    ///< End time (== start for instant)

    std::string display_name;  ///< Human-readable name
    std::string category;      ///< Event category (e.g., "HIP_API", "GPU")

    track_info_ptr_t track;  ///< Associated track

    std::optional<double> value;

    [[nodiscard]] timestamp_ns_t duration() const noexcept
    {
        return end_timestamp - start_timestamp;
    }

    [[nodiscard]] bool is_instant() const noexcept
    {
        return kind == event_kind_t::instant;
    }
    [[nodiscard]] bool is_region() const noexcept { return kind == event_kind_t::region; }
};

using timeline_event_list_t = std::vector<timeline_event_t>;

// ============================================================================
// Filter Types
// ============================================================================

/// Time window filter for queries
struct time_window_t
{
    std::optional<timestamp_ns_t> start{ std::nullopt };  ///< Filter: start >= this
    std::optional<timestamp_ns_t> end{ std::nullopt };    ///< Filter: end <= this
};

/// Pagination for large result sets (extension point for chunking)
struct pagination_t
{
    std::optional<size_t> limit{ std::nullopt };   ///< Max events to return
    std::optional<size_t> offset{ std::nullopt };  ///< Skip first N events
};

/// Sort order
enum class sort_order_t
{
    ascending,
    descending
};

struct sort_t
{
    std::string  property  = "start";  ///< Property to sort by
    sort_order_t direction = sort_order_t::ascending;
};

/// Combined filter for event queries
struct event_filter_t
{
    time_window_t         time_window;           ///< Time range filter
    pagination_t          pagination;            ///< Limit/offset for chunking
    std::optional<sort_t> sort{ std::nullopt };  ///< Sort order

    /// Which event types to include (empty = all)
    std::vector<event_type_t> types;

    /// Optional WHERE filter
    std::optional<std::string> where;
};

// ============================================================================
// Summary Statistics
// ============================================================================

/// Aggregated statistics for a group of events
struct event_summary_t
{
    std::string    name;
    size_t         count;
    timestamp_ns_t total_duration;
    timestamp_ns_t avg_duration;
    timestamp_ns_t min_duration;
    timestamp_ns_t max_duration;
};

using event_summary_list_t = std::vector<event_summary_t>;

/// Total counts of each event type
struct event_counts_t
{
    size_t regions;
    size_t kernel_dispatches;
    size_t memory_copies;
    size_t memory_allocations;
    size_t samples;
    size_t pmc_events;
};

}  // namespace rocstorage::data_types
