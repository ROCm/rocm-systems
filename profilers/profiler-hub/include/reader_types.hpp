// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <profiler-hub/shared_types.hpp>

namespace profiler_hub::reader_types
{

using timestamp_ns_t = size_t;

enum class event_kind_t
{
    region,  ///< Has start and end, displayed as bar/span
    instant  ///< Single point in time, displayed as marker/dot
};

enum class event_type_t
{
    region,
    kernel_dispatch,
    memory_copy,
    memory_allocate,
    sample,
    pmc_event
};

struct time_window_t
{
    std::optional<timestamp_ns_t> start{ std::nullopt };  ///< Filter: start >= this
    std::optional<timestamp_ns_t> end{ std::nullopt };    ///< Filter: end <= this
};

struct pagination_t
{
    std::optional<size_t> limit{ std::nullopt };   ///< Max events to return
    std::optional<size_t> offset{ std::nullopt };  ///< Skip first N events
};

enum class sort_order_t
{
    ascending,
    descending
};

/**
 * @brief Property to sort timeline events by.
 *
 * Typed enum prevents ORDER BY injection that a raw string would permit.
 */
enum class sort_property_t
{
    start,     ///< Sort by start timestamp
    end,       ///< Sort by end timestamp
    duration,  ///< Sort by (end - start)
    name,      ///< Sort by display name
};

struct sort_t
{
    sort_property_t property  = sort_property_t::start;
    sort_order_t    direction = sort_order_t::ascending;
};

struct event_filter_t
{
    time_window_t         time_window;           ///< Time range filter
    pagination_t          pagination;            ///< Limit/offset for chunking
    std::optional<sort_t> sort{ std::nullopt };  ///< Sort order

    /// Which event types to include (empty = all)
    std::vector<event_type_t> types;
};

struct event_summary_t
{
    std::string    name;
    size_t         count{};
    timestamp_ns_t total_duration{};
    timestamp_ns_t avg_duration{};
    timestamp_ns_t min_duration{};
    timestamp_ns_t max_duration{};
};

using event_summary_list_t = std::vector<event_summary_t>;

using event_counts_t = std::unordered_map<event_type_t, size_t>;

// --------------------- Info Tables ---------------------

struct node_info_t
{
    size_t      node_id;
    size_t      hash;
    std::string machine_id;
    std::string system_name;
    std::string hostname;
    std::string release;
    std::string version;
    std::string hardware_name;
    std::string domain_name;
};

using node_info_ptr_t  = std::shared_ptr<node_info_t>;
using node_info_list_t = std::vector<node_info_ptr_t>;

struct process_info_t
{
    std::optional<size_t> ppid{};
    size_t                pid{};
    std::optional<size_t> init{};
    std::optional<size_t> fini{};
    std::optional<size_t> start{};
    std::optional<size_t> end{};
    std::string           command{};
    std::string           environment{};
    std::string           extdata{};

    std::shared_ptr<node_info_t> node_info;
};

using process_info_ptr_t  = std::shared_ptr<process_info_t>;
using process_info_list_t = std::vector<process_info_ptr_t>;

struct agent_info_t
{
    std::string           agent_type;
    size_t                type_index;
    std::optional<size_t> absolute_index{};
    std::optional<size_t> logical_index{};
    std::optional<size_t> uuid;
    std::string           name;
    std::string           model_name;
    std::string           vendor_name;
    std::string           product_name;
    std::string           user_name;
    std::string           extdata{};

    std::shared_ptr<node_info_t>    node_info;
    std::shared_ptr<process_info_t> process_info;
};

using agent_info_ptr_t  = std::shared_ptr<agent_info_t>;
using agent_info_list_t = std::vector<agent_info_ptr_t>;

struct pmc_info_t
{
    std::string                   name{};
    std::shared_ptr<agent_info_t> agent_info;

    std::string           target_arch{};
    std::optional<size_t> event_code{};
    std::optional<size_t> instance_id{};
    std::string           symbol{};
    std::string           description{};
    std::string           long_description{};
    std::string           component{};
    std::string           units{};
    std::string           value_type{};
    std::string           block{};
    std::string           expression{};
    std::optional<size_t> is_constant{};
    std::optional<size_t> is_derived{};
    std::string           extdata{};

    std::shared_ptr<node_info_t>    node_info;
    std::shared_ptr<process_info_t> process_info;
};

using pmc_info_ptr_t  = std::shared_ptr<pmc_info_t>;
using pmc_info_list_t = std::vector<pmc_info_ptr_t>;

struct thread_info_t
{
    std::optional<size_t> parent_process_id{};
    size_t                thread_id{};
    std::string           name{};
    std::optional<size_t> start{};
    std::optional<size_t> end{};
    std::string           extdata{};

    std::shared_ptr<node_info_t>    node_info;
    std::shared_ptr<process_info_t> process_info;
};

using thread_info_ptr_t  = std::shared_ptr<thread_info_t>;
using thread_info_list_t = std::vector<thread_info_ptr_t>;

struct stream_info_t
{
    size_t      stream_id{};
    std::string name{};
    std::string extdata{};

    std::shared_ptr<node_info_t>    node_info;
    std::shared_ptr<process_info_t> process_info;
};

using stream_info_ptr_t  = std::shared_ptr<stream_info_t>;
using stream_info_list_t = std::vector<stream_info_ptr_t>;

struct queue_info_t
{
    size_t      queue_id{};
    std::string name{};
    std::string extdata{};

    std::shared_ptr<node_info_t>    node_info;
    std::shared_ptr<process_info_t> process_info;
};

using queue_info_ptr_t  = std::shared_ptr<queue_info_t>;
using queue_info_list_t = std::vector<queue_info_ptr_t>;

struct code_object_info_t
{
    size_t                id{};
    std::string           uri{};
    std::optional<size_t> load_base{};
    std::optional<size_t> load_size{};
    std::optional<size_t> load_delta{};
    std::string           storage_type{};
    std::string           extdata{};

    std::shared_ptr<node_info_t>    node_info;
    std::shared_ptr<process_info_t> process_info;
    std::shared_ptr<agent_info_t>   agent_info;
};

using code_object_info_ptr_t  = std::shared_ptr<code_object_info_t>;
using code_object_info_list_t = std::vector<code_object_info_ptr_t>;

struct kernel_symbol_info_t
{
    size_t                id{};
    std::string           name{};
    std::string           display_name{};
    std::optional<size_t> kernel_object{};
    std::optional<size_t> kernarg_segment_size{};
    std::optional<size_t> kernarg_segment_alignment{};
    std::optional<size_t> group_segment_size{};
    std::optional<size_t> private_segment_size{};
    std::optional<size_t> sgpr_count{};
    std::optional<size_t> arch_vgpr_count{};
    std::optional<size_t> accum_vgpr_count{};
    std::string           extdata{};

    std::shared_ptr<node_info_t>        node_info;
    std::shared_ptr<process_info_t>     process_info;
    std::shared_ptr<code_object_info_t> code_object_info;
};

using kernel_symbol_info_ptr_t  = std::shared_ptr<kernel_symbol_info_t>;
using kernel_symbol_info_list_t = std::vector<kernel_symbol_info_ptr_t>;

/**
 * @brief Classifies a track by the identity shape of its contents.
 *
 * Determines which identity shared_ptrs in track_info_t are populated, whether
 * the caller calls get_interval_track() or get_scalar_track(), and which
 * get_*_details() method applies to opaque_ids drawn from that track.
 */
enum class track_type_t
{
    cpu_thread,  ///< thread_info populated. Interval track of region events.
    gpu_queue,   ///< agent_info + queue_info populated. Interval track of kernel
                 ///< dispatches.
    dma,         ///< agent_info + stream_info populated. Interval track of memory copies
          ///< only, keyed (nid, pid, queue_id, stream_id) — a single event table.
    counter,  ///< thread_info + (optional) agent_info. Scalar track of counter samples.
    stream    ///< stream_info populated. Interval track that AGGREGATES three event
            ///< tables — kernel_dispatch + memory_copy + memory_allocate — that share a
            ///< stream, keyed (nid, pid, stream_id). Unlike dma (memory-copy only), a
            ///< stream track's events span multiple per-type tables, so each returned
            ///< interval_event_t carries op_kind to select the get_*_details() overload.
};

/**
 * @brief For v3 synthesized cpu_thread (region) tracks, distinguishes whether the
 *        track carries region events that have an associated rocpd_sample.
 *
 * v3 region tracks are synthesized from rocpd_region (there is no reliable
 * rocpd_track registry for them). A single (nid, pid, tid) thread can produce two
 * tracks: one of regions with no sample (main) and one of regions that do have a
 * sample (sample) — mirroring roc-optiq's region-main / region-sample split. All
 * other track types (and every v4.0 track) are @ref region_track_kind_t::none.
 */
enum class region_track_kind_t
{
    none,    ///< Not a v3 synthesized region track.
    main,    ///< Region events with no associated rocpd_sample row.
    sample,  ///< Region events that have an associated rocpd_sample row.
};

struct track_info_t
{
    size_t
        id{};  ///< Track identifier. Pass to get_interval_track()/get_scalar_track().
               ///< Opaque and stable for the reader's lifetime; never a topology tuple.
    track_type_t type{};  ///< Determines which identity fields below are populated and
                          ///< which event-fetch / detail method applies.
    region_track_kind_t region_kind{
        region_track_kind_t::none
    };  ///< v3 cpu_thread region tracks only: main vs.
        ///< sample split. none for all other tracks.
    std::string name{};
    std::string extdata{};

    std::shared_ptr<node_info_t>    node_info;     ///< Always populated.
    std::shared_ptr<process_info_t> process_info;  ///< Always populated.
    std::shared_ptr<thread_info_t>  thread_info;   ///< cpu_thread, counter.
    std::shared_ptr<agent_info_t>   agent_info;   ///< gpu_queue, dma; optionally counter.
    std::shared_ptr<queue_info_t>   queue_info;   ///< gpu_queue.
    std::shared_ptr<stream_info_t>  stream_info;  ///< dma, stream.
};

using track_info_ptr_t  = std::shared_ptr<track_info_t>;
using track_info_list_t = std::vector<track_info_ptr_t>;

// --------------------- Call Stack & Source Context (string-owning) -------

struct address_range_info_t
{
    size_t      address_base{};
    size_t      address_low{};
    size_t      address_high{};
    std::string extdata = "{}";
};

struct program_counter_info_t
{
    std::string           function;
    std::string           filename;
    std::optional<size_t> line_number;
    std::string           extdata = "{}";
};

struct stack_frame_t
{
    std::optional<program_counter_info_t> program_counter;
    std::optional<address_range_info_t>   address_range;
    std::string                           extdata = "{}";
};

using call_stack_t = std::deque<stack_frame_t>;

struct source_code_info_t
{
    std::optional<std::string> filename;
    std::optional<size_t>      starting_line_number;
    std::vector<std::string>   source_code_lines;
    std::vector<std::string>   assembly_instruction_lines;
    std::string                extdata = "{}";
};

struct line_info_entry_t
{
    std::optional<source_code_info_t>     source_code;
    std::optional<program_counter_info_t> program_counter;
    std::optional<address_range_info_t>   address_range;
};

using source_context_list_t = std::vector<line_info_entry_t>;

// --------------------- Data Tables ---------------------

struct arg_data_t
{
    size_t      position{};  ///< Argument position (0-indexed)
    std::string type;        ///< Argument type name
    std::string name;        ///< Argument parameter name
    std::string value;       ///< Serialized argument value
    std::string extdata;
};

using arg_data_ptr_t  = std::shared_ptr<arg_data_t>;
using arg_data_list_t = std::vector<arg_data_ptr_t>;

struct event_data_t
{
    size_t stack_id;         ///< Unique identifier for this call stack instance
    size_t parent_stack_id;  ///< Parent stack ID for nested events
    size_t correlation_id;   ///< Correlation ID linking related events

    call_stack_t          call_stack;      ///< Call stack at event time
    source_context_list_t line_info_list;  ///< Source context information

    std::string event_category;  ///< Event category name (e.g., "HIP_API", "HSA_API")
    std::string extdata;
};

using event_data_ptr_t  = std::shared_ptr<event_data_t>;
using event_data_list_t = std::vector<event_data_ptr_t>;

struct region_data_t
{
    std::shared_ptr<event_data_t> event;  ///< Common event metadata

    timestamp_ns_t start_timestamp;  ///< Region start time (nanoseconds)
    timestamp_ns_t end_timestamp;    ///< Region end time (nanoseconds)
    std::string    name;             ///< Region name (e.g., function name, annotation)
    std::string    extdata;

    std::vector<arg_data_t> args;  ///< Optional function arguments
};

using region_data_ptr_t  = std::shared_ptr<region_data_t>;
using region_data_list_t = std::vector<region_data_ptr_t>;

struct sample_data_t
{
    timestamp_ns_t                timestamp{};  ///< Sample time (nanoseconds)
    std::shared_ptr<track_info_t> track;
    std::string                   extdata;
};

using sample_data_ptr_t  = std::shared_ptr<sample_data_t>;
using sample_data_list_t = std::vector<sample_data_ptr_t>;

struct pmc_event_data_t
{
    std::shared_ptr<event_data_t> event;    ///< Common event metadata
    double                        value{};  ///< Counter value
    std::string                   extdata;
    sample_data_t                 sample;  ///< Timestamp information
};

using pmc_event_data_ptr_t  = std::shared_ptr<pmc_event_data_t>;
using pmc_event_data_list_t = std::vector<pmc_event_data_ptr_t>;

struct kernel_dispatch_data_t
{
    size_t         dispatch_id{};      ///< Unique dispatch identifier
    timestamp_ns_t start_timestamp{};  ///< Kernel start time (nanoseconds)
    timestamp_ns_t end_timestamp{};    ///< Kernel end time (nanoseconds)

    std::optional<size_t>
        private_segment_size{};                  ///< Private memory per work-item (bytes)
    std::optional<size_t> group_segment_size{};  ///< LDS memory per workgroup (bytes)
    size_t                workgroup_size_x{};    ///< Workgroup size in X dimension
    size_t                workgroup_size_y{};    ///< Workgroup size in Y dimension
    size_t                workgroup_size_z{};    ///< Workgroup size in Z dimension
    size_t                grid_size_x{};         ///< Grid size in X dimension
    size_t                grid_size_y{};         ///< Grid size in Y dimension
    size_t                grid_size_z{};         ///< Grid size in Z dimension
    std::string           name;                  ///< Kernel name
    std::string           extdata;

    event_data_ptr_t         event;
    node_info_ptr_t          node_info;
    process_info_ptr_t       process_info;
    thread_info_ptr_t        thread_info;
    agent_info_ptr_t         agent_info;
    kernel_symbol_info_ptr_t kernel_symbol_info;  ///< Kernel symbol id
    code_object_info_ptr_t   code_object_info;    ///< Code object id
    stream_info_ptr_t        stream_info;
    queue_info_ptr_t         queue_info;
};

using kernel_dispatch_data_ptr_t  = std::shared_ptr<kernel_dispatch_data_t>;
using kernel_dispatch_data_list_t = std::vector<kernel_dispatch_data_ptr_t>;

struct memory_copy_data_t
{
    timestamp_ns_t        start_timestamp{};  ///< Copy start time (nanoseconds)
    timestamp_ns_t        end_timestamp{};    ///< Copy end time (nanoseconds)
    std::optional<size_t> dst_address;        ///< Destination memory address
    std::optional<size_t> src_address;        ///< Source memory address
    size_t                size;               ///< Transfer size (bytes)
    std::string           name;               ///< Operation name
    std::string           region_name;        ///< Region name
    std::string           extdata;

    event_data_ptr_t event;         ///< Common event metadata
    agent_info_ptr_t dst_agent_id;  ///< Destination agent id
    agent_info_ptr_t src_agent_id;  ///< Source agent id

    node_info_ptr_t    node_info;
    process_info_ptr_t process_info;
    thread_info_ptr_t  thread_info;

    stream_info_ptr_t stream_info;
    queue_info_ptr_t  queue_info;
};

using memory_copy_data_ptr_t  = std::shared_ptr<memory_copy_data_t>;
using memory_copy_data_list_t = std::vector<memory_copy_data_ptr_t>;

struct memory_alloc_data_t
{
    std::string           type;  ///< Allocation type (e.g., "hipMalloc", "hipHostMalloc")
    std::string           level;  ///< Memory level (e.g., "device", "host", "managed")
    timestamp_ns_t        start_timestamp{};  ///< Allocation start time (nanoseconds)
    timestamp_ns_t        end_timestamp{};    ///< Allocation end time (nanoseconds)
    std::optional<size_t> address;            ///< Allocated memory address
    size_t                size;               ///< Allocation size (bytes)
    std::string           extdata;

    event_data_ptr_t event;  ///< Common event metadata

    node_info_ptr_t    node_info;
    process_info_ptr_t process_info;
    thread_info_ptr_t  thread_info;
    agent_info_ptr_t   agent_info;
    stream_info_ptr_t  stream_info;
    queue_info_ptr_t   queue_info;
};

using memory_alloc_data_ptr_t  = std::shared_ptr<memory_alloc_data_t>;
using memory_alloc_data_list_t = std::vector<memory_alloc_data_ptr_t>;

struct unique_timeline_event_id_t
{
    size_t       id;
    event_type_t type;
};

struct timeline_event_t
{
    unique_timeline_event_id_t unique_identifier;

    timestamp_ns_t start_timestamp;
    timestamp_ns_t end_timestamp;

    std::string display_name;
    std::string category;

    track_info_ptr_t track;
};

using timeline_event_list_t = std::vector<timeline_event_t>;

struct counter_timeline_event_t
{
    unique_timeline_event_id_t unique_identifier;

    timestamp_ns_t timestamp;
    size_t         value;

    track_info_ptr_t track;
};

using counter_timeline_event_list_t = std::vector<counter_timeline_event_t>;

// --------------------- Track event types (track-scoped queries) ----------

struct interval_event_t
{
    size_t opaque_id{};  ///< SQLite row id of the event in its per-type table
                         ///< (region / kernel_dispatch / memory_copy). Pass to the
                         ///< get_*_details() method selected by the track's type, or by
                         ///< op_kind below for stream tracks.
    timestamp_ns_t start{};  ///< Event start (nanoseconds).
    timestamp_ns_t end{};    ///< Event end (nanoseconds).
    std::string    display_name;  ///< Human-readable label for the bar.
    std::string    category;      ///< Event category display string (e.g. "rocm_hip_api",
                           ///< "timer_sampling"); empty when the event carries none.
    std::optional<event_type_t>
        op_kind{};  ///< Which per-type table this event's opaque_id
                    ///< belongs to, and thus which get_*_details() overload
                    ///< applies. nullopt for single-table tracks (use the track's
                    ///< type); populated for stream tracks, whose events span
                    ///< kernel_dispatch / memory_copy / memory_allocate.
    int level{};  ///< Nesting depth; 0 = outermost. Computed in-reader.
    std::optional<size_t>
        parent_id{};  ///< opaque_id of the enclosing event when truly nested;
                      ///< nullopt when overlapping-but-not-nested.
};

using interval_event_list_t = std::vector<interval_event_t>;

struct scalar_event_t
{
    size_t         opaque_id{};  ///< rocpd_sample.id; pass to get_scalar_details().
    timestamp_ns_t timestamp{};  ///< Sample time (nanoseconds).
    double         value{};      ///< Counter value (REAL).
};

using scalar_event_list_t = std::vector<scalar_event_t>;

struct flow_t
{
    size_t source_opaque_id{};  ///< opaque_id of the originating event (e.g. CPU region).
    size_t dest_opaque_id{};    ///< opaque_id of the destination event (e.g. GPU kernel
                                ///< dispatch).
};

using flow_list_t = std::vector<flow_t>;

struct track_stats_t
{
    std::optional<timestamp_ns_t>
        min_ts;  ///< Earliest start on the track; nullopt if empty.
    std::optional<timestamp_ns_t> max_ts;  ///< Latest end (samples: latest timestamp) on
                                           ///< the track; nullopt if empty.
    size_t count{};  ///< Number of events (or samples) on the track.
};

}  // namespace profiler_hub::reader_types
