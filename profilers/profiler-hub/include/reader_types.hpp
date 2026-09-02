// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <profiler-hub/shared_types.hpp>

namespace profiler_hub::reader_types
{

using timestamp_t = size_t;

/// Opaque track identifier. Treat as opaque: the only portable operations are equality,
/// ordering, hashing (so it can key a map), and reading the public `value` field to
/// serialize/reconstruct it. The integer is a ProfilerHub-private DB identity; do not
/// synthesize or do arithmetic on it. The underlying integer is size_t, not uint32_t,
/// so real DB ids cannot truncate and the consumer's SIZE_MAX invalid sentinel survives
/// round-trips. Unlike event_id_t/flow_id_t (which fully hide their value), a track id
/// is a stable DB identity the consumer must serialize, so the integer stays publicly
/// reachable.
struct track_id_t
{
    size_t value{};
    bool   operator==(const track_id_t& o) const noexcept { return value == o.value; }
    bool   operator!=(const track_id_t& o) const noexcept { return value != o.value; }
    bool   operator<(const track_id_t& o) const noexcept { return value < o.value; }
};

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

/// Visible time window for track/flow reads. Interval and flow reads keep any event that
/// OVERLAPS [start, end] (boundary-inclusive); an event is dropped only when it lies
/// entirely outside. Point reads (scalar tracks) keep events whose timestamp falls within
/// [start, end]. An unset bound is open-ended; an empty {} window applies no filter.
struct time_window_t
{
    std::optional<timestamp_t> start{
        std::nullopt
    };  ///< Window lower bound (ns); unset = open
    std::optional<timestamp_t> end{
        std::nullopt
    };  ///< Window upper bound (ns); unset = open
};

// Track reads chunk with pagination_t{limit, offset} only — a straight window into the
// ordered event list. There is no level-of-detail decimation on track reads; flow-edge
// decimation is available as max_edges on get_flows_in_window (see reader.hpp).
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
    std::string name;
    size_t      count{};
    timestamp_t total_duration{};
    timestamp_t avg_duration{};
    timestamp_t min_duration{};
    timestamp_t max_duration{};
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
    size_t                id{};
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
    size_t      pmc_id{};  ///< numeric PMC id (rocpd_info_pmc primary key).
    std::string name{};
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

    // True when more than one rocpd_pmc_event row shares the same (event_id, pmc_id) —
    // a legacy producer bug where two distinct physical quantities were written under one
    // pmc_id. Consumers should treat values for this pmc as unreliable / interleaved.
    bool ambiguous{ false };

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
 * get_*_details() method applies to event handles drawn from that track.
 */
enum class track_type_t
{
    cpu_thread,  ///< thread_info populated. Interval track of region events.
    gpu_queue,   ///< agent_info + queue_info populated. Interval track of kernel
                 ///< dispatches.
    dma,         ///< agent_info populated (stream_info nullopt). Interval track of memory
          ///< copies only, keyed (nid, pid, queue_id, dst_agent_id) by destination agent
          ///< to match Optiq's memory-copy swimlanes — a single event table. Stream-level
          ///< grouping of memory copies lives on the `stream` track type instead.
    counter,  ///< thread_info + pmc_info + (optional) agent_info. Scalar track of counter
              ///< samples. pmc_info carries the full PMC metadata panel (name, symbol,
              ///< description, units, block, expression, …) keyed by the real pmc_id.
    stream,   ///< stream_info populated. Interval track that AGGREGATES three event
             ///< tables — kernel_dispatch + memory_copy + memory_allocate — that share a
             ///< stream, keyed (nid, pid, stream_id). Unlike dma (memory-copy only), a
             ///< stream track's events span multiple per-type tables; each returned
             ///< interval_entry_t::id encodes its event type so the reader routes it to
             ///< the correct get_*_details() overload with no companion tag.
    memory,  ///< agent_info + queue_info populated. Interval track of memory-allocate
             ///< events (rocpd_memory_allocate), keyed (nid, agent_id, queue_id, pid) to
             ///< match Optiq's GetRocprofMemoryAllocTrackQuery GROUP BY exactly. Both
             ///< agent_id and queue_id are nullable; NULL is preserved as a distinct
             ///< group value, not dropped. Distinct from the `dma` (memory-copy) and
             ///< `stream` (cross-table aggregate) track types.
    kernel_dispatch_pmc,  ///< agent_info populated. Interval track of kernel-dispatch PMC
                          ///< events (rocpd_pmc_event JOIN rocpd_kernel_dispatch), keyed
                          ///< (nid, agent_id, pmc_id, pid) to match Optiq's
                          ///< GetRocprofPerformanceCountersTrackQuery GROUP BY exactly.
                          ///< Distinct from the `counter` (SMI sample-based) track type.
                          ///< Use get_interval_track(); scalar read returns empty.
    memory_activity       ///< agent_info populated. Scalar track of cumulative
                          ///< bytes-allocated per agent over time, keyed
                          ///< (nid, pid, agent_id). Computed from
                          ///< rocpd_memory_allocate: ALLOC adds size, FREE
                          ///< subtracts size (agent_id/size recovered from prior
                          ///< ALLOC via address map), REALLOC/RECLAIM are no-op.
                          ///< Mirrors Optiq GetRocprofMemoryActivity* (load_id 7).
                          ///< Use get_scalar_track(); interval read returns empty.
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

/**
 * @brief How a track's overlapping intervals should be interpreted vertically.
 *
 * Splits the historically overloaded interval `level` into two concepts: a
 * containment `parent` edge (valid only when a track is a genuine synchronous call
 * stack) and a geometric packing `lane` (always valid). @ref nesting_model_t is the
 * per-track metadata that tells a renderer which applies.
 */
enum class nesting_model_t
{
    stack,  ///< Overlaps are true containment: interval_entry_t::parent is populated and
            ///< `lane` coincides with call depth on real (non-overlapping-sibling) data.
    lane,   ///< Overlaps are concurrency: interval_entry_t::parent is always no-parent;
            ///< only `lane` (the packing row) is meaningful.
};

struct track_info_t
{
    track_id_t
        id{};  ///< Track identifier. Pass to get_interval_track()/get_scalar_track().
               ///< Opaque and stable for the reader's lifetime; never a topology tuple.
    track_type_t        type{};
    region_track_kind_t region_kind{ region_track_kind_t::none };
    std::string         name{};
    std::string         extdata{};

    // Identity is carried as relational shared_ptr objects
    // (node/process/thread/agent/queue/stream/pmc), sharing one instance across every
    // track that references it. There is no flat scalar-id (node_id/process_id/
    // sub_process_id) or C-ABI form.
    std::shared_ptr<node_info_t>    node_info;     ///< Always populated.
    std::shared_ptr<process_info_t> process_info;  ///< Always populated.
    std::shared_ptr<thread_info_t>  thread_info;   ///< cpu_thread, counter.
    std::shared_ptr<agent_info_t>
        agent_info;  ///< gpu_queue, dma, memory, memory_activity, kernel_dispatch_pmc;
                     ///< optionally counter.
    std::shared_ptr<queue_info_t>  queue_info;   ///< gpu_queue, memory.
    std::shared_ptr<stream_info_t> stream_info;  ///< stream.
    std::shared_ptr<pmc_info_t>    pmc_info;     ///< counter, kernel_dispatch_pmc.

    // nesting_model gates whether interval_entry_t::parent is populated for this track's
    // events (stack only); max_lane exposes the track's peak concurrency so height
    // consumers can migrate off the deprecated interval_entry_t::level.
    nesting_model_t nesting{
        nesting_model_t::lane
    };  ///< stack = containment (parent populated); lane = concurrency (no parent).
    uint32_t max_lane{};  ///< Peak concurrency / stack depth = number of packing lanes.
                          ///< 0 until the track is first read via get_interval_track();
                          ///< scalar-only tracks stay 0.

    /// v4 only. True when this track_id appeared in both the counter (rocpd_sample/pmc)
    /// and memory-allocate (rocpd_memory_allocate) discovery sets — an ambiguous schema
    /// state where classification as counter (current precedence) silently drops the
    /// memory-allocate events. Callers should treat ambiguous_classification==true as a
    /// data-integrity warning. No known real DB triggers this today.
    bool ambiguous_classification{ false };
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

// Detail property values are a typed variant, not stringly. This mirrors the SDK rocpd
// writer's `struct sql_insert_value`
// (rocprofiler-sdk/source/lib/output/generateRocpd.cpp) — the exact read/write seam
// profiler-hub sits opposite — which uses the same alternatives. `monostate` =
// present-but-empty; `nullptr_t` = explicitly-absent optional. Revisit if the SDK value
// model changes.
using arg_value_t =
    std::variant<std::monostate, int64_t, uint64_t, double, std::string, std::nullptr_t>;

struct arg_t
{
    std::string key;    ///< Property name (source struct field / argument name).
    arg_value_t value;  ///< Typed value; see arg_value_t.
};

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

    timestamp_t start_timestamp;  ///< Region start time (nanoseconds)
    timestamp_t end_timestamp;    ///< Region end time (nanoseconds)
    std::string name;             ///< Region name (e.g., function name, annotation)
    std::string extdata;

    std::vector<arg_data_t> args;  ///< Optional function arguments
};

using region_data_ptr_t  = std::shared_ptr<region_data_t>;
using region_data_list_t = std::vector<region_data_ptr_t>;

struct sample_data_t
{
    timestamp_t                   timestamp{};  ///< Sample time (nanoseconds)
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
    size_t      dispatch_id{};      ///< Unique dispatch identifier
    timestamp_t start_timestamp{};  ///< Kernel start time (nanoseconds)
    timestamp_t end_timestamp{};    ///< Kernel end time (nanoseconds)

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
    timestamp_t           start_timestamp{};  ///< Copy start time (nanoseconds)
    timestamp_t           end_timestamp{};    ///< Copy end time (nanoseconds)
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
    timestamp_t           start_timestamp{};  ///< Allocation start time (nanoseconds)
    timestamp_t           end_timestamp{};    ///< Allocation end time (nanoseconds)
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

    timestamp_t start_timestamp;
    timestamp_t end_timestamp;

    std::string display_name;
    std::string category;

    track_info_ptr_t track;
};

using timeline_event_list_t = std::vector<timeline_event_t>;

struct counter_timeline_event_t
{
    unique_timeline_event_id_t unique_identifier;

    timestamp_t timestamp;
    size_t      value;

    track_info_ptr_t track;
};

using counter_timeline_event_list_t = std::vector<counter_timeline_event_t>;

// --------------------- Opaque event handle -------------------------------

namespace detail
{
struct event_id_access;
struct flow_id_access;
}  // namespace detail

/**
 * @brief Opaque, ProfilerHub-minted event handle.
 *
 * Uniquely identifies one event within a reader session across every track type
 * and every source table -- no two distinct events ever share a handle. Treat it
 * as opaque: the only supported operations are equality, ordering, and hashing
 * (so it can key a std::map / std::unordered_map). The internal encoding is
 * private and may change; consumers must never depend on it and never need a
 * companion type tag to interpret it. Pass a handle straight back to the
 * get_*_details() accessor of interest; the reader recovers internally which
 * source table and row it names.
 */
class event_id_t
{
public:
    event_id_t() = default;

    bool operator==(const event_id_t& o) const noexcept
    {
        return m_source_db == o.m_source_db && m_type == o.m_type &&
               m_row_id == o.m_row_id;
    }
    bool operator!=(const event_id_t& o) const noexcept { return !(*this == o); }
    bool operator<(const event_id_t& o) const noexcept
    {
        return std::tie(m_source_db, m_type, m_row_id) <
               std::tie(o.m_source_db, o.m_type, o.m_row_id);
    }

private:
    friend struct detail::event_id_access;

    uint32_t m_source_db{};   ///< Source-database discriminator; 0 until multi-DB reads.
    event_type_t m_type{};    ///< Which per-type table m_row_id indexes (routing key).
    size_t       m_row_id{};  ///< Per-type-table row id. Never exposed publicly.
};

namespace detail
{
/// Reader-internal minting/decoding of event_id_t. NOT part of the consumer
/// contract -- the reader implementation uses it to build handles and route
/// detail queries; consumers must treat event_id_t as opaque.
struct event_id_access
{
    static event_id_t make(event_type_t type, size_t row_id, uint32_t source_db = 0)
    {
        event_id_t h;
        h.m_source_db = source_db;
        h.m_type      = type;
        h.m_row_id    = row_id;
        return h;
    }
    static event_type_t type(const event_id_t& h) { return h.m_type; }
    static size_t       row_id(const event_id_t& h) { return h.m_row_id; }
    static uint32_t     source_db(const event_id_t& h) { return h.m_source_db; }
};
}  // namespace detail

/**
 * @brief Unified detail record for any event, keyed by its opaque handle.
 *
 * One collapsed detail path across all six event_type_t cases: a fixed common header
 * plus a generic `properties` bag of named, typed values. Replaces the seven typed
 * get_*_details() accessors. Linked entities (agent, kernel_symbol, code_object, stream,
 * queue, node/process/thread) appear in `properties` as their integer id, NOT as a
 * resolved sub-struct — consumers do a follow-up lookup by id.
 */
struct event_info_t
{
    event_id_t                 id;          ///< Opaque handle this detail describes.
    std::string                name;        ///< Type's name field; empty if none.
    std::string                category;    ///< Event category display string.
    timestamp_t                ts{};        ///< Start / only timestamp (nanoseconds).
    std::optional<timestamp_t> te;          ///< End timestamp; nullopt for point events.
    std::vector<arg_t>         properties;  ///< All non-header fields, typed.
};

// --------------------- Track event types (track-scoped queries) ----------

struct interval_entry_t
{
    event_id_t id{};  ///< Opaque handle for this event. Pass to the get_*_details()
                      ///< accessor of interest; a mismatched accessor returns nullopt.
    timestamp_t start{};       ///< Event start (nanoseconds).
    timestamp_t end{};         ///< Event end (nanoseconds).
    std::string display_name;  ///< Human-readable label for the bar.
    std::string category;      ///< Event category display string (e.g. "rocm_hip_api",
                               ///< "timer_sampling"); empty when the event carries none.
    // `lane` is the geometric packing row and is always valid; `parent_id` is a true
    // containment edge, populated only on `stack` tracks (track_info_t::nesting == stack)
    // and carrying the opaque event_id_t, never a raw row id. `level` is retained for
    // backward compatibility (Optiq reads it for height) — stack tracks: containment
    // depth; lane tracks: == lane. Height consumers should migrate to
    // track_info_t::max_lane.
    int level{};      ///< Deprecated. Nesting depth on stack tracks; == lane on lane
                      ///< tracks. Prefer `lane` (row) + `parent_id` (containment).
    uint32_t lane{};  ///< Geometric packing row so overlapping intervals never collide.
                      ///< Always valid, every interval track. 0 = first row.
    std::optional<event_id_t>
        parent_id{};  ///< Containment parent's opaque handle; populated only on `stack`
                      ///< tracks, and only when this event is truly enclosed. nullopt on
                      ///< lane tracks and for top-level events.
};

using interval_entry_list_t = std::vector<interval_entry_t>;

struct scalar_sample_t
{
    event_id_t  id{};         ///< Opaque handle; pass to get_event_info().
    timestamp_t timestamp{};  ///< Sample time (nanoseconds).
    double      value{};      ///< Counter value (REAL).
};

using scalar_sample_list_t = std::vector<scalar_sample_t>;

// Reversible mapping (see get_flows).
enum class flow_kind_t
{
    launch_to_dispatch,   ///< region -> kernel_dispatch (CPU launch to GPU dispatch).
    copy_submit_to_exec,  ///< region -> memory_copy / memory_allocate (submit to exec).
    stream_dependency,    ///< same-type GPU siblings (kd->kd, mc->mc, ma->ma).
    generic,              ///< region -> region and anything else.
};

/**
 * @brief Opaque handle grouping the edges of one flow / chain.
 *
 * All edges derived from a single causal chain share one flow_id_t; a multi-hop chain
 * A -> B -> C is two edges carrying the same flow_id. Treat it as opaque: the only
 * supported operations are equality, ordering, and hashing (so it can key a map). The
 * internal encoding is private and may change; consumers must never depend on it.
 */
class flow_id_t
{
public:
    flow_id_t() = default;

    bool operator==(const flow_id_t& o) const noexcept { return m_value == o.m_value; }
    bool operator!=(const flow_id_t& o) const noexcept { return m_value != o.m_value; }
    bool operator<(const flow_id_t& o) const noexcept { return m_value < o.m_value; }

private:
    friend struct detail::flow_id_access;

    uint64_t m_value{};  ///< Opaque group key. Never exposed publicly.
};

namespace detail
{
/// Reader-internal minting/decoding of flow_id_t. NOT part of the consumer contract --
/// the reader derives the group key; consumers must treat flow_id_t as opaque.
struct flow_id_access
{
    static flow_id_t make(uint64_t value)
    {
        flow_id_t h;
        h.m_value = value;
        return h;
    }
    static uint64_t value(const flow_id_t& h) { return h.m_value; }
};
}  // namespace detail

// The source/dest field names are kept for backward compatibility.
struct flow_edge_t
{
    event_id_t  source{};   ///< Opaque handle of the source event (arrow tail).
    event_id_t  dest{};     ///< Opaque handle of the destination event (arrow head).
    flow_id_t   flow_id{};  ///< Edges sharing this id form one flow / chain.
    flow_kind_t kind{ flow_kind_t::generic };  ///< Semantic class of the edge.
};

using flow_list_t = std::vector<flow_edge_t>;

struct track_stats_t
{
    std::optional<timestamp_t>
        min_ts;  ///< Earliest start on the track; nullopt if empty.
    std::optional<timestamp_t> max_ts;  ///< Latest end (samples: latest timestamp) on
                                        ///< the track; nullopt if empty.
    size_t count{};                     ///< Number of events (or samples) on the track.
};

}  // namespace profiler_hub::reader_types

namespace std
{
template <>
struct hash<profiler_hub::reader_types::event_id_t>
{
    size_t operator()(const profiler_hub::reader_types::event_id_t& h) const noexcept
    {
        namespace rt = profiler_hub::reader_types;
        size_t seed  = std::hash<uint32_t>{}(rt::detail::event_id_access::source_db(h));
        auto   mix   = [&seed](size_t v) {
            seed ^= v + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        };
        mix(std::hash<int>{}(static_cast<int>(rt::detail::event_id_access::type(h))));
        mix(std::hash<size_t>{}(rt::detail::event_id_access::row_id(h)));
        return seed;
    }
};

template <>
struct hash<profiler_hub::reader_types::flow_id_t>
{
    size_t operator()(const profiler_hub::reader_types::flow_id_t& h) const noexcept
    {
        namespace rt = profiler_hub::reader_types;
        return std::hash<uint64_t>{}(rt::detail::flow_id_access::value(h));
    }
};

template <>
struct hash<profiler_hub::reader_types::track_id_t>
{
    size_t operator()(const profiler_hub::reader_types::track_id_t& h) const noexcept
    {
        return std::hash<size_t>{}(h.value);
    }
};
}  // namespace std
