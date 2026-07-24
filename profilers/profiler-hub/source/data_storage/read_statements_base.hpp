// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/sqlite_backend.hpp"

#include "profiler-hub/reader_types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace profiler_hub::data_storage
{

// ---------------------------------------------------------------------------
// Shared result structs. These describe the row shapes the reader consumes and
// are backend-agnostic: both the v3 and v4.0 read_statements implementations
// populate the same structs (the SQL that fills them differs per schema).
// ---------------------------------------------------------------------------

struct node_info_result
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

struct process_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<size_t>      ppid;
    std::optional<size_t>      init;
    std::optional<size_t>      fini;
    std::optional<size_t>      start;
    std::optional<size_t>      end;
    std::optional<std::string> command;
    std::string                environment;
    std::string                extdata;
};

struct string_result
{
    size_t      id{};
    std::string value;
};

struct stream_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<std::string> name;
    std::string                extdata;
};

struct queue_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<std::string> name;
    std::string                extdata;
};

struct thread_info_result
{
    size_t                     id{};
    size_t                     nid{};
    std::optional<size_t>      ppid;
    size_t                     pid{};
    size_t                     tid{};
    std::optional<std::string> name;
    std::optional<size_t>      start;
    std::optional<size_t>      end;
    std::string                extdata;
};

struct agent_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<std::string> type;
    std::optional<size_t>      absolute_index;
    std::optional<size_t>      logical_index;
    std::optional<size_t>      type_index;
    std::optional<size_t>      uuid;
    std::optional<std::string> name;
    std::optional<std::string> model_name;
    std::optional<std::string> vendor_name;
    std::optional<std::string> product_name;
    std::optional<std::string> user_name;
    std::string                extdata;
};

// Extended with optional agent_id / queue_id / stream_id for v4.0, whose
// rocpd_track is the universal identity anchor (carries these columns). v3's
// track query does not select them, so they stay nullopt on v3.
struct track_info_result
{
    size_t                id{};
    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::optional<size_t> agent_id;
    std::optional<size_t> queue_id;
    std::optional<size_t> stream_id;
    std::optional<size_t> name_id;
    std::string           extdata;
};

struct kernel_symbol_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    size_t                     code_object_id{};
    std::optional<std::string> kernel_name;
    std::optional<std::string> display_name;
    std::optional<size_t>      kernel_object;
    std::optional<size_t>      kernarg_segment_size;
    std::optional<size_t>      kernarg_segment_alignment;
    std::optional<size_t>      group_segment_size;
    std::optional<size_t>      private_segment_size;
    std::optional<size_t>      sgpr_count;
    std::optional<size_t>      arch_vgpr_count;
    std::optional<size_t>      accum_vgpr_count;
    std::string                extdata;
};

struct code_object_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<size_t>      agent_id;
    std::optional<std::string> uri;
    std::optional<size_t>      load_base;
    std::optional<size_t>      load_size;
    std::optional<size_t>      load_delta;
    std::optional<std::string> storage_type;
    std::string                extdata;
};

struct pmc_info_result
{
    size_t                     id{};
    size_t                     nid{};
    size_t                     pid{};
    std::optional<size_t>      agent_id;
    std::optional<std::string> target_arch;
    std::optional<size_t>      event_code;
    std::optional<size_t>      instance_id;
    std::string                name{};
    std::string                symbol{};
    std::optional<std::string> description;
    std::optional<std::string> long_description;
    std::optional<std::string> component;
    std::optional<std::string> units;
    std::optional<std::string> value_type;
    std::optional<std::string> block;
    std::optional<std::string> expression;
    std::optional<size_t>      is_constant;
    std::optional<size_t>      is_derived;
    std::string                extdata;
};

struct timeline_event_result
{
    size_t id{};

    size_t start_timestamp{};
    size_t end_timestamp{};

    std::optional<size_t> display_name_id;
    // Category decoded to its display string by the backend (v3: rocpd_string,
    // v4: rocpd_info_category), so the reader stays version-agnostic.
    std::optional<std::string> category_name;

    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::optional<size_t> track_id;
};

struct sample_timeline_event_result
{
    size_t                id{};
    size_t                timestamp{};
    std::optional<size_t> category_id;
    size_t                track_id{};
};

// ----- Event detail result structs -----

struct region_detail_result
{
    size_t                id{};
    size_t                start{};
    size_t                end{};
    std::optional<size_t> name_id;
    std::optional<size_t> event_id;
    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::string           extdata;
};

struct kernel_dispatch_detail_result
{
    size_t                id{};
    size_t                dispatch_id{};
    size_t                start{};
    size_t                end{};
    std::optional<size_t> kernel_id;
    std::optional<size_t> private_segment_size;
    std::optional<size_t> group_segment_size;
    size_t                workgroup_size_x{};
    size_t                workgroup_size_y{};
    size_t                workgroup_size_z{};
    size_t                grid_size_x{};
    size_t                grid_size_y{};
    size_t                grid_size_z{};
    std::optional<size_t> region_name_id;
    std::optional<size_t> event_id;
    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::string           extdata;
};

struct memory_copy_detail_result
{
    size_t                id{};
    size_t                start{};
    size_t                end{};
    std::optional<size_t> name_id;
    std::optional<size_t> dst_agent_id;
    std::optional<size_t> dst_address;
    std::optional<size_t> src_agent_id;
    std::optional<size_t> src_address;
    size_t                size{};
    std::optional<size_t> region_name_id;
    std::optional<size_t> event_id;
    size_t                nid{};
    std::optional<size_t> pid;
    std::optional<size_t> tid;
    std::string           extdata;
};

struct memory_alloc_detail_result
{
    size_t                     id{};
    std::optional<std::string> type;
    std::optional<std::string> level;
    size_t                     start{};
    size_t                     end{};
    std::optional<size_t>      address;
    size_t                     size{};
    std::optional<size_t>      event_id;
    size_t                     nid{};
    std::optional<size_t>      pid;
    std::optional<size_t>      tid;
    std::string                extdata;
};

struct event_detail_result
{
    size_t                id{};
    std::optional<size_t> category_id;
    std::optional<size_t> stack_id;
    std::optional<size_t> parent_stack_id;
    std::optional<size_t> correlation_id;
    std::string           call_stack;
    std::string           line_info;
    std::string           extdata;
};

struct arg_detail_result
{
    size_t      position{};
    std::string type;
    std::string name;
    std::string value;
    std::string extdata;
};

/// Raw row shape read straight out of the event tables. call_stack / line_info are
/// the schema-native encodings (v3: JSON blob strings on rocpd_event). The backend
/// decodes these into event_id_result before the reader sees them, so the reader
/// stays version-agnostic. v4 has no direct raw analogue (call stack / line info are
/// relational) and assembles event_id_result without this struct.
struct event_id_raw_result
{
    std::optional<size_t>      event_id;
    std::optional<std::string> category_name;
    std::optional<size_t>      stack_id;
    std::optional<size_t>      parent_stack_id;
    std::optional<size_t>      correlation_id;
    std::string                call_stack;
    std::string                line_info;
    std::string                event_extdata;
};

/// Lightweight result for resolving event metadata from event-specific tables.
/// call_stack / line_info are decoded into version-neutral reader structures by the
/// backend (v3: deserialized from JSON; v4: assembled from rocpd_call_stack /
/// rocpd_line_info + the pc/source-code/address-range info tables).
struct event_id_result
{
    std::optional<size_t> event_id;
    // Category decoded to its display string by the backend (v3: rocpd_string,
    // v4: rocpd_info_category), so the reader stays version-agnostic.
    std::optional<std::string>          category_name;
    std::optional<size_t>               stack_id;
    std::optional<size_t>               parent_stack_id;
    std::optional<size_t>               correlation_id;
    reader_types::call_stack_t          call_stack;
    reader_types::source_context_list_t line_info;
    std::string                         event_extdata;
};

struct count_result
{
    size_t count{};
};

struct time_range_result
{
    std::optional<size_t> min_start;
    std::optional<size_t> max_end;
};

/// One GROUP-BY-name aggregate row for get_kernel_summary / get_region_summary.
/// name_ref is the raw name id (kernel_symbol id for kernels, rocpd_string id for
/// regions) which the reader resolves to a display name; nullopt when the grouped
/// name id is NULL. Durations are (end - start) aggregates over the group; avg is
/// computed by the reader from total/count.
struct summary_result
{
    std::optional<size_t> name_ref;
    size_t                count{};
    size_t                total_duration{};
    size_t                min_duration{};
    size_t                max_duration{};
};

// ----- Track-scoped query result structs (interval / scalar / flow) -----

/// One interval row on a track. name_ref is a string id for region/memory_copy
/// tracks and a kernel_symbol id for kernel_dispatch tracks; the reader resolves
/// it to a display_name based on the track type. category is the event's already-
/// resolved category display string (v3 via rocpd_string, v4 via rocpd_info_category);
/// nullopt when the interval query does not join a category source.
struct interval_row_result
{
    size_t                     id{};
    size_t                     start{};
    size_t                     end{};
    std::optional<size_t>      name_ref;
    std::optional<std::string> category;
    /// Which per-type table this row came from, selected as an integer literal per
    /// UNION leg by the stream interval query (values match reader_types::event_type_t:
    /// kernel_dispatch=1, memory_copy=2, memory_allocate=3). nullopt for single-table
    /// interval queries, which do not select it.
    std::optional<size_t> op_kind;
};

/// One scalar (counter) sample on a counter track.
struct scalar_row_result
{
    size_t id{};
    size_t timestamp{};
    double value{};
};

/// Bounds/count for a single track, computed via MIN/MAX/COUNT aggregates over the
/// same events the matching interval/scalar track query would return. min_ts/max_ts
/// are nullopt when the track has no events (aggregate over empty set is NULL).
struct track_stats_result
{
    std::optional<size_t> min_ts;
    std::optional<size_t> max_ts;
    size_t                count{};
};

/// A single candidate flow leg from the stack-clique join. Carries both endpoints' row
/// ids plus the fields get_flows needs to orient and group the directed edge: each
/// endpoint's start, the shared clique stack_id, and each endpoint's parent_stack_id.
struct flow_row_result
{
    size_t                source_id{};
    size_t                dest_id{};
    size_t                source_start{};   ///< SQL source endpoint start timestamp.
    size_t                dest_start{};     ///< SQL dest endpoint start timestamp.
    size_t                stack_id{};       ///< Shared clique stack_id (both endpoints).
    std::optional<size_t> source_parent{};  ///< SQL source endpoint parent_stack_id.
    std::optional<size_t> dest_parent{};    ///< SQL dest endpoint parent_stack_id.
};

/// Distinct gpu_queue topology context synthesized from rocpd_kernel_dispatch.
struct distinct_gpu_queue_result
{
    size_t nid{};
    size_t pid{};
    size_t agent_id{};
    size_t queue_id{};
};

/// Distinct dma topology context synthesized from rocpd_memory_copy. Keyed on the
/// destination agent (dst_agent_id) to match Optiq's GetRocprofMemoryCopyTrackQuery
/// swimlane grouping; stream identity lives on the separate `stream` track type.
/// queue_id / dst_agent_id are kept as distinct group values, NULL included.
struct distinct_dma_result
{
    size_t                nid{};
    size_t                pid{};
    std::optional<size_t> queue_id;
    std::optional<size_t> dst_agent_id;
};

/// Distinct memory-track topology synthesized from rocpd_memory_allocate. Keyed on
/// (nid, agent_id, queue_id, pid) to match Optiq's GetRocprofMemoryAllocTrackQuery
/// GROUP BY exactly. agent_id / queue_id are nullable; NULL is a distinct group value.
struct distinct_memory_result
{
    size_t                nid{};
    size_t                pid{};
    std::optional<size_t> agent_id;
    std::optional<size_t> queue_id;
};

/// Distinct kernel-dispatch PMC track topology. v3: synthesized from rocpd_pmc_event
/// INNER JOIN rocpd_kernel_dispatch. v4.0: same joins plus rocpd_track for nid/pid.
/// Keyed (nid, agent_id, pmc_id, pid) to match Optiq's
/// GetRocprofPerformanceCountersTrackQuery GROUP BY exactly. All four fields are
/// non-nullable (Optiq uses INNER JOINs with no NULL handling).
struct distinct_kd_pmc_result
{
    size_t nid{};
    size_t pid{};
    size_t agent_id{};
    size_t pmc_id{};
};

/// Distinct memory-activity track topology. One series per (nid, pid, agent_id) from
/// rocpd_memory_allocate, matching Optiq's per-agent grouping. agent_id is nullable;
/// NULL is a distinct group value (preserved, not dropped).
struct distinct_mem_activity_result
{
    size_t                nid{};
    size_t                pid{};
    std::optional<size_t> agent_id;
};

/// One raw rocpd_memory_allocate row for memory_activity running-sum computation.
/// agent_id is nullable (FREE rows may carry NULL agent_id; recovery is done in C++
/// via address self-join). size is always present (schema NOT NULL).
struct mem_activity_raw_result
{
    size_t                id{};
    size_t                start{};
    std::optional<size_t> address;
    size_t                size{};
    std::optional<size_t> agent_id;
    std::string           type{};  ///< "ALLOC", "FREE", "REALLOC", "RECLAIM"
};

/// Distinct stream-track topology. v3: synthesized from the inline stream_id on
/// rocpd_kernel_dispatch / rocpd_memory_copy / rocpd_memory_allocate. v4.0: from
/// rocpd_track.stream_id. One row per (nid, pid, stream_id) with a non-null stream_id;
/// a stream track aggregates dispatch + copy + alloc events sharing that stream.
struct distinct_stream_result
{
    size_t nid{};
    size_t pid{};
    size_t stream_id{};
};

/// Distinct region-track topology synthesized from rocpd_region (v3). One row per
/// (nid, pid, tid, is_sample): a thread with both plain and sampled regions yields
/// two rows (main + sample), mirroring roc-optiq's region-main / region-sample split.
struct distinct_region_result
{
    size_t nid{};
    size_t pid{};
    size_t tid{};
    size_t is_sample{};  ///< 0 => region events with no sample (main); 1 => with sample.
};

/// A track_id referenced by at least one rocpd_sample row (=> counter track).
struct sample_track_id_result
{
    size_t track_id{};
};

/// MAX(id) of rocpd_track, used as the base for synthetic track ids.
struct max_track_id_result
{
    std::optional<size_t> max_id;
};

/// pmc_id that has more than one rocpd_pmc_event row for the same event_id — a
/// legacy producer bug where two quantities were stamped with one pmc_id. One row
/// per ambiguous pmc_id.
struct ambiguous_pmc_id_result
{
    size_t pmc_id{};
};

/// Maps a counter track_id to its PMC id and name (rocpd_info_pmc). One row per
/// counter track.
struct counter_track_name_result
{
    size_t      track_id{};
    size_t      pmc_id{};
    std::string name;
};

/// Combined scalar detail (sample + counter value) resolved by rocpd_sample.id.
struct scalar_detail_result
{
    size_t                id{};
    size_t                track_id{};
    size_t                timestamp{};
    double                value{};
    std::optional<size_t> event_id;
};

// ---------------------------------------------------------------------------
// Abstract read-statements interface.
//
// This is the version-dispatch seam (task 002B): reader_impl holds a
// shared_ptr<read_statements_base>, selected once at construction based on the
// detected schema version. Both schema_v3::read_statements and
// schema_v4::read_statements derive from this and provide their own SQL.
//
// Accessors split into two groups:
//   * PURE VIRTUAL  — the shared subset both backends implement (info tables,
//     track info, counter/scalar/flow track-scoped queries).
//   * VIRTUAL with a default-empty body — backend-specific statements. The v3
//     backend overrides the legacy timeline/detail/synthesis/multi-column
//     interval accessors; the v4.0 backend overrides the track_id-anchored
//     interval accessors. Each backend inherits empty stubs for the accessors
//     it does not implement. The reader guards those paths (never invokes an
//     unimplemented accessor), so the empty std::function objects returned here
//     are never called.
// ---------------------------------------------------------------------------
struct read_statements_base
{
    read_statements_base()                                       = default;
    read_statements_base(const read_statements_base&)            = delete;
    read_statements_base(read_statements_base&&)                 = delete;
    read_statements_base& operator=(const read_statements_base&) = delete;
    read_statements_base& operator=(read_statements_base&&)      = delete;
    virtual ~read_statements_base()                              = default;

    // ----- func typedefs (shared) -----
    using string_statement_func_t =
        std::function<sqlite_backend::result_set<string_result>()>;
    using node_info_statement_func_t =
        std::function<sqlite_backend::result_set<node_info_result>()>;
    using process_info_statement_func_t =
        std::function<sqlite_backend::result_set<process_info_result>()>;
    using stream_info_statement_func_t =
        std::function<sqlite_backend::result_set<stream_info_result>()>;
    using queue_info_statement_func_t =
        std::function<sqlite_backend::result_set<queue_info_result>()>;
    using thread_info_statement_func_t =
        std::function<sqlite_backend::result_set<thread_info_result>()>;
    using agent_info_statement_func_t =
        std::function<sqlite_backend::result_set<agent_info_result>()>;
    using track_info_statement_func_t =
        std::function<sqlite_backend::result_set<track_info_result>()>;
    using kernel_symbol_info_statement_func_t =
        std::function<sqlite_backend::result_set<kernel_symbol_info_result>()>;
    using code_object_info_statement_func_t =
        std::function<sqlite_backend::result_set<code_object_info_result>()>;
    using pmc_info_statement_func_t =
        std::function<sqlite_backend::result_set<pmc_info_result>()>;

    using timeline_event_statement_func_t =
        std::function<sqlite_backend::result_set<timeline_event_result>()>;
    using timeline_event_time_filtered_func_t =
        std::function<sqlite_backend::result_set<timeline_event_result>(size_t, size_t)>;
    using timeline_event_track_filtered_func_t = std::function<sqlite_backend::result_set<
        timeline_event_result>(size_t, size_t, size_t, size_t)>;
    using timeline_event_track_and_time_filtered_func_t =
        std::function<sqlite_backend::result_set<
            timeline_event_result>(size_t, size_t, size_t, size_t, size_t, size_t)>;

    using region_detail_func_t =
        std::function<sqlite_backend::result_set<region_detail_result>(size_t)>;
    using kernel_dispatch_detail_func_t =
        std::function<sqlite_backend::result_set<kernel_dispatch_detail_result>(size_t)>;
    using memory_copy_detail_func_t =
        std::function<sqlite_backend::result_set<memory_copy_detail_result>(size_t)>;
    using memory_alloc_detail_func_t =
        std::function<sqlite_backend::result_set<memory_alloc_detail_result>(size_t)>;
    using event_detail_func_t =
        std::function<sqlite_backend::result_set<event_detail_result>(size_t)>;
    using arg_detail_func_t =
        std::function<sqlite_backend::result_set<arg_detail_result>(size_t)>;
    // Materialized (not a lazy result_set): the backend runs the query and decodes
    // call_stack / line_info into event_id_result before returning. v4 assembles the
    // call stack / line info from multiple relational rows, which a single lazy
    // result_set cannot express, so both backends return a fully-built vector.
    using event_id_func_t = std::function<std::vector<event_id_result>(size_t)>;
    using count_func_t    = std::function<sqlite_backend::result_set<count_result>()>;
    using count_time_filtered_func_t =
        std::function<sqlite_backend::result_set<count_result>(size_t, size_t)>;
    using time_range_func_t =
        std::function<sqlite_backend::result_set<time_range_result>()>;
    using summary_func_t = std::function<sqlite_backend::result_set<summary_result>()>;
    using summary_time_filtered_func_t =
        std::function<sqlite_backend::result_set<summary_result>(size_t, size_t)>;

    using correlated_event_func_t =
        std::function<sqlite_backend::result_set<timeline_event_result>(size_t, size_t)>;

    // Track-scoped query func types.
    using interval_track_1_func_t =
        std::function<sqlite_backend::result_set<interval_row_result>(size_t)>;
    using interval_track_2_func_t =
        std::function<sqlite_backend::result_set<interval_row_result>(size_t, size_t)>;
    using interval_track_3_func_t = std::function<
        sqlite_backend::result_set<interval_row_result>(size_t, size_t, size_t)>;
    using interval_track_4_func_t = std::function<
        sqlite_backend::result_set<interval_row_result>(size_t, size_t, size_t, size_t)>;

    using scalar_track_func_t =
        std::function<sqlite_backend::result_set<scalar_row_result>(size_t)>;

    // Track-stats func types: MIN/MAX/COUNT aggregates, one arity per identity-tuple
    // shape (v4 track_id-anchored + scalar use the 1-arg form).
    using stats_track_1_func_t =
        std::function<sqlite_backend::result_set<track_stats_result>(size_t)>;
    using stats_track_2_func_t =
        std::function<sqlite_backend::result_set<track_stats_result>(size_t, size_t)>;
    using stats_track_3_func_t = std::function<
        sqlite_backend::result_set<track_stats_result>(size_t, size_t, size_t)>;
    using stats_track_4_func_t = std::function<
        sqlite_backend::result_set<track_stats_result>(size_t, size_t, size_t, size_t)>;

    using flow_func_t = std::function<sqlite_backend::result_set<flow_row_result>()>;
    using flow_time_filtered_func_t =
        std::function<sqlite_backend::result_set<flow_row_result>(size_t, size_t)>;

    using distinct_gpu_queue_func_t =
        std::function<sqlite_backend::result_set<distinct_gpu_queue_result>()>;
    using distinct_dma_func_t =
        std::function<sqlite_backend::result_set<distinct_dma_result>()>;
    using distinct_memory_func_t =
        std::function<sqlite_backend::result_set<distinct_memory_result>()>;
    using distinct_kd_pmc_func_t =
        std::function<sqlite_backend::result_set<distinct_kd_pmc_result>()>;
    using distinct_mem_activity_func_t =
        std::function<sqlite_backend::result_set<distinct_mem_activity_result>()>;
    // All rocpd_memory_allocate rows for (nid, pid), ordered by start. C++ computes
    // per-agent running sums and recovers FREE agent_id via address self-join.
    using mem_activity_raw_func_t =
        std::function<sqlite_backend::result_set<mem_activity_raw_result>(size_t,
                                                                          size_t)>;
    // v4.0 only: track_ids referenced by rocpd_memory_allocate, used in the generic
    // classification loop to distinguish memory tracks from gpu_queue tracks (both may
    // have agent_id + queue_id on their rocpd_track row).
    using memory_alloc_track_ids_func_t =
        std::function<sqlite_backend::result_set<sample_track_id_result>()>;
    using distinct_region_func_t =
        std::function<sqlite_backend::result_set<distinct_region_result>()>;
    using distinct_stream_func_t =
        std::function<sqlite_backend::result_set<distinct_stream_result>()>;
    using sample_track_id_func_t =
        std::function<sqlite_backend::result_set<sample_track_id_result>()>;
    using max_track_id_func_t =
        std::function<sqlite_backend::result_set<max_track_id_result>()>;
    using counter_track_name_func_t =
        std::function<sqlite_backend::result_set<counter_track_name_result>()>;

    using scalar_detail_func_t =
        std::function<sqlite_backend::result_set<scalar_detail_result>(size_t)>;

    using ambiguous_pmc_ids_func_t =
        std::function<sqlite_backend::result_set<ambiguous_pmc_id_result>()>;

    // ----- set structs (shared) -----
    struct timeline_event_statement_set
    {
        timeline_event_statement_func_t               base;
        timeline_event_time_filtered_func_t           time_filtered;
        timeline_event_track_filtered_func_t          track_filtered;
        timeline_event_track_and_time_filtered_func_t track_and_time_filtered;
    };

    struct correlated_event_statement_set
    {
        correlated_event_func_t region;
        correlated_event_func_t kernel_dispatch;
        correlated_event_func_t memory_copy;
        correlated_event_func_t memory_allocate;
    };

    struct flow_statement_set
    {
        flow_func_t               base;
        flow_time_filtered_func_t time_filtered;
    };

    // ======================================================================
    // Shared subset — pure virtual (both backends implement).
    // ======================================================================
    [[nodiscard]] virtual string_statement_func_t       string_statement() const    = 0;
    [[nodiscard]] virtual node_info_statement_func_t    node_info_statement() const = 0;
    [[nodiscard]] virtual process_info_statement_func_t process_info_statement()
        const                                                                        = 0;
    [[nodiscard]] virtual stream_info_statement_func_t stream_info_statement() const = 0;
    [[nodiscard]] virtual queue_info_statement_func_t  queue_info_statement() const  = 0;
    [[nodiscard]] virtual thread_info_statement_func_t thread_info_statement() const = 0;
    [[nodiscard]] virtual agent_info_statement_func_t  agent_info_statement() const  = 0;
    [[nodiscard]] virtual track_info_statement_func_t  track_info_statement() const  = 0;
    [[nodiscard]] virtual kernel_symbol_info_statement_func_t
    kernel_symbol_info_statement() const = 0;
    [[nodiscard]] virtual code_object_info_statement_func_t code_object_info_statement()
        const                                                                        = 0;
    [[nodiscard]] virtual pmc_info_statement_func_t       pmc_info_statement() const = 0;
    [[nodiscard]] virtual const ambiguous_pmc_ids_func_t& ambiguous_pmc_ids() const  = 0;

    [[nodiscard]] virtual const sample_track_id_func_t& distinct_sample_track_ids()
        const = 0;
    [[nodiscard]] virtual const counter_track_name_func_t& counter_track_names()
        const                                                                  = 0;
    [[nodiscard]] virtual const scalar_track_func_t&  scalar_track() const     = 0;
    [[nodiscard]] virtual const scalar_detail_func_t& scalar_detail() const    = 0;
    [[nodiscard]] virtual const scalar_detail_func_t& pmc_event_detail() const = 0;

    [[nodiscard]] virtual const flow_statement_set& region_to_kernel_dispatch_flows()
        const = 0;
    [[nodiscard]] virtual const flow_statement_set& region_to_memory_copy_flows()
        const = 0;
    [[nodiscard]] virtual const flow_statement_set& region_to_memory_allocate_flows()
        const = 0;
    // Full stack-clique legs beyond region->GPU: region->region and same-type siblings.
    [[nodiscard]] virtual const flow_statement_set& region_to_region_flows() const = 0;
    [[nodiscard]] virtual const flow_statement_set& kernel_dispatch_sibling_flows()
        const                                                                         = 0;
    [[nodiscard]] virtual const flow_statement_set& memory_copy_sibling_flows() const = 0;
    [[nodiscard]] virtual const flow_statement_set& memory_allocate_sibling_flows()
        const = 0;

    // ======================================================================
    // Backend-specific — default-empty. Overridden by whichever backend
    // implements them; the other backend inherits the empty stub (never
    // invoked, guarded in the reader).
    // ======================================================================

    // Legacy timeline event statement sets (v3).
    [[nodiscard]] virtual const timeline_event_statement_set& region_statements() const
    {
        static const timeline_event_statement_set e{};
        return e;
    }
    [[nodiscard]] virtual const timeline_event_statement_set& kernel_dispatch_statements()
        const
    {
        static const timeline_event_statement_set e{};
        return e;
    }
    [[nodiscard]] virtual const timeline_event_statement_set& memory_allocate_statements()
        const
    {
        static const timeline_event_statement_set e{};
        return e;
    }
    [[nodiscard]] virtual const timeline_event_statement_set& memory_copy_statements()
        const
    {
        static const timeline_event_statement_set e{};
        return e;
    }

    // Legacy detail statements (v3).
    [[nodiscard]] virtual const region_detail_func_t& region_detail() const
    {
        static const region_detail_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const kernel_dispatch_detail_func_t& kernel_dispatch_detail()
        const
    {
        static const kernel_dispatch_detail_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const memory_copy_detail_func_t& memory_copy_detail() const
    {
        static const memory_copy_detail_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const memory_alloc_detail_func_t& memory_alloc_detail() const
    {
        static const memory_alloc_detail_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const event_detail_func_t& event_detail() const
    {
        static const event_detail_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const arg_detail_func_t& arg_detail() const
    {
        static const arg_detail_func_t e{};
        return e;
    }

    // Legacy event-id resolution (v3).
    [[nodiscard]] virtual const event_id_func_t& region_event_id() const
    {
        static const event_id_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const event_id_func_t& kernel_dispatch_event_id() const
    {
        static const event_id_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const event_id_func_t& memory_copy_event_id() const
    {
        static const event_id_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const event_id_func_t& memory_alloc_event_id() const
    {
        static const event_id_func_t e{};
        return e;
    }

    // Legacy correlated events (v3).
    [[nodiscard]] virtual const correlated_event_statement_set&
    correlated_event_statements() const
    {
        static const correlated_event_statement_set e{};
        return e;
    }

    // Legacy counts / time ranges (v3).
    [[nodiscard]] virtual const count_func_t& region_count() const
    {
        static const count_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const count_func_t& kernel_dispatch_count() const
    {
        static const count_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const count_func_t& memory_copy_count() const
    {
        static const count_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const count_func_t& memory_alloc_count() const
    {
        static const count_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const count_time_filtered_func_t& region_count_time_filtered()
        const
    {
        static const count_time_filtered_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const count_time_filtered_func_t&
    kernel_dispatch_count_time_filtered() const
    {
        static const count_time_filtered_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const count_time_filtered_func_t&
    memory_copy_count_time_filtered() const
    {
        static const count_time_filtered_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const count_time_filtered_func_t&
    memory_alloc_count_time_filtered() const
    {
        static const count_time_filtered_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const time_range_func_t& region_time_range() const
    {
        static const time_range_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const time_range_func_t& kernel_dispatch_time_range() const
    {
        static const time_range_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const time_range_func_t& memory_copy_time_range() const
    {
        static const time_range_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const time_range_func_t& memory_alloc_time_range() const
    {
        static const time_range_func_t e{};
        return e;
    }

    // GROUP-BY-name aggregates for get_kernel_summary / get_region_summary. Both
    // backends override; the default static-empty function is never called.
    [[nodiscard]] virtual const summary_func_t& kernel_summary() const
    {
        static const summary_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const summary_time_filtered_func_t&
    kernel_summary_time_filtered() const
    {
        static const summary_time_filtered_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const summary_func_t& region_summary() const
    {
        static const summary_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const summary_time_filtered_func_t&
    region_summary_time_filtered() const
    {
        static const summary_time_filtered_func_t e{};
        return e;
    }

    // Track synthesis (v3-only; v4.0 tracks are real rocpd_track rows).
    [[nodiscard]] virtual const distinct_gpu_queue_func_t& distinct_gpu_queue_tracks()
        const
    {
        static const distinct_gpu_queue_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const distinct_dma_func_t& distinct_dma_tracks() const
    {
        static const distinct_dma_func_t e{};
        return e;
    }
    // v3: synthesized from rocpd_memory_allocate; v4.0: via rocpd_memory_allocate JOIN
    // rocpd_track. One row per (nid, agent_id, queue_id, pid), NULL included as a
    // distinct group value for both nullable columns.
    [[nodiscard]] virtual const distinct_memory_func_t& distinct_memory_tracks() const
    {
        static const distinct_memory_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const distinct_region_func_t& distinct_region_tracks() const
    {
        static const distinct_region_func_t e{};
        return e;
    }
    // Distinct stream tracks (both backends). v3 unions the inline stream_id across the
    // three event tables; v4.0 reads rocpd_track.stream_id. See distinct_stream_result.
    [[nodiscard]] virtual const distinct_stream_func_t& distinct_stream_tracks() const
    {
        static const distinct_stream_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const max_track_id_func_t& max_track_id() const
    {
        static const max_track_id_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const memory_alloc_track_ids_func_t& memory_alloc_track_ids()
        const
    {
        static const memory_alloc_track_ids_func_t e{};
        return e;
    }
    // v3 + v4.0: distinct kernel-dispatch PMC tracks, one per (nid, agent_id, pmc_id,
    // pid).
    [[nodiscard]] virtual const distinct_kd_pmc_func_t& distinct_kd_pmc_tracks() const
    {
        static const distinct_kd_pmc_func_t e{};
        return e;
    }
    // v4.0 only: rocpd_track ids referenced by rocpd_kernel_dispatch via rocpd_pmc_event,
    // used in the generic classification loop to prevent gpu_queue from claiming them.
    [[nodiscard]] virtual const sample_track_id_func_t& kd_pmc_track_ids() const
    {
        static const sample_track_id_func_t e{};
        return e;
    }

    // v3 + v4.0: distinct memory-activity tracks, one per (nid, pid, agent_id).
    [[nodiscard]] virtual const distinct_mem_activity_func_t&
    distinct_mem_activity_tracks() const
    {
        static const distinct_mem_activity_func_t e{};
        return e;
    }
    // All rocpd_memory_allocate rows for (nid, pid), ordered by start. Used by both
    // get_scalar_track and get_track_stats for memory_activity (C++ computes per-agent
    // running sums and recovers FREE agent_id via address self-join).
    [[nodiscard]] virtual const mem_activity_raw_func_t& mem_activity_raw_track() const
    {
        static const mem_activity_raw_func_t e{};
        return e;
    }

    // Multi-column interval track statements (v3: keyed by identity tuples).
    // Region tracks split main (regions without a sample) vs. sample (regions with
    // one), matching the region_track_kind_t of the synthesized track.
    [[nodiscard]] virtual const interval_track_3_func_t& region_interval_track_main()
        const
    {
        static const interval_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_3_func_t& region_interval_track_sample()
        const
    {
        static const interval_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_4_func_t& kernel_dispatch_interval_track()
        const
    {
        static const interval_track_4_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_4_func_t& memory_copy_interval_qa() const
    {
        static const interval_track_4_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_3_func_t& memory_copy_interval_q_only()
        const
    {
        static const interval_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_3_func_t& memory_copy_interval_a_only()
        const
    {
        static const interval_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_2_func_t& memory_copy_interval_neither()
        const
    {
        static const interval_track_2_func_t e{};
        return e;
    }
    // memory (memory_allocate) interval track statements (v3: keyed by (nid, pid,
    // agent_id, queue_id) with 4 NULL variants for the two nullable columns).
    [[nodiscard]] virtual const interval_track_4_func_t& memory_alloc_interval_qa() const
    {
        static const interval_track_4_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_3_func_t& memory_alloc_interval_q_only()
        const
    {
        static const interval_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_3_func_t& memory_alloc_interval_a_only()
        const
    {
        static const interval_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_2_func_t& memory_alloc_interval_neither()
        const
    {
        static const interval_track_2_func_t e{};
        return e;
    }

    // track_id-anchored interval track statements (v4.0: single WHERE track_id = ?).
    [[nodiscard]] virtual const interval_track_1_func_t& region_interval_track_v4() const
    {
        static const interval_track_1_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_1_func_t&
    kernel_dispatch_interval_track_v4() const
    {
        static const interval_track_1_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const interval_track_1_func_t& memory_copy_interval_track_v4()
        const
    {
        static const interval_track_1_func_t e{};
        return e;
    }
    // v4.0: memory allocations keyed by track_id (rocpd_memory_allocate.track_id).
    [[nodiscard]] virtual const interval_track_1_func_t& memory_alloc_interval_track_v4()
        const
    {
        static const interval_track_1_func_t e{};
        return e;
    }
    // kernel_dispatch_pmc interval track (both backends): keyed by (nid, pid, agent_id,
    // pmc_id). v3 uses inline start/end; v4.0 overrides with the timestamp-spine variant.
    [[nodiscard]] virtual const interval_track_4_func_t& kd_pmc_interval_track() const
    {
        static const interval_track_4_func_t e{};
        return e;
    }

    // Stream track interval query (both backends): a 3-way UNION over
    // kernel_dispatch + memory_copy + memory_allocate filtered to one stream, binding
    // stream_id once per leg (hence the 3-arg form). Each row carries op_kind so the
    // reader can select the right name lookup and get_*_details() overload.
    [[nodiscard]] virtual const interval_track_3_func_t& stream_interval_track() const
    {
        static const interval_track_3_func_t e{};
        return e;
    }

    // ----- Track-stats aggregates (MIN/MAX/COUNT) -----
    // Shape-matched to the interval/scalar track statements above so the aggregate
    // scopes to exactly the events that track would return. v3 overrides the
    // multi-column variants; v4.0 overrides the track_id-anchored *_v4 variants; both
    // override scalar_stats. Unimplemented variants inherit these empty stubs (the
    // reader routes only to the ones its backend implements).
    [[nodiscard]] virtual const stats_track_3_func_t& region_stats_track_main() const
    {
        static const stats_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_3_func_t& region_stats_track_sample() const
    {
        static const stats_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_4_func_t& kernel_dispatch_stats_track() const
    {
        static const stats_track_4_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_4_func_t& memory_copy_stats_qa() const
    {
        static const stats_track_4_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_3_func_t& memory_copy_stats_q_only() const
    {
        static const stats_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_3_func_t& memory_copy_stats_a_only() const
    {
        static const stats_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_2_func_t& memory_copy_stats_neither() const
    {
        static const stats_track_2_func_t e{};
        return e;
    }
    // memory (v3): stats aggregates keyed by (nid, pid, agent_id, queue_id), 4 variants.
    [[nodiscard]] virtual const stats_track_4_func_t& memory_alloc_stats_qa() const
    {
        static const stats_track_4_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_3_func_t& memory_alloc_stats_q_only() const
    {
        static const stats_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_3_func_t& memory_alloc_stats_a_only() const
    {
        static const stats_track_3_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_2_func_t& memory_alloc_stats_neither() const
    {
        static const stats_track_2_func_t e{};
        return e;
    }

    [[nodiscard]] virtual const stats_track_1_func_t& region_stats_track_v4() const
    {
        static const stats_track_1_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_1_func_t& kernel_dispatch_stats_track_v4()
        const
    {
        static const stats_track_1_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_1_func_t& memory_copy_stats_track_v4() const
    {
        static const stats_track_1_func_t e{};
        return e;
    }
    [[nodiscard]] virtual const stats_track_1_func_t& memory_alloc_stats_track_v4() const
    {
        static const stats_track_1_func_t e{};
        return e;
    }
    // kernel_dispatch_pmc stats (both backends): keyed by (nid, pid, agent_id, pmc_id).
    // v3 uses inline start/end; v4.0 overrides with the timestamp-spine variant.
    [[nodiscard]] virtual const stats_track_4_func_t& kd_pmc_stats_track() const
    {
        static const stats_track_4_func_t e{};
        return e;
    }

    // Stream track stats (both backends): MIN(start)/MAX(end)/COUNT over the same 3-way
    // UNION as stream_interval_track, binding stream_id once per leg.
    [[nodiscard]] virtual const stats_track_3_func_t& stream_stats_track() const
    {
        static const stats_track_3_func_t e{};
        return e;
    }

    // Counter (scalar) track stats over rocpd_sample; implemented by both backends.
    [[nodiscard]] virtual const stats_track_1_func_t& scalar_stats() const
    {
        static const stats_track_1_func_t e{};
        return e;
    }
};

}  // namespace profiler_hub::data_storage
