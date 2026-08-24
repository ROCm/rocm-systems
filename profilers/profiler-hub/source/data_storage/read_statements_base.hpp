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
// Result structs shared by v3 and v4.0 read_statements; per-schema SQL fills them.
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

// v4.0 populates agent_id/queue_id/stream_id (rocpd_track carries them); v3 leaves
// them nullopt.
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

    std::optional<size_t>      display_name_id;
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

/// Raw row shape read straight out of the event tables (v3 only); call_stack/line_info
/// are JSON blob strings on rocpd_event, decoded into event_id_result before the reader
/// sees them. v4 has no analogous raw struct — call stack/line info are relational there.
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

struct event_id_result
{
    std::optional<size_t>               event_id;
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

/// name_ref is the raw name id (kernel_symbol id for kernels, rocpd_string id for
/// regions); nullopt when the grouped name id is NULL. avg is computed by the reader
/// from total/count, not stored here.
struct summary_result
{
    std::optional<size_t> name_ref;
    size_t                count{};
    size_t                total_duration{};
    size_t                min_duration{};
    size_t                max_duration{};
};

/// name_ref is a string id for region/memory_copy tracks and a kernel_symbol id for
/// kernel_dispatch tracks; resolved to a display_name based on track type. category
/// is nullopt when the interval query does not join a category source.
struct interval_row_result
{
    size_t                     id{};
    size_t                     start{};
    size_t                     end{};
    std::optional<size_t>      name_ref;
    std::optional<std::string> category;
    /// Row-source discriminant for the stream UNION (kernel_dispatch=1, memory_copy=2,
    /// memory_allocate=3, matching reader_types::event_type_t); nullopt for single-table
    /// interval queries.
    std::optional<size_t> op_kind;
};

struct scalar_row_result
{
    size_t id{};
    size_t timestamp{};
    double value{};
};

/// min_ts/max_ts are nullopt when the track has no events (aggregate over empty set
/// is NULL).
struct track_stats_result
{
    std::optional<size_t> min_ts;
    std::optional<size_t> max_ts;
    size_t                count{};
};

/// Fields get_flows needs to orient and group the directed edge: each endpoint's
/// start, the shared clique stack_id, each endpoint's parent_stack_id.
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

struct distinct_gpu_queue_result
{
    size_t nid{};
    size_t pid{};
    size_t agent_id{};
    size_t queue_id{};
};

/// Keyed on the destination agent (dst_agent_id) to match Optiq's
/// GetRocprofMemoryCopyTrackQuery swimlane grouping; stream identity lives on the
/// separate `stream` track type. queue_id / dst_agent_id are kept as distinct group
/// values, NULL included.
struct distinct_dma_result
{
    size_t                nid{};
    size_t                pid{};
    std::optional<size_t> queue_id;
    std::optional<size_t> dst_agent_id;
};

/// Keyed on (nid, agent_id, queue_id, pid) to match Optiq's
/// GetRocprofMemoryAllocTrackQuery GROUP BY exactly. agent_id / queue_id are nullable;
/// NULL is a distinct group value.
struct distinct_memory_result
{
    size_t                nid{};
    size_t                pid{};
    std::optional<size_t> agent_id;
    std::optional<size_t> queue_id;
};

/// Keyed (nid, agent_id, pmc_id, pid) to match Optiq's
/// GetRocprofPerformanceCountersTrackQuery GROUP BY exactly; all four fields are
/// non-nullable (Optiq uses INNER JOINs with no NULL handling).
struct distinct_kd_pmc_result
{
    size_t nid{};
    size_t pid{};
    size_t agent_id{};
    size_t pmc_id{};
};

/// One series per (nid, pid, agent_id) from rocpd_memory_allocate, matching Optiq's
/// per-agent grouping. agent_id is nullable; NULL is a distinct group value
/// (preserved, not dropped).
struct distinct_mem_activity_result
{
    size_t                nid{};
    size_t                pid{};
    std::optional<size_t> agent_id;
};

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

/// One row per (nid, pid, stream_id) with a non-null stream_id; a stream track
/// aggregates dispatch + copy + alloc events sharing that stream.
struct distinct_stream_result
{
    size_t nid{};
    size_t pid{};
    size_t stream_id{};
};

/// One row per (nid, pid, tid, is_sample): a thread with both plain and sampled
/// regions yields two rows (main + sample), mirroring roc-optiq's region-main /
/// region-sample split.
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

/// pmc_id with more than one rocpd_pmc_event row for the same event_id (legacy
/// producer bug stamping two quantities with one pmc_id).
struct ambiguous_pmc_id_result
{
    size_t pmc_id{};
};

struct counter_track_name_result
{
    size_t      track_id{};
    size_t      pmc_id{};
    std::string name;
};

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
// This is the version-dispatch seam: reader_impl holds a
// shared_ptr<read_statements_base>, selected once at construction based on the
// detected schema version. Both schema_v3::read_statements and
// schema_v4::read_statements derive from this and provide their own SQL.
//
// The v4.0 schema targeted here is a draft and is expected to change; this seam is
// where that revision lands.
// ---------------------------------------------------------------------------
struct read_statements_base
{
    read_statements_base()                                       = default;
    read_statements_base(const read_statements_base&)            = delete;
    read_statements_base(read_statements_base&&)                 = delete;
    read_statements_base& operator=(const read_statements_base&) = delete;
    read_statements_base& operator=(read_statements_base&&)      = delete;
    virtual ~read_statements_base()                              = default;

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
    // Materialized (not a lazy result_set): a single lazy result_set cannot express
    // the one-to-many call-stack/line-info fan-out, so both backends return a
    // fully-built vector.
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

    // Arity encodes the identity-tuple shape: 1-arg is used by v4 track_id-anchored +
    // scalar variants; 2/3/4-arg by v3 multi-column keys.
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
    using mem_activity_raw_func_t =
        std::function<sqlite_backend::result_set<mem_activity_raw_result>(size_t,
                                                                          size_t)>;
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
        const                                                                      = 0;
    [[nodiscard]] virtual const flow_statement_set& region_to_region_flows() const = 0;
    [[nodiscard]] virtual const flow_statement_set& kernel_dispatch_sibling_flows()
        const                                                                         = 0;
    [[nodiscard]] virtual const flow_statement_set& memory_copy_sibling_flows() const = 0;
    [[nodiscard]] virtual const flow_statement_set& memory_allocate_sibling_flows()
        const = 0;

    // Backend-specific, default-empty; unused variants are guarded by the reader
    // before being called.

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

    // Both backends override this; the default static-empty function is never called.
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

    [[nodiscard]] virtual const distinct_mem_activity_func_t&
    distinct_mem_activity_tracks() const
    {
        static const distinct_mem_activity_func_t e{};
        return e;
    }
    // All rocpd_memory_allocate rows for (nid, pid), ordered by start. Used by both
    // get_scalar_track and get_track_stats for memory_activity.
    [[nodiscard]] virtual const mem_activity_raw_func_t& mem_activity_raw_track() const
    {
        static const mem_activity_raw_func_t e{};
        return e;
    }

    // v3-only: multi-column interval track statements keyed by identity tuples. Region
    // tracks split main (no sample) vs. sample, matching region_track_kind_t.
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
    [[nodiscard]] virtual const interval_track_1_func_t& memory_alloc_interval_track_v4()
        const
    {
        static const interval_track_1_func_t e{};
        return e;
    }
    // kernel_dispatch_pmc interval (both backends), keyed by (nid, pid, agent_id,
    // pmc_id). v3 uses inline start/end; v4.0 joins the timestamp spine.
    [[nodiscard]] virtual const interval_track_4_func_t& kd_pmc_interval_track() const
    {
        static const interval_track_4_func_t e{};
        return e;
    }

    // Stream track query (both backends): a 3-way UNION binding stream_id once per leg
    // (hence the 3-arg form); each row carries op_kind for the reader's per-type
    // dispatch.
    [[nodiscard]] virtual const interval_track_3_func_t& stream_interval_track() const
    {
        static const interval_track_3_func_t e{};
        return e;
    }

    // ----- Track-stats aggregates (MIN/MAX/COUNT) -----
    // Shape-matched to the interval/scalar track statements above so the aggregate
    // scopes to exactly the events that track would return. Unimplemented variants
    // inherit these empty stubs.
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
    [[nodiscard]] virtual const stats_track_4_func_t& kd_pmc_stats_track() const
    {
        static const stats_track_4_func_t e{};
        return e;
    }

    // Stream track stats (both backends): same 3-way UNION as stream_interval_track,
    // MIN/MAX/COUNT instead of full rows.
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
