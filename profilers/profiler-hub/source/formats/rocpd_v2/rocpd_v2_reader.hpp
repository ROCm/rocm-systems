// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "reader_impl.hpp"

#include "data_storage/backends/sqlite_backend.hpp"
#include "reader_catalog.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace profiler_hub
{
class trim_schema_t;

namespace rocpd_v2
{
// Node id of the single synthesized node.
constexpr size_t k_node_id = 0;

// Operation types that are memory copies. rocpd_string may repeat a text under
// several ids, so the set is selected by text.
constexpr std::string_view k_copy_op_types =
    "(SELECT id FROM rocpd_string WHERE string LIKE 'Copy%')";

struct api_row_t
{
    size_t                id{};
    size_t                start{};
    size_t                end{};
    std::optional<size_t> name_id;
    size_t                pid{};
    size_t                tid{};
};

struct op_row_t
{
    size_t                id{};
    size_t                start{};
    size_t                end{};
    std::optional<size_t> description_id;
    std::optional<size_t> op_type_id;
    size_t                gpu_id{};
    size_t                queue_id{};
    size_t                is_copy{};
};

// Columns in op_row_t order for rocpd_op aliased as O.
[[nodiscard]] std::string op_columns();

// Columns in api_row_t order for rocpd_api aliased as A.
constexpr std::string_view k_api_columns =
    "A.id, A.start, A.end, A.apiName_id, A.pid, A.tid";

[[nodiscard]] std::string op_kind_where(std::string_view alias, bool is_copy);

[[nodiscard]] std::string quote_text(std::string_view text);

// rpd stores addresses as hexadecimal text such as "0x7fb07c298000".
[[nodiscard]] std::optional<size_t> parse_address(const std::optional<std::string>& text);
}  // namespace rocpd_v2

/**
 * @brief Reader for rocpd schema 2, the rocmProfileData (.rpd) layout.
 *
 * Host API calls are rows of rocpd_api keyed by OS pid and tid. GPU operations
 * are rows of rocpd_op keyed by gpuId and queueId, and rocpd_api_ops links the
 * two. Monitor samples are rows of rocpd_monitor keyed by deviceId and
 * monitorType. The schema has no node, process, thread, or agent tables, so
 * the catalog synthesizes them from those keys:
 *   - one node, id 0, named after the file;
 *   - a process per pid, whose row id is the pid;
 *   - a thread per (pid, tid), numbered from 1;
 *   - a GPU agent per gpuId or positive deviceId, whose row id is that id;
 *   - a counter per (deviceId, monitorType), numbered from 1.
 * Operations whose type starts with "Copy" are memory copies. Every other
 * operation is a kernel dispatch.
 */
class rocpd_v2_reader final : public reader_t::impl
{
public:
    rocpd_v2_reader(std::unique_ptr<profiler_hub::storage_t> storage,
                    std::shared_ptr<reader_catalog_t>        catalog);

    void build_catalog_category(catalog_category_t category,
                                reader_catalog_t&  catalog) override;
    void ensure_track_topology_indexes() override;

    // Timeline event queries
    [[nodiscard]] reader_types::timeline_event_list_t get_events(
        const reader_types::event_filter_t& filter) override;

    [[nodiscard]] reader_types::timeline_event_list_t get_events_for_track(
        reader_types::track_info_ptr_t      track,
        const reader_types::event_filter_t& filter) override;

    [[nodiscard]] std::optional<std::pair<size_t, size_t>> get_event_id_span(
        reader_types::event_type_t type) override;

    void visit_track_events_in_id_range(const reader_types::track_info_ptr_t& track,
                                        reader_types::event_type_t            type,
                                        size_t                                id_begin,
                                        size_t                                id_end,
                                        reader_t::event_visitor_t             visitor,
                                        void* context) override;

    [[nodiscard]] reader_types::counter_timeline_event_list_t
    get_counter_events_for_track(reader_types::track_info_ptr_t      track,
                                 const reader_types::event_filter_t& filter) override;

    [[nodiscard]] size_t get_event_count(
        const reader_types::event_filter_t& filter) override;

    [[nodiscard]] reader_types::histogram_t get_histogram(size_t max_buckets) override;
    [[nodiscard]] reader_types::histogram_t get_histogram(
        const histogram_grid_t& grid) override;

    // Event detail queries
    [[nodiscard]] std::optional<reader_types::region_data_t> get_region_details(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] std::optional<reader_types::kernel_dispatch_data_t>
    get_kernel_dispatch_details(const reader_types::timeline_event_t& event) override;

    [[nodiscard]] std::optional<reader_types::memory_copy_data_t> get_memory_copy_details(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] reader_types::track_info_ptr_t track_of(
        const reader_types::timeline_event_t& event) override;

    // Event property queries
    [[nodiscard]] reader_types::call_stack_t get_call_stack(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] reader_types::arg_data_list_t get_arguments(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] reader_types::timeline_event_list_t get_correlated_events(
        const reader_types::timeline_event_t& event) override;

    // Database metadata
    [[nodiscard]] reader_types::time_window_t  get_data_time_range() override;
    [[nodiscard]] reader_types::event_counts_t get_event_counts(
        const reader_types::time_window_t& window) override;

    [[nodiscard]] uint32_t read_table_events(const resolved_event_query& query,
                                             uint32_t                    event_type,
                                             uint64_t                    after_id,
                                             uint64_t                    last_id,
                                             ph_table_event*             rows,
                                             uint32_t                    capacity) override;
    [[nodiscard]] bool     table_event_id_span(uint32_t  event_type,
                                               uint64_t& first_id,
                                               uint64_t& last_id) override;
    [[nodiscard]] bool     summarize_name_durations(
        const resolved_event_query&           query,
        uint32_t                              event_type,
        std::vector<name_duration_summary_t>& rows) override;

private:
    void build_string_list(reader_catalog_t& catalog);
    void build_nodes(reader_catalog_t& catalog);
    void build_processes(reader_catalog_t& catalog);
    void build_threads(reader_catalog_t& catalog);
    void build_agents(reader_catalog_t& catalog);
    void build_tracks(reader_catalog_t& catalog);
    void build_queues(reader_catalog_t& catalog);
    void build_pmc_infos(reader_catalog_t& catalog);

    [[nodiscard]] reader_types::timeline_event_t region_event(
        const rocpd_v2::api_row_t& row) const;
    [[nodiscard]] reader_types::timeline_event_t op_event(
        const rocpd_v2::op_row_t& row) const;

    // Row of the first host call linked to an operation, or none.
    [[nodiscard]] std::optional<rocpd_v2::api_row_t> launching_api(size_t op_id);

    // Columns shared by the detail and table queries: grid and workgroup
    // sizes, segment sizes, and the kernel name from rocpd_kernelapi K.
    [[nodiscard]] std::string kernel_api_columns() const;
    // Size and addresses from rocpd_copyapi C.
    [[nodiscard]] std::string copy_api_columns() const;
    // LEFT JOINs from rocpd_op O to its first linked host call A and that
    // call's kernel and copy arguments K and C.
    [[nodiscard]] std::string op_api_joins() const;

    [[nodiscard]] reader_types::track_info_ptr_t thread_track(size_t pid,
                                                              size_t tid) const;
    [[nodiscard]] reader_types::track_info_ptr_t op_track(bool   is_copy,
                                                          size_t gpu_id,
                                                          size_t queue_id) const;
    [[nodiscard]] reader_types::thread_info_ptr_t thread_info(size_t pid,
                                                              size_t tid) const;
    [[nodiscard]] reader_types::process_info_ptr_t process_info(size_t pid) const;
    [[nodiscard]] reader_types::agent_info_ptr_t   agent_info(size_t gpu_id) const;

    [[nodiscard]] std::string_view text_of(std::optional<size_t> string_id) const;
    // An operation without a description is named by its type.
    [[nodiscard]] std::optional<size_t> op_name_id(std::optional<size_t> description_id,
                                                   std::optional<size_t> op_type_id) const;
    [[nodiscard]] size_t canonical_string_id(size_t string_id) const;

    [[nodiscard]] bool has_table(std::string_view name);

    std::shared_ptr<data_storage::sqlite_backend> m_backend;
    bool                                          m_has_kernel_api{};
    bool                                          m_has_copy_api{};
};

[[nodiscard]] std::unique_ptr<reader_t::impl> make_rocpd_v2_reader(
    std::unique_ptr<profiler_hub::storage_t> storage,
    std::shared_ptr<reader_catalog_t>        catalog);

[[nodiscard]] std::unique_ptr<trim_schema_t> make_rocpd_v2_trim_schema();

}  // namespace profiler_hub
