// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "format_source.hpp"
#include "pytorch_store.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace profiler_hub::pytorch
{

/**
 * A Kineto trace parsed into memory in load_catalog and kept there. Event ids are dense
 * per type and numbered in track order from 1, so each (track, type) owns one contiguous
 * id run.
 */
class pytorch_source final : public format_source
{
public:
    explicit pytorch_source(const source_request_t& request);

    [[nodiscard]] const std::shared_ptr<reader_catalog_t>& catalog() const override;
    void                             load_catalog(common::thread_pool&   workers,
                                                  const catalog_cache_t& cache) override;
    void                             attach_reader(cached_track_reader& reader) override;
    [[nodiscard]] connection_source& track_reads() override;

    [[nodiscard]] reader_types::track_info_ptr_t track_of(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] reader_types::histogram_t histogram(
        const histogram_grid_t& grid) override;

    [[nodiscard]] reader_types::timeline_event_list_t correlated_events(
        const reader_types::timeline_event_t& event) override;
    [[nodiscard]] reader_types::call_stack_t call_stack(
        const reader_types::timeline_event_t& event) override;
    [[nodiscard]] std::optional<reader_types::region_data_t> region_details(
        const reader_types::timeline_event_t& event) override;
    [[nodiscard]] std::optional<reader_types::kernel_dispatch_data_t>
    kernel_dispatch_details(const reader_types::timeline_event_t& event) override;
    [[nodiscard]] std::optional<reader_types::memory_copy_data_t> memory_copy_details(
        const reader_types::timeline_event_t& event) override;
    [[nodiscard]] std::optional<reader_types::memory_alloc_data_t> memory_alloc_details(
        const reader_types::timeline_event_t& event) override;

    [[nodiscard]] bool summarize_name_durations(
        const resolved_event_query&           query,
        uint32_t                              event_type,
        std::vector<name_duration_summary_t>& rows) override;
    [[nodiscard]] bool     supports_table_events() const override { return true; }
    [[nodiscard]] uint32_t read_table_events(const resolved_event_query& query,
                                             uint32_t                    event_type,
                                             uint64_t                    after_id,
                                             uint64_t                    last_id,
                                             ph_table_event*             rows,
                                             uint32_t capacity) override;
    [[nodiscard]] bool     table_event_id_span(uint32_t  event_type,
                                               uint64_t& first_id,
                                               uint64_t& last_id) override;

private:
    friend class pytorch_row_reader;

    class row_seam final : public connection_source
    {
    public:
        explicit row_seam(pytorch_source& owner)
        : m_owner{ owner }
        {}

        [[nodiscard]] std::unique_ptr<track_row_reader> acquire() override;
        [[nodiscard]] std::unique_ptr<track_row_reader> try_acquire() override;

    private:
        pytorch_source& m_owner;
    };

    static constexpr std::size_t k_type_slots =
        static_cast<std::size_t>(reader_types::event_type_t::hip_event) + 1;

    struct type_run_t
    {
        std::size_t                first_id{};
        std::vector<std::uint32_t> rows;  // row index of each id in the run
    };

    struct track_rows_t
    {
        std::vector<store_row_t>             rows;
        std::array<type_run_t, k_type_slots> by_type;
    };

    // The ids [first_id, first_id + count) of one type on one track.
    struct run_t
    {
        reader_types::event_type_t type{};
        std::size_t                first_id{};
        std::size_t                count{};
        std::uint32_t              track{};
    };

    [[nodiscard]] const track_rows_t& track_rows(
        const reader_types::track_info_t& track) const;

    // Where an event's row is: its track and its place in that track's rows.
    struct located_t
    {
        std::size_t        track{};
        const store_row_t* row{};
        std::size_t        id{};
    };

    // One event that has a correlation, for the join correlated_events makes.
    struct correlation_t
    {
        std::int64_t link{};
        std::size_t  id{};
        std::uint8_t type{};
    };

    // A kernel's launch geometry as rocpd keeps it: both sizes in work-items.
    struct kernel_shape_t
    {
        std::array<std::uint64_t, 3> workgroup{};
        std::array<std::uint64_t, 3> grid{};
        std::optional<std::uint64_t> group_segment;
    };

    [[nodiscard]] kernel_shape_t           kernel_shape_of(const event_t& stored) const;
    [[nodiscard]] std::optional<located_t> locate(
        const reader_types::unique_timeline_event_id_t& id) const;
    [[nodiscard]] std::string_view               args_of(const event_t& event) const;
    [[nodiscard]] std::string                    extdata_of(const store_row_t& row) const;
    [[nodiscard]] reader_types::event_data_ptr_t event_data_of(
        const store_row_t& row) const;
    [[nodiscard]] reader_types::timeline_event_t timeline_event_of(
        const located_t&           at,
        reader_types::event_type_t type) const;
    [[nodiscard]] reader_types::agent_info_ptr_t agent_info(std::uint64_t agent) const;

    // The destination agent of a copy is its device, except for a copy back to the host.
    [[nodiscard]] std::uint64_t copy_agent_of(const located_t& at) const;
    [[nodiscard]] bool          table_match(const resolved_event_query& query,
                                            reader_types::event_type_t  type,
                                            const located_t&            at) const;
    void                        publish_row(reader_types::event_type_t type,
                                            const located_t&           at,
                                            ph_table_event&            out) const;

    // The agent a memory event is on: the CPU agent, a GPU's, or 0 when its device is
    // unknown.
    [[nodiscard]] std::uint64_t          agent_of(const memory_sample_t& sample) const;
    [[nodiscard]] const memory_sample_t* sample_of(std::uint32_t event) const;
    void                                 add_memory_tracks();

    std::string                       m_path;
    std::shared_ptr<reader_catalog_t> m_catalog;
    row_seam                          m_seam;
    std::vector<track_rows_t>         m_tracks;  // by track id
    std::vector<event_t>          m_events;  // file order; store_row_t::event indexes it
    std::string                   m_args;    // the args text of m_events
    std::vector<memory_sample_t>  m_memory;  // by event index
    std::vector<bound_flow_t>     m_flows;   // by event index
    std::vector<category_class_t> m_cat_classes;   // by category index
    std::vector<correlation_t>    m_correlations;  // by link, then type, then id
    std::vector<reader_types::stream_info_ptr_t>
                                  m_streams;  // by track id; null off stream tracks
    std::uint64_t                 m_cpu_agent_id{};
    std::size_t                   m_first_name_id{};  // the string id of name id 0
    std::vector<std::string_view> m_names;            // by name id
    std::vector<std::string_view> m_cats;             // by category index
    std::vector<run_t>            m_runs;             // by type, then first id
    cached_track_reader*          m_reader{};
};

}  // namespace profiler_hub::pytorch
