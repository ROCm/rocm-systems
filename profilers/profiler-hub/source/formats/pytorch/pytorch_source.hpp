// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "format_source.hpp"
#include "pytorch_store.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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

    std::string                       m_path;
    std::shared_ptr<reader_catalog_t> m_catalog;
    row_seam                          m_seam;
    std::vector<track_rows_t>         m_tracks;  // by track id
    std::vector<event_t>     m_events;  // file order; store_row_t::event indexes it
    std::string              m_args;    // the args text of m_events
    std::vector<const char*> m_names;   // by name id
    std::vector<run_t>       m_runs;    // by type, then first id
    cached_track_reader*     m_reader{};
};

}  // namespace profiler_hub::pytorch
