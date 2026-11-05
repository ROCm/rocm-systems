// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "connection_source.hpp"
#include "profiler-hub/cpp/reader.hpp"
#include "pytorch_store.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace profiler_hub::pytorch
{

/**
 * Serves a parsed Kineto trace to cached_track_reader in place of SQLite connections.
 * Event ids are dense per type and numbered in track order, so each (track, type) owns
 * one contiguous id run. Names point into the catalog's strings. Once release_rows() has
 * run, every read throws: the reader must already hold every track by then.
 */
class memory_source final : public connection_source
{
public:
    /** Parses @p path and fills @p catalog, which must outlive the source. */
    [[nodiscard]] static std::unique_ptr<memory_source> open(const std::string& path,
                                                             reader_catalog_t&  catalog);

    [[nodiscard]] std::unique_ptr<track_row_reader> acquire() override;
    [[nodiscard]] std::unique_ptr<track_row_reader> try_acquire() override;

    void release_rows();

private:
    friend class memory_row_reader;

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

    memory_source() = default;

    [[nodiscard]] const track_rows_t& track_rows(
        const reader_types::track_info_t& track) const;

    std::vector<track_rows_t> m_tracks;  // by track id
    std::vector<const char*>  m_names;   // by name id
    std::atomic<bool>         m_released{ false };
};

}  // namespace profiler_hub::pytorch
