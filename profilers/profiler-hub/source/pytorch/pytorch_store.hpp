// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "host_tree.hpp"
#include "kineto_loader.hpp"
#include "profiler-hub/c/profiler_hub_types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace profiler_hub
{
struct reader_catalog_t;
}

namespace profiler_hub::pytorch
{
struct store_track_t
{
    std::uint32_t             id{};
    std::string               name;
    ph_track_category_t       category{};
    std::uint32_t             nid{};
    std::uint32_t             pid{};
    std::uint32_t             tid{};
    std::uint32_t             agent_id{};
    std::uint32_t             queue_id{};
    std::uint32_t             stream_id{};
    std::uint64_t             start_ts{};
    std::uint64_t             end_ts{};
    std::vector<ph_event_t>   events;
    std::vector<std::uint8_t> category_ids;
};

class pytorch_store_t
{
public:
    static std::shared_ptr<const pytorch_store_t> open(const std::string& path);

    const std::vector<store_track_t>& tracks() const { return m_tracks; }
    bool has_track(std::uint32_t id) const { return id >= 1 && id <= m_tracks.size(); }

    std::shared_ptr<reader_catalog_t> catalog() const { return m_catalog; }

    const std::vector<ph_event_t>& track_events(std::uint32_t id) const;

    std::vector<ph_event_t> query(std::uint32_t id,
                                  std::uint64_t start_ts,
                                  std::uint64_t end_ts) const;

    const load_diag_t&  load_diag() const { return m_diag; }
    const tree_stats_t& tree_stats(std::uint32_t host_track_id) const;

    std::string summary_line() const;

private:
    pytorch_store_t() = default;
    void derive(loaded_trace_t& t);

    std::string                m_path;
    bool                       m_gzip           = false;
    std::uint64_t              m_base_time_ns   = 0;
    std::int32_t               m_schema_version = 0;
    std::size_t                m_device_count   = 0;
    load_diag_t                m_diag{};
    std::vector<store_track_t> m_tracks;
    std::vector<tree_stats_t>  m_tree_stats;
    std::uint64_t              m_clamped_negative_dur = 0;
    std::uint64_t              m_gpu_events{};
    std::uint64_t              m_gpu_unmatched{};
    std::uint64_t              m_pid_fallback_tracks = 0;
    std::uint64_t              m_derive_ms{};
    std::uint64_t              m_total_ms{};
    std::uint64_t              m_peak_rss_kb = 0;

    std::shared_ptr<reader_catalog_t> m_catalog;
};
}  // namespace profiler_hub::pytorch
