// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "host_tree.hpp"
#include "kineto_loader.hpp"
#include "profiler-hub/cpp/reader_types.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace profiler_hub::pytorch
{

struct store_row_t
{
    std::uint64_t              start{};
    std::uint64_t              end{};
    std::uint32_t              name_id{};
    reader_types::event_type_t type{};
};

struct store_track_t
{
    std::uint32_t              id{};
    std::string                name;
    reader_types::track_kind_t category{};
    std::int32_t               pid{};
    std::int32_t               tid{};
    std::uint32_t              agent_id{};
    std::uint32_t              stream_id{};
    std::uint64_t              start_ts{};
    std::uint64_t              end_ts{};
    std::vector<store_row_t>   rows;  // start ascending, then end descending
    tree_stats_t               tree{};
};

// A parsed Kineto trace as per-track rows. The rows are only held until profiler-hub's
// reader has cached every track.
struct pytorch_store_t
{
    std::vector<store_track_t>                                   tracks;
    std::vector<std::string>                                     names;
    std::vector<std::string>                                     cats;
    std::map<std::pair<std::int32_t, std::int32_t>, std::string> thread_names;
    std::map<std::int32_t, std::string>                          process_names;
    std::uint64_t                                                parse_ms{};
    std::uint64_t                                                derive_ms{};
};

pytorch_store_t
load_store(const std::string& path);

}  // namespace profiler_hub::pytorch
