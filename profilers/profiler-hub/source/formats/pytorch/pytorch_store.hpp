// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "kineto_loader.hpp"
#include "profiler-hub/cpp/reader_types.hpp"

#include <cstdint>
#include <functional>
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
    std::uint32_t              event{};  // index into pytorch_store_t::events
};

struct store_track_t
{
    std::uint32_t              id{};
    std::string                name;
    reader_types::track_kind_t category{};
    std::int64_t               pid{};
    std::int64_t               tid{};
    std::uint64_t              agent_id{};
    std::uint64_t              stream_id{};
    std::uint64_t              start_ts{};
    std::uint64_t              end_ts{};
    std::vector<store_row_t>   rows;  // start ascending, then end descending
};

// A parsed Kineto trace as per-track rows, and the events and args text the rows point
// at.
struct pytorch_store_t
{
    std::vector<store_track_t>   tracks;
    std::vector<event_t>         events;  // file order
    std::vector<memory_sample_t> memory;  // by event index
    std::uint64_t
        cpu_agent_id{};  // set by fill_catalog; 0 when no memory event is on the CPU
    std::string                                                  args;
    std::vector<std::string>                                     names;
    std::vector<std::string>                                     cats;
    std::map<std::pair<std::int64_t, std::int64_t>, std::string> thread_names;
    std::map<std::int64_t, std::string>                          process_names;
    std::uint64_t                                                parse_ms{};
    std::uint64_t                                                derive_ms{};
    load_diag_t                                                  diag{};
};

// @p cancelled is polled while the file is read; see load_kineto.
pytorch_store_t
load_store(const std::string& path, std::function<bool()> cancelled = {});

}  // namespace profiler_hub::pytorch
