// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "category_class.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace profiler_hub::pytorch
{
struct event_t
{
    std::uint32_t name_id{};
    std::uint8_t  cat_id{};
    std::int32_t  pid{};
    std::int32_t  tid{};
    std::int32_t  stream{};
    std::int32_t  device{};
    std::int64_t  ts_ns{};
    std::int64_t  dur_ns{};
    std::int64_t  correlation{};
};
static_assert(sizeof(event_t) == 48, "event_t is the measured 48-byte layout");

struct load_diag_t
{
    std::uint64_t total{};
    std::uint64_t x_events{};
    std::uint64_t flow_s{};
    std::uint64_t flow_f{};
    std::uint64_t meta_events{};
    std::uint64_t other_events{};
    std::uint64_t bad_events{};
    std::uint64_t gunzip_ms{};
    std::uint64_t parse_ms{};
};

struct loaded_trace_t
{
    std::uint64_t                                                base_time_ns{};
    std::int32_t                                                 schema_version{};
    std::vector<std::string>                                     names;
    std::vector<std::string>                                     cats;
    std::vector<category_class_t>                                cat_classes;
    std::vector<event_t>                                         events;
    std::map<std::pair<std::int32_t, std::int32_t>, std::string> thread_names;
    std::map<std::int32_t, std::string>                          process_names;
    std::vector<std::string>                                     device_names;
    bool                                                         gzip{};
    load_diag_t                                                  diag{};
};

loaded_trace_t
load_kineto(const std::string& path);
}  // namespace profiler_hub::pytorch
