// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "category_class.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace profiler_hub::pytorch
{
struct event_t
{
    std::uint32_t name_id{};
    std::uint32_t cat_id{};
    std::int64_t  pid{};
    std::int64_t  tid{};
    std::int64_t  stream{};
    std::int64_t  device{};
    std::int64_t  ts_ns{};
    std::int64_t  dur_ns{};
    std::int64_t  corr{ -1 };  // args "correlation"
    std::int64_t  ext{ -1 };   // args "External id"
    std::uint64_t args_off{};  // the args object's text in loaded_trace_t::args
    std::uint32_t args_len{};  // 0 when the event has no args or an empty object
};
static_assert(sizeof(event_t) == 88, "event_t is the 88-byte layout");

struct load_diag_t
{
    std::uint64_t total{};
    std::uint64_t x_events{};
    std::uint64_t flow_s{};
    std::uint64_t flow_f{};
    std::uint64_t meta_events{};
    std::uint64_t other_events{};
    std::uint64_t bad_events{};
    std::uint64_t clamped_events{};
    std::uint64_t gunzip_ms{};
    std::uint64_t parse_ms{};
};

struct loaded_trace_t
{
    std::uint64_t                 base_time_ns{};
    std::int32_t                  schema_version{};
    std::vector<std::string>      names;
    std::vector<std::string>      cats;
    std::vector<category_class_t> cat_classes;
    std::vector<event_t>          events;
    std::string                   args;
    bool                          link_by_correlation{ true };
    std::map<std::pair<std::int64_t, std::int64_t>, std::string> thread_names;
    std::map<std::int64_t, std::string>                          process_names;
    std::map<std::int64_t, std::string>                          device_names;
    std::map<std::int64_t, std::string>                          lane_names;
    bool                                                         gzip{};
    load_diag_t                                                  diag{};
};

// Reads a plain or gzipped Kineto JSON file into memory. @p cancelled is polled before
// each chunk of input is read; once it returns true the call throws operation_cancelled.
loaded_trace_t
load_kineto(const std::string& path, std::function<bool()> cancelled = {});
}  // namespace profiler_hub::pytorch
