// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocprofiler-sdk/callback_tracing.h>

#include <cstdint>
#include <string_view>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

// Thin output layer for writing marker data to Perfetto, timemory, and cache.
// No business logic - just outputs to configured destinations.
// All logic (range tracking, operation handling) belongs in marker_client.
class marker_writer
{
public:
    marker_writer()  = default;
    ~marker_writer() = default;

    marker_writer(const marker_writer&)            = delete;
    marker_writer& operator=(const marker_writer&) = delete;
    marker_writer(marker_writer&&)                 = default;
    marker_writer& operator=(marker_writer&&)      = default;

    // roctxRangePush - ENTER phase
    void write_push_start(std::string_view name);

    // roctxRangePop - EXIT phase
    void write_pop(std::string_view name, uint64_t begin_ts, uint64_t end_ts,
                   rocprofiler_callback_tracing_record_t record);

    // roctxRangeStart (process-wide) - ENTER phase
    void write_range_start(std::string_view name);

    // roctxRangeStop - EXIT phase
    void write_range_stop(std::string_view name, uint64_t begin_ts, uint64_t end_ts,
                          rocprofiler_callback_tracing_record_t record);

    // roctxMark - instant marker
    void write_mark(std::string_view name, uint64_t ts,
                    rocprofiler_callback_tracing_record_t record);
};

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
