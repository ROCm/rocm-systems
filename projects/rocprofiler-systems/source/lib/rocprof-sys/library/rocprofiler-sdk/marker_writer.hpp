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
// All logic (range tracking, operation handling) belongs in roctx_client.
class marker_writer
{
public:
    marker_writer();
    ~marker_writer() = default;

    marker_writer(const marker_writer&)            = delete;
    marker_writer& operator=(const marker_writer&) = delete;
    marker_writer(marker_writer&&)                 = default;
    marker_writer& operator=(marker_writer&&)      = default;

    void write_begin(std::string_view name) const;

    void write_end(std::string_view name, uint64_t begin_ts, uint64_t end_ts,
                   std::string& args, rocprofiler_callback_tracing_record_t record) const;

private:
    bool m_use_perfetto{ false };
    bool m_use_timemory{ false };
};

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
