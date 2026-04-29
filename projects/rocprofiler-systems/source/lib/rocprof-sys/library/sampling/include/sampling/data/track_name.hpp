// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>

namespace rocprofsys::sampling
{

// Per-thread sampling track name builders. Single source of truth for the
// strings registered in trace_cache metadata, written into rocpd, and emitted
// to perfetto. NFR-P-3 / NFR-P-6: byte-equivalent to the legacy strings.

struct timer_track_tag
{};
struct overflow_track_tag
{};

inline std::string
make_thread_track_name(timer_track_tag, std::size_t seq_id, std::size_t sys_id)
{
    return "Thread " + std::to_string(seq_id) + " Timer (S) " + std::to_string(sys_id);
}

inline std::string
make_thread_track_name(overflow_track_tag, std::size_t seq_id, std::size_t sys_id)
{
    return "Thread " + std::to_string(seq_id) + " Overflow (S) " + std::to_string(sys_id);
}

}  // namespace rocprofsys::sampling
