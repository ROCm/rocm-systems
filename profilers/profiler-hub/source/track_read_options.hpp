// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

namespace profiler_hub
{

/** Tuning of the whole-track read path. */
struct track_read_options
{
    /** Thread tracks with at least this many events are read in parallel. */
    size_t parallel_read_min_events{ 1000000 };
    /** Number of id ranges, and so at most of helper threads, of a parallel read. */
    size_t parallel_read_parts{ 8 };
};

}  // namespace profiler_hub
