// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofiler-systems/categories.h"

#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/data/track_name.hpp"
#include <array>

namespace rocprofsys::sampling
{

// Per-track-tag traits — single source of truth for category id, category
// string, and the sample type associated with each tag. Drives the
// template-method dispatch in real_trace_cache_sink and real_perfetto_sink.

template <class Tag>
struct track_traits
{
    static_assert(sizeof(Tag) == 0,
                  "track_traits is undefined for this tag — provide an explicit "
                  "specialisation of track_traits<Tag> alongside its track-name "
                  "tag struct in sampling/data/track_name.hpp");
};

template <>
struct track_traits<timer_track_tag>
{
    using sample_t                  = timer_sample;
    static constexpr auto category  = ROCPROFSYS_CATEGORY_TIMER_SAMPLING;
    static constexpr auto label_str = "timer_sampling";
};

template <>
struct track_traits<overflow_track_tag>
{
    using sample_t                  = overflow_sample;
    static constexpr auto category  = ROCPROFSYS_CATEGORY_OVERFLOW_SAMPLING;
    static constexpr auto label_str = "overflow_sampling";
};

// Per-thread counter track name prefixes. Used by both trace_cache_sink
// (emission) and do_setup_wiring (track registration) so they cannot drift.
inline constexpr std::array<char const*, 4> thread_counter_prefixes = {
    "thread_cpu_time", "thread_peak_memory", "thread_context_switch", "thread_page_fault"
};

}  // namespace rocprofsys::sampling
