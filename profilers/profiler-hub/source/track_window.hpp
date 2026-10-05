#pragma once

#include "profiler-hub/cpp/reader_types.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace profiler_hub
{

/**
 * Builds the reader filter for a C API time window. Both bounds 0 means no
 * filter; only an end of 0 means no upper bound. Bounds above the largest
 * signed 64-bit value are clamped, because the database binds them as signed.
 */
[[nodiscard]] inline reader_types::event_filter_t
make_window_filter(uint64_t start_ts, uint64_t end_ts)
{
    using bound_t = reader_types::timestamp_ns_t;

    constexpr auto max_bound =
        static_cast<bound_t>(std::numeric_limits<std::int64_t>::max());

    reader_types::event_filter_t filter;
    if(start_ts != 0 || end_ts != 0)
    {
        filter.time_window.start = std::min<bound_t>(start_ts, max_bound);
        filter.time_window.end =
            (end_ts != 0) ? std::min<bound_t>(end_ts, max_bound) : max_bound;
    }
    return filter;
}

}  // namespace profiler_hub
