#pragma once

#include <cstddef>
#include <functional>

namespace profiler_hub
{

/**
 * Parses a plain unsigned decimal number.
 * @return @p fallback if @p text is null, empty, not a number, has trailing
 *         characters or does not fit in size_t.
 */
[[nodiscard]] size_t
parse_size(const char* text, size_t fallback) noexcept;

/** Tuning knobs of the whole-track read path. */
struct track_read_options
{
    using env_lookup_t = std::function<const char*(const char*)>;

    /** Thread tracks with at least this many events are read in parallel. */
    size_t parallel_read_min_events{ 1000000 };
    /** Number of id ranges (and at most helper threads) of a parallel read; at least 1.
     */
    size_t parallel_read_parts{ 8 };

    /** Reads PH_READ_MIN_EVENTS and PH_READ_PARTS, keeping the defaults for invalid
     * values. */
    [[nodiscard]] static track_read_options from_env(const env_lookup_t& lookup);
};

}  // namespace profiler_hub
