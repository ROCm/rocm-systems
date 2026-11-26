// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <ratio>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
using wall_clock_nanoseconds = std::chrono::duration<std::uint64_t, std::nano>;

/// A timing unit of the report, named as timemory prints it, e.g. { "msec", 1ms }.
struct wall_clock_unit
{
    std::string_view         name;
    std::chrono::nanoseconds length;
};

inline constexpr wall_clock_unit k_wall_clock_seconds{ .name = "sec",
                                                       .length =
                                                           std::chrono::seconds{ 1 } };
inline constexpr std::int32_t    k_wall_clock_max_width = 120;
inline constexpr std::int16_t    k_wall_clock_precision = 6;
inline constexpr std::uint64_t   k_no_correlation       = 0;

/// Looks up a ROCPROFSYS_TIMING_UNITS value by the names timemory accepts, in any case.
[[nodiscard("an unknown unit must be reported")]] std::optional<wall_clock_unit>
find_wall_clock_unit(std::string_view name);

/// Shape and formatting of the report; the defaults are the profile defaults.
struct wall_clock_settings
{
    bool          flat             = false;
    bool          timeline         = false;
    bool          collapse_threads = false;
    std::uint16_t max_depth        = std::numeric_limits<std::uint16_t>::max();
    // Width of the text labels; 0 or less does not cut them
    std::int32_t    max_width = k_wall_clock_max_width;
    wall_clock_unit unit      = k_wall_clock_seconds;
    std::int16_t    precision = k_wall_clock_precision;
    std::uint64_t   rank      = 0;
    std::uint64_t   mpi_size  = 1;
};

/// A span of time on one thread. A `correlation_id` of k_no_correlation means the span
/// is not tied to an API call; `begin` and `end` share one clock.
struct wall_clock_interval
{
    std::uint64_t          thread_id;
    std::string            name;
    std::uint64_t          correlation_id;
    wall_clock_nanoseconds begin;
    wall_clock_nanoseconds end;
};

namespace wall_clock_detail
{
template <typename Key>
class intern_table
{
public:
    std::uint32_t intern(const Key& key)
    {
        auto const [entry, inserted] =
            m_ids.try_emplace(key, static_cast<std::uint32_t>(m_keys.size()));
        if(inserted)
        {
            m_keys.push_back(key);
        }
        return entry->second;
    }

    [[nodiscard]] const Key& at(std::uint32_t index) const { return m_keys[index]; }

private:
    std::unordered_map<Key, std::uint32_t> m_ids;
    std::vector<Key>                       m_keys;
};

struct stored_interval
{
    wall_clock_nanoseconds begin;
    wall_clock_nanoseconds end;
    std::uint64_t          correlation_id;
    std::uint64_t          thread_id;
    std::uint32_t          name;
};

struct interval_store
{
    intern_table<std::string>   names;
    std::deque<stored_interval> regions;
    std::deque<stored_interval> gpu_activity;

    [[nodiscard]] stored_interval compact(const wall_clock_interval& interval)
    {
        return { .begin          = interval.begin,
                 .end            = interval.end,
                 .correlation_id = interval.correlation_id,
                 .thread_id      = interval.thread_id,
                 .name           = names.intern(interval.name) };
    }
};
}  // namespace wall_clock_detail

/**
 * Call-tree report of wall-clock time in the layout of the timemory wall_clock text and
 * JSON files. Regions nest by time containment per thread; GPU activity (kernel, memory
 * copy, scratch memory) nests under the region with the same correlation id, or sits at
 * the top of its thread if none does. Collapsing threads merges every thread into one
 * tree whose rows carry no thread number. Intervals are kept until rendering because a
 * region's ancestors are only known once they end.
 */
class wall_clock_report
{
public:
    explicit wall_clock_report(wall_clock_settings settings);

    /**
     * Adds a CPU region; regions nest by time containment per thread.
     * @throws std::runtime_error If the region ends before it begins.
     */
    void add_region(const wall_clock_interval& region);

    /**
     * Adds a kernel, memory copy or scratch memory interval.
     * @throws std::runtime_error If the interval ends before it begins.
     */
    void add_gpu_activity(const wall_clock_interval& activity);

    [[nodiscard]] bool empty() const noexcept;

    /// The table of the timemory wall_clock text file.
    [[nodiscard("the table is the report")]] std::string text() const;

    /// The document of the timemory wall_clock JSON file.
    [[nodiscard("the document is the report")]] std::string json() const;

private:
    wall_clock_settings               m_settings;
    wall_clock_detail::interval_store m_intervals;
};

}  // namespace rocprofsys
