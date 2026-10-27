// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <ratio>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
using wall_clock_nanoseconds = std::chrono::duration<std::uint64_t, std::nano>;

/// A timing unit of the report, named like its std::chrono literal, e.g. { "ms", 1ms }.
struct wall_clock_unit
{
    std::string_view         name;
    std::chrono::nanoseconds length;
};

inline constexpr wall_clock_unit k_wall_clock_seconds{ "s", std::chrono::seconds{ 1 } };
inline constexpr std::int32_t    k_wall_clock_max_width = 120;
inline constexpr std::int16_t    k_wall_clock_precision = 6;

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
    std::int32_t    rank      = 0;
    std::int32_t    mpi_size  = 1;
};

struct wall_clock_interval
{
    std::uint64_t          thread_id;
    std::string            name;
    std::uint64_t          correlation_id;
    wall_clock_nanoseconds begin;
    wall_clock_nanoseconds end;
};

/**
 * Call-tree report of wall-clock time in the layout of the timemory wall_clock text and
 * JSON files. Regions nest by time containment per thread; GPU activity (kernel, memory
 * copy, scratch memory) nests under the region with the same correlation id, or sits at
 * the top of its thread if none does. Collapsing threads merges every thread into one
 * tree whose rows carry no thread number.
 */
class wall_clock_report
{
public:
    explicit wall_clock_report(wall_clock_settings settings) noexcept;

    /**
     * Adds a CPU region; regions nest by time containment per thread.
     * @throws std::runtime_error If the region ends before it begins.
     */
    void add_region(wall_clock_interval region);

    /**
     * Adds a kernel, memory copy or scratch memory interval.
     * @throws std::runtime_error If the interval ends before it begins.
     */
    void add_gpu_activity(wall_clock_interval activity);

    [[nodiscard]] bool empty() const noexcept;

    /// The table of the timemory wall_clock text file.
    [[nodiscard("the table is the report")]] std::string text() const;

    /// The document of the timemory wall_clock JSON file.
    [[nodiscard("the document is the report")]] std::string json() const;

private:
    wall_clock_settings              m_settings;
    std::vector<wall_clock_interval> m_regions;
    std::vector<wall_clock_interval> m_gpu_activity;
};

}  // namespace rocprofsys
