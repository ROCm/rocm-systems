// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <ratio>
#include <string>
#include <vector>

namespace rocprofsys::trace_cache
{
struct wall_clock_layout
{
    bool flat     = false;
    bool timeline = false;
};

struct wall_clock_thread_trees;

using wall_clock_nanoseconds = std::chrono::duration<std::uint64_t, std::nano>;

struct wall_clock_interval
{
    std::uint64_t          thread_id;
    std::string            name;
    std::uint64_t          correlation_id;
    wall_clock_nanoseconds begin;
    wall_clock_nanoseconds end;
};

/**
 * Call-tree report of wall-clock time in the layout of the timemory wall_clock text
 * file. Regions nest by time containment per thread; GPU activity (kernel, memory copy,
 * scratch memory) nests under the region with the same correlation id, or sits at the
 * top of its thread if none does. The tree is built once, on the first write; the
 * intervals are released afterwards, so nothing may be added then.
 */
class wall_clock_report
{
public:
    explicit wall_clock_report(wall_clock_layout layout);

    void add_region(wall_clock_interval region);

    void add_gpu_activity(wall_clock_interval activity);

    [[nodiscard]] bool empty() const noexcept;

    /// Writes the table in the layout of the timemory wall_clock text file.
    void write_text(std::ostream& out);

    /// Writes the rows in the layout of the timemory wall_clock JSON file.
    void write_json(std::ostream& out, std::int32_t rank, std::int32_t mpi_size);

private:
    const wall_clock_thread_trees& trees();

    wall_clock_layout                              m_layout;
    std::vector<wall_clock_interval>               m_regions;
    std::vector<wall_clock_interval>               m_gpu_activity;
    std::shared_ptr<const wall_clock_thread_trees> m_trees;
};

}  // namespace rocprofsys::trace_cache
