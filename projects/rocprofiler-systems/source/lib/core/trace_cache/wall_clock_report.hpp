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
    bool flat             = false;
    bool timeline         = false;
    bool collapse_threads = false;
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
 * top of its thread if none does. Collapsing threads merges every thread into one tree
 * whose rows carry no thread number.
 */
class wall_clock_report
{
public:
    /**
     * Builds the call trees of the given intervals.
     * @param layout       Flat, timeline and thread-collapsing options of the tree.
     * @param regions      CPU regions; nest by time containment per thread.
     * @param gpu_activity Kernels, memory copies and scratch memory.
     * @throws std::runtime_error If an interval ends before it begins.
     */
    wall_clock_report(wall_clock_layout layout, std::vector<wall_clock_interval> regions,
                      const std::vector<wall_clock_interval>& gpu_activity);

    /// Writes the table in the layout of the timemory wall_clock text file.
    void write_text(std::ostream& out) const;

    /// Writes the rows in the layout of the timemory wall_clock JSON file.
    void write_json(std::ostream& out, std::int32_t rank, std::int32_t mpi_size) const;

private:
    wall_clock_layout                              m_layout;
    std::shared_ptr<const wall_clock_thread_trees> m_trees;
};

}  // namespace rocprofsys::trace_cache
