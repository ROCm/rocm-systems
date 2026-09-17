// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/artifact.hpp"
#include "core/output/process_metadata.hpp"

#include <sys/types.h>

#include <cstdint>
#include <span>
#include <vector>

namespace rocprofsys::output
{

struct process_node
{
    process_metadata          meta;
    std::vector<artifact>     rows;
    std::vector<process_node> children;
    std::uint64_t             own_size_bytes{ 0 };
    std::uint64_t             cumulative_size_bytes{ 0 };
};

/*
 * Keep the track of the missing or excluded pids while
 * building the process tree for rendering
 *
 * missing_metadata_pids - PIDs seen in registered rows/process records
 * but missing their own metadata record
 *
 * cyclic_ppid_pids - PIDs excluded from every root's tree because their
 * ppid chain forms a cycle (corrupted metadata)
 */
struct process_tree_diagnostics
{
    std::vector<pid_t> missing_metadata_pids;
    std::vector<pid_t> cyclic_ppid_pids;
};

class process_tree
{
public:
    // Builds a tree from `rows`/`processes` grouped by pid/ppid
    process_tree(std::span<const artifact>         rows,
                 std::span<const process_metadata> processes);

    [[nodiscard]] const std::vector<process_node>& roots() const noexcept
    {
        return m_roots;
    }

    [[nodiscard]] const process_tree_diagnostics& diagnostics() const noexcept
    {
        return m_diagnostics;
    }

private:
    std::vector<process_node> m_roots;
    process_tree_diagnostics  m_diagnostics;
};

}  // namespace rocprofsys::output
