// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output_file_registry.hpp"

#include <sys/types.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocprofsys::output
{

enum class role_hint
{
    main,
    gpu,
    engine,
};

struct process_metadata
{
    pid_t            pid{ -1 };
    pid_t            ppid{ -1 };
    std::string      command;
    std::vector<int> gpu_ids;
};

struct helper_range
{
    pid_t       min_pid{ -1 };
    pid_t       max_pid{ -1 };
    std::size_t count{ 0 };
};

struct process_node
{
    process_metadata            meta;
    std::vector<output_file>    rows;
    std::optional<role_hint>    role;
    std::optional<helper_range> collapsed;
    std::vector<process_node>   children;
};

struct build_diagnostics
{
    std::vector<pid_t> missing_metadata_pids;
};

struct build_result
{
    // Orphan PIDs (PPID absent from the input set) attach as
    // additional roots alongside the main process.
    std::vector<process_node> roots;
    build_diagnostics         diagnostics;
};

// Pure: no I/O, no globals. Per-node rows are sorted descending by
// size_bytes.
[[nodiscard]] build_result
build_tree(const std::vector<output_file>&      rows,
           const std::vector<process_metadata>& processes);

}  // namespace rocprofsys::output
