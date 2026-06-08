// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "output/process_tree_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocprofsys
{
class output_file_registry;
}

namespace rocprofsys::output
{

struct run_metadata;

struct summary_model
{
    build_result   built{};
    std::size_t    process_count{ 0 };
    std::string    output_dir{};
    std::uintmax_t total_output_bytes{ 0 };
    // Sort-unique union of GPU ids used across the whole run, and the
    // node's total GPU count when known (drives the "(all)" qualifier).
    std::vector<int>           utilized_gpu_ids{};
    std::optional<std::size_t> node_gpu_count{};
};

// Snapshots the registry, runs build_tree -> collapse_helpers ->
// classify, derives process_count and output_dir. `meta.output_dir_abs`
// wins when set; otherwise the dir is derived from a registered row.
[[nodiscard]] summary_model
build_summary_model(const output_file_registry& registry, const run_metadata& meta);

}  // namespace rocprofsys::output
