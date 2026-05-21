// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "output/process_tree_builder.hpp"

#include <cstddef>
#include <string>

namespace rocprofsys
{
class output_file_registry;
}

namespace rocprofsys::output
{

struct run_metadata;

struct summary_model
{
    build_result built{};
    std::size_t  process_count{ 0 };
    std::string  output_dir{};
};

// Snapshots the registry, runs build_tree -> collapse_helpers ->
// classify, derives process_count and output_dir. `meta.output_dir_abs`
// wins when set; otherwise the dir is derived from a registered row.
[[nodiscard]] summary_model
build_summary_model(const output_file_registry& registry, const run_metadata& meta);

}  // namespace rocprofsys::output
