// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/artifact.hpp"
#include "core/output/process_tree.hpp"

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>

namespace rocprofsys::output
{

struct run_metadata
{
    std::string              run_label;
    std::chrono::nanoseconds duration{ 0 };
    std::string              output_dir_abs;

    [[nodiscard]] static run_metadata capture(
        std::chrono::steady_clock::time_point load_baseline);
};

[[nodiscard]] std::size_t
display_width(std::string_view text);

[[nodiscard]] std::string
summarize_command(std::string_view command);

[[nodiscard]] std::size_t
box_width(std::span<const std::string> header_lines,
          std::span<const std::string> tree_lines);

// Writes nothing at all if `rows` is empty — there is no output to summarize.
void
write_summary(std::ostream& os, const process_tree& tree, const run_metadata& meta,
              std::span<const artifact> rows);

}  // namespace rocprofsys::output
