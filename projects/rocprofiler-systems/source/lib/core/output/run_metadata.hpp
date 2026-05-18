// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstddef>
#include <string>

namespace rocprofsys::output
{

struct run_metadata
{
    std::string              run_label;
    std::chrono::nanoseconds duration{ 0 };
    std::size_t              process_count{ 0 };
    std::string              output_dir_abs;

    // Builds an iso-8601 UTC label and a wall-clock duration from
    // the given steady-clock baseline (typically library-load time).
    // process_count and output_dir_abs are left for the caller to
    // populate; the renderer fills sensible defaults when they are
    // not set.
    [[nodiscard]] static run_metadata capture(
        std::chrono::steady_clock::time_point load_baseline);
};

}  // namespace rocprofsys::output
