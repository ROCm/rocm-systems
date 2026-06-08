// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <string>

namespace rocprofsys::output
{

struct run_metadata
{
    std::string              run_label;
    std::chrono::nanoseconds duration{ 0 };
    std::string              output_dir_abs;

    // ISO-8601 UTC label + wall-clock duration from a steady-clock
    // baseline (typically library-load time). output_dir_abs is left
    // empty, in which case the renderer derives it from a registered row.
    [[nodiscard]] static run_metadata capture(
        std::chrono::steady_clock::time_point load_baseline);
};

}  // namespace rocprofsys::output
