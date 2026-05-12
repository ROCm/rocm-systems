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
};

}  // namespace rocprofsys::output
