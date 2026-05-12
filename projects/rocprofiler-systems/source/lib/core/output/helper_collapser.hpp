// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/process_tree_builder.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::output
{

// Helper if no GPU agents AND largest row size < threshold.
inline constexpr std::uintmax_t HELPER_SIZE_THRESHOLD_BYTES = 16ULL * 1024;

// Sibling helpers of count >= 2 collapse into one synthetic node
// with `collapsed` set. Singletons render normally.
[[nodiscard]] std::vector<process_node>
collapse_helpers(std::vector<process_node> roots);

}  // namespace rocprofsys::output
