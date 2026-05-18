// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output/process_tree_builder.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::output
{

// A childless, GPU-agent-free process whose largest registered row
// is smaller than this is treated as a short-lived helper fork
// (e.g. a setup script, a launcher subprocess) and collapsed with
// its siblings into a single range-summary line. Real GPU worker
// processes routinely emit traces orders of magnitude larger; the
// 16 KiB ceiling separates "noise" from "signal" comfortably.
inline constexpr std::uintmax_t HELPER_MAX_SIZE_BYTES = 16ULL * 1024;

// Sibling helpers of count >= 2 collapse into one synthetic node
// with `collapsed` set. Singletons render normally.
[[nodiscard]] std::vector<process_node>
collapse_helpers(std::vector<process_node> roots);

}  // namespace rocprofsys::output
