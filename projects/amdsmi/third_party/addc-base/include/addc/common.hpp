// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <vector>

namespace addc
{

inline constexpr int kUnclassifiedAfid = 16999;
// Compatibility alias for existing source consumers.
inline constexpr int kAfidSentinel = kUnclassifiedAfid;

inline std::vector<int> afid_list(int v)
{
    return {v};
}

} // namespace addc
