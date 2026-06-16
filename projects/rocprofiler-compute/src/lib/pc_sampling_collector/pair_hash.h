// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstddef>
#include <functional>
#include <utility>

namespace rocprofiler_compute_tool
{
// Generic hash for std::pair, combining the element hashes with the boost-style
// hash_combine mix. Single-sourced so the magic constant lives in one place.
struct pair_hash_t
{
    template<typename A, typename B>
    size_t operator()(const std::pair<A, B>& p) const
    {
        const size_t h1 = std::hash<A>{}(p.first);
        const size_t h2 = std::hash<B>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};
}  // namespace rocprofiler_compute_tool
