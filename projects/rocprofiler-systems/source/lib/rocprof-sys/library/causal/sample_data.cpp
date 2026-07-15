// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/causal/sample_data.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <set>

namespace rocprofsys
{
namespace causal
{
namespace
{
auto samples = std::map<std::uint32_t, std::map<uintptr_t, std::uint64_t>>{};
}

std::vector<sample_data>
get_samples(std::uint32_t _index)
{
    auto _data = std::vector<sample_data>{};
    _data.reserve(samples.at(_index).size());
    for(const auto& itr : samples.at(_index))
    {
        _data.emplace_back(sample_data{ itr.first, itr.second });
    }
    return _data;
}

std::map<std::uint32_t, std::vector<sample_data>>
get_samples()
{
    auto _data = std::map<std::uint32_t, std::vector<sample_data>>{};

    for(const auto& itr : samples)
    {
        _data[itr.first] = get_samples(itr.first);
    }

    return _data;
}

void
add_sample(std::uint32_t _index, uintptr_t _addr, std::uint64_t _count)
{
    samples[_index][_addr] += _count;
}

void
add_samples(std::uint32_t _index, const std::vector<uintptr_t>& _v)
{
    for(const auto& itr : _v)
        add_sample(_index, itr);
}

void
add_samples(std::uint32_t _index, const std::map<uintptr_t, std::uint64_t>& _v)
{
    for(const auto& itr : _v)
        add_sample(_index, itr.first, itr.second);
}
}  // namespace causal
}  // namespace rocprofsys
