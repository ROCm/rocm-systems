// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
auto samples = std::map<uint32_t, std::map<uintptr_t, uint64_t>>{};
}

std::vector<sample_data>
get_samples(uint32_t _index)
{
    auto _data = std::vector<sample_data>{};
    _data.reserve(samples.at(_index).size());
    for(const auto& itr : samples.at(_index))
    {
        _data.emplace_back(sample_data{ itr.first, itr.second });
    }
    return _data;
}

std::map<uint32_t, std::vector<sample_data>>
get_samples()
{
    auto _data = std::map<uint32_t, std::vector<sample_data>>{};

    for(const auto& itr : samples)
    {
        _data[itr.first] = get_samples(itr.first);
    }

    return _data;
}

void
add_sample(uint32_t _index, uintptr_t _addr, uint64_t _count)
{
    samples[_index][_addr] += _count;
}

void
add_samples(uint32_t _index, const std::vector<uintptr_t>& _v)
{
    for(const auto& itr : _v)
        add_sample(_index, itr);
}

void
add_samples(uint32_t _index, const std::map<uintptr_t, uint64_t>& _v)
{
    for(const auto& itr : _v)
        add_sample(_index, itr.first, itr.second);
}
}  // namespace causal
}  // namespace rocprofsys
