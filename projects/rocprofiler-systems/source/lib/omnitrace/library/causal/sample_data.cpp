// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

namespace omnitrace
{
namespace causal
{
namespace
{
auto samples = std::map<uint32_t, std::set<sample_data>>{};
}

std::set<sample_data>
get_samples(uint32_t _index)
{
    return samples[_index];
}

std::map<uint32_t, std::set<sample_data>>
get_samples()
{
    return samples;
}

void
add_sample(uint32_t _index, uintptr_t _v)
{
    auto& _samples = samples[_index];
    auto  _value   = sample_data{ _v };
    _value.count   = 1;
    auto itr       = _samples.find(_value);
    if(itr == _samples.end())
        _samples.emplace(_value);
    else
        itr->count += 1;
}

void
add_samples(uint32_t _index, const std::vector<uintptr_t>& _v)
{
    for(const auto& itr : _v)
        add_sample(_index, itr);
}
}  // namespace causal
}  // namespace omnitrace
