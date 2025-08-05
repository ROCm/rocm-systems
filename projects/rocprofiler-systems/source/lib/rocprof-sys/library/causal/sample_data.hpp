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

#pragma once

#include "core/defines.hpp"
#include "core/timemory.hpp"

#include <chrono>
#include <cstdint>

namespace rocprofsys
{
namespace causal
{
struct sample_data
{
    uintptr_t        address = 0x0;
    mutable uint64_t count   = 0;

    bool operator==(sample_data _v) const { return (address == _v.address); }
    bool operator!=(sample_data _v) const { return !(*this == _v); }
    bool operator<(sample_data _v) const { return (address < _v.address); }
    bool operator>(sample_data _v) const { return (address > _v.address); }
    bool operator<=(sample_data _v) const { return (address <= _v.address); }
    bool operator>=(sample_data _v) const { return (address >= _v.address); }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned)
    {
        ar(cereal::make_nvp("address", address), cereal::make_nvp("count", count));
    }
};

std::map<uint32_t, std::vector<sample_data>>
get_samples();

void
add_samples(uint32_t, const std::vector<uintptr_t>&);

std::vector<sample_data> get_samples(uint32_t);

void add_sample(uint32_t, uintptr_t, uint64_t = 1);

void
add_samples(uint32_t, const std::map<uintptr_t, uint64_t>&);
}  // namespace causal
}  // namespace rocprofsys
