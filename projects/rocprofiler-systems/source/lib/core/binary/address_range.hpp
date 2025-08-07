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

#include "core/binary/fwd.hpp"
#include "core/common.hpp"
#include "core/timemory.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/utility/macros.hpp>

#include <cstdint>
#include <limits>

namespace rocprofsys
{
namespace binary
{
struct address_range
{
    // set to low to max and high to min to support std::min(...)
    // and std::max(...) assignment
    uintptr_t low  = std::numeric_limits<uintptr_t>::max();
    uintptr_t high = std::numeric_limits<uintptr_t>::min();

    ROCPROFSYS_DEFAULT_OBJECT(address_range)

    explicit address_range(uintptr_t _v);
    address_range(uintptr_t _low, uintptr_t _high);

    bool contains(uintptr_t) const;
    bool contains(address_range) const;
    bool overlaps(address_range) const;
    bool contiguous_with(address_range) const;

    bool operator==(address_range _v) const;
    bool operator!=(address_range _v) const { return !(*this == _v); }
    bool operator<(address_range _v) const;
    bool operator>(address_range _v) const;

    address_range& operator+=(uintptr_t);
    address_range& operator-=(uintptr_t);
    address_range& operator+=(address_range);

    bool         is_range() const;
    hash_value_t hash() const;
    std::string  as_string(int _depth = 0) const;
    bool         is_valid() const;
    uintptr_t    size() const;
    explicit     operator bool() const { return is_valid(); }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned)
    {
        ar(cereal::make_nvp("low", low));
        ar(cereal::make_nvp("high", high));
    }
};
}  // namespace binary

inline binary::address_range
operator+(binary::address_range _lhs, uintptr_t _v)
{
    return (_lhs += _v);
}

inline binary::address_range
operator+(uintptr_t _v, binary::address_range _lhs)
{
    return (_lhs += _v);
}
}  // namespace rocprofsys

namespace std
{
template <>
struct hash<::rocprofsys::binary::address_range>
{
    using address_range_t = ::rocprofsys::binary::address_range;

    auto operator()(const address_range_t& _v) const { return _v.hash(); }
    auto operator()(address_range_t&& _v) const { return _v.hash(); }
};
}  // namespace std
