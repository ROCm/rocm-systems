// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

    address_range()                                    = default;
    address_range(const address_range&)                = default;
    address_range(address_range&&) noexcept            = default;
    address_range& operator=(const address_range&)     = default;
    address_range& operator=(address_range&&) noexcept = default;

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

    std::string as_hex() const;

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
