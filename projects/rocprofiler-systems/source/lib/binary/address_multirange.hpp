// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/binary/address_range.hpp"
#include "core/binary/fwd.hpp"

#include <timemory/utility/macros.hpp>

#include <cstdint>
#include <utility>

namespace rocprofsys
{
namespace binary
{
struct address_multirange
{
    struct coarse
    {};

    address_multirange& operator+=(std::pair<coarse, uintptr_t>&&);
    address_multirange& operator+=(std::pair<coarse, address_range>&& _v);
    address_multirange& operator+=(uintptr_t _v);
    address_multirange& operator+=(address_range _v);

    template <typename Tp>
        requires(std::is_integral_v<concepts::unqualified_type_t<Tp>> ||
                 std::is_same_v<concepts::unqualified_type_t<Tp>, address_range>)
    bool contains(Tp&& _v) const;

    auto size() const { return m_fine_ranges.size(); }
    auto empty() const { return m_fine_ranges.empty(); }
    auto range_size() const { return m_coarse_range.size(); }
    auto get_coarse_range() const { return m_coarse_range; }
    auto get_ranges() const { return m_fine_ranges; }

private:
    address_range           m_coarse_range = {};
    std::set<address_range> m_fine_ranges  = {};
};

template <typename Tp>
    requires(std::is_integral_v<concepts::unqualified_type_t<Tp>> ||
             std::is_same_v<concepts::unqualified_type_t<Tp>, address_range>)
ROCPROFSYS_INLINE bool
address_multirange::contains(Tp&& _v) const
{
    if(!m_coarse_range.contains(_v)) return false;
    return std::any_of(m_fine_ranges.begin(), m_fine_ranges.end(),
                       [_v](auto&& itr) { return itr.contains(_v); });
}
}  // namespace binary
}  // namespace rocprofsys
