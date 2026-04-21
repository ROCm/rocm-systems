// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "address_multirange.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace rocprofsys
{
namespace binary
{
address_multirange&
address_multirange::operator+=(std::pair<coarse, uintptr_t>&& _v)
{
    m_coarse_range = address_range{ std::min(m_coarse_range.low, _v.second),
                                    std::max(m_coarse_range.high, _v.second + 1) };
    return *this;
}

address_multirange&
address_multirange::operator+=(std::pair<coarse, address_range>&& _v)
{
    m_coarse_range = address_range{ std::min(m_coarse_range.low, _v.second.low),
                                    std::max(m_coarse_range.high, _v.second.high) };

    return *this;
}

address_multirange&
address_multirange::operator+=(uintptr_t _v)
{
    *this += std::make_pair(coarse{}, _v);

    // for(auto&& itr : m_fine_ranges)
    //    if(itr.contains(_v)) return *this;

    m_fine_ranges.emplace(address_range{ _v });
    return *this;
}

address_multirange&
address_multirange::operator+=(address_range _v)
{
    *this += std::make_pair(coarse{}, _v);

    // for(auto&& itr : m_fine_ranges)
    //    if(itr.contains(_v)) return *this;

    m_fine_ranges.emplace(_v);
    return *this;
}
}  // namespace binary
}  // namespace rocprofsys
