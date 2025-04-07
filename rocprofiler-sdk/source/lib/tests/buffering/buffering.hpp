// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/container/record_header_buffer.hpp"

#include <random>
#include <string>
#include <type_traits>

namespace rocprofiler
{
namespace test
{
template <typename Tp, size_t N>
struct raw_array
{
    raw_array()  = default;
    ~raw_array() = default;

    raw_array(const raw_array&)     = default;
    raw_array(raw_array&&) noexcept = default;

    raw_array& operator=(const raw_array&) = default;
    raw_array& operator=(raw_array&&) noexcept = default;

    bool operator==(const raw_array<Tp, N>& rhs) const;
    bool operator!=(const raw_array<Tp, N>& rhs) const { return !(*this == rhs); }

    Tp& operator[](size_t n) { return data[n]; }
    Tp  operator[](size_t n) const { return data[n]; }

    std::string to_string() const;

    Tp data[N];
};

template <typename Tp, size_t N>
bool
raw_array<Tp, N>::operator==(const raw_array<Tp, N>& rhs) const
{
    for(size_t i = 0; i < N; ++i)
    {
        if constexpr(std::is_integral_v<Tp>)
        {
            if((*this)[i] != rhs[i]) return false;
        }
        else
        {
            auto _diff = (*this)[i] - rhs[i];
            if(_diff < Tp{0.0}) _diff *= Tp{-1.0};
            if(_diff > std::numeric_limits<Tp>::round_error()) return false;
        }
    }
    return true;
}

template <typename Tp, size_t N>
std::string
raw_array<Tp, N>::to_string() const
{
    auto _ss = std::stringstream{};
    for(size_t i = 0; i < N; ++i)
    {
        _ss << " " << std::setw(8) << std::fixed << std::setprecision(3) << (*this)[i];
        if(i % 16 == 15) _ss << "\n";
    }
    return _ss.str();
}

template <typename Tp, size_t N>
auto
generate(raw_array<Tp, N>& _v, Tp _min, Tp _max)
{
    using rng_t = std::conditional_t<std::is_integral<Tp>::value,
                                     std::uniform_int_distribution<Tp>,
                                     std::uniform_real_distribution<Tp>>;
    auto _rd    = std::random_device{};
    auto _gen   = std::mt19937_64{_rd()};
    auto _rng   = rng_t{_min, _max};
    for(size_t i = 0; i < N; ++i)
        _v[i] = _rng(_gen);
}
}  // namespace test
}  // namespace rocprofiler
