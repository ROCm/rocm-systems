// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>

namespace rocprofiler
{
namespace tool
{
/// \struct statistics
/// \tparam Tp data type for statistical accumulation
/// \tparam Fp floating point data type to use for division
/// \brief A generic class for statistical accumulation.
///
template <typename Tp, typename Fp = double>
struct statistics
{
public:
    using value_type = Tp;
    using float_type = Fp;
    using this_type  = statistics<Tp, Fp>;
    static_assert(std::is_arithmetic<Tp>::value, "only supports arithmetic types");

public:
    statistics()                      = default;
    ~statistics()                     = default;
    statistics(const statistics&)     = default;
    statistics(statistics&&) noexcept = default;
    statistics& operator=(const statistics&) = default;
    statistics& operator=(statistics&&) noexcept = default;

    explicit statistics(value_type val)
    : m_cnt(1)
    , m_sum(val)
    , m_sqr(val * val)
    , m_min(val)
    , m_max(val)
    {}

    statistics& operator=(value_type val)
    {
        m_cnt = 1;
        m_sum = val;
        m_min = val;
        m_max = val;
        m_sqr = (val * val);
        return *this;
    }

public:
    // Accumulated values
    int64_t    get_count() const { return m_cnt; }
    value_type get_min() const { return m_min; }
    value_type get_max() const { return m_max; }
    value_type get_sum() const { return m_sum; }
    value_type get_sqr() const { return m_sqr; }
    float_type get_mean() const { return static_cast<float_type>(m_sum) / m_cnt; }
    float_type get_variance() const
    {
        if(m_cnt < 2) return (m_sum - m_sum);

        auto _sum_of_squared_samples = m_sqr;
        auto _sum_squared_mean       = (m_sum * m_sum) / static_cast<float_type>(m_cnt);
        return (_sum_of_squared_samples - _sum_squared_mean) / static_cast<float_type>(m_cnt - 1);
    }

    float_type get_stddev() const { return ::std::sqrt(::std::abs(get_variance())); }

    // Modifications
    void reset()
    {
        m_cnt = 0;
        m_sum = value_type{};
        m_sqr = value_type{};
        m_min = value_type{};
        m_max = value_type{};
    }

public:
    // Operators (value_type)
    statistics& operator+=(value_type val)
    {
        if(m_cnt == 0)
        {
            m_sum = val;
            m_sqr = (val * val);
            m_min = val;
            m_max = val;
        }
        else
        {
            m_sum += val;
            m_sqr += (val * val);
            m_min = ::std::min(m_min, val);
            m_max = ::std::max(m_max, val);
        }
        ++m_cnt;

        return *this;
    }

    statistics& operator-=(value_type val)
    {
        if(m_cnt > 1) --m_cnt;
        m_sum -= val;
        m_sqr -= (val * val);
        m_min -= val;
        m_max -= val;
        return *this;
    }

    statistics& operator*=(value_type val)
    {
        m_sum *= val;
        m_sqr *= (val * val);
        m_min *= val;
        m_max *= val;
        return *this;
    }

    statistics& operator/=(value_type val)
    {
        m_sum /= val;
        m_sqr /= (val * val);
        m_min /= val;
        m_max /= val;
        return *this;
    }

public:
    // Operators (this_type)
    statistics& operator+=(const statistics& rhs)
    {
        if(m_cnt == 0)
        {
            m_sum = rhs.m_sum;
            m_sqr = rhs.m_sqr;
            m_min = rhs.m_min;
            m_max = rhs.m_max;
        }
        else
        {
            m_sum += rhs.m_sum;
            m_sqr += rhs.m_sqr;
            m_min = ::std::min(m_min, rhs.m_min);
            m_max = ::std::max(m_max, rhs.m_max);
        }
        m_cnt += rhs.m_cnt;
        return *this;
    }

    // Operators (this_type)
    statistics& operator-=(const statistics& rhs)
    {
        if(m_cnt > 0)
        {
            m_sum -= rhs.m_sum;
            m_sqr -= rhs.m_sqr;
            m_min = ::std::min(m_min, rhs.m_min);
            m_max = ::std::max(m_max, rhs.m_max);
        }
        return *this;
    }

private:
    // summation of each history^1
    int64_t    m_cnt = 0;
    value_type m_sum = value_type{};
    value_type m_sqr = value_type{};
    value_type m_min = value_type{};
    value_type m_max = value_type{};

public:
    // friend operator for addition
    friend statistics operator+(const statistics& lhs, const statistics& rhs)
    {
        return statistics(lhs) += rhs;
    }

    friend statistics operator-(const statistics& lhs, const statistics& rhs)
    {
        return statistics(lhs) -= rhs;
    }
};
}  // namespace tool
}  // namespace rocprofiler

namespace std
{
template <typename Tp>
::rocprofiler::tool::statistics<Tp>
max(::rocprofiler::tool::statistics<Tp> lhs, const Tp& rhs)
{
    return lhs.get_max(rhs);
}

template <typename Tp>
::rocprofiler::tool::statistics<Tp>
min(::rocprofiler::tool::statistics<Tp> lhs, const Tp& rhs)
{
    return lhs.get_min(rhs);
}
}  // namespace std
