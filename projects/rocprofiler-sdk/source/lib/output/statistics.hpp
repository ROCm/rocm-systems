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

#include "domain_type.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"

#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <type_traits>
#include <vector>

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
    float_type get_percent(float_type _total) const;
    float_type get_percent(const this_type&) const;

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

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned int) const
    {
        ar(cereal::make_nvp("count", m_cnt));
        ar(cereal::make_nvp("sum", m_sum));
        ar(cereal::make_nvp("sqr", m_sqr));
        ar(cereal::make_nvp("min", m_min));
        ar(cereal::make_nvp("max", m_max));
        ar(cereal::make_nvp("mean", get_mean()));
        ar(cereal::make_nvp("stddev", get_stddev()));
        ar(cereal::make_nvp("variance", get_variance()));
    }
};

template <typename Tp, typename Fp>
typename statistics<Tp, Fp>::float_type
statistics<Tp, Fp>::get_percent(float_type _total) const
{
    constexpr float_type one_hundred = 100.0;
    const float_type     _sum        = get_sum();

    ROCP_WARNING_IF(static_cast<int64_t>(_sum) > static_cast<int64_t>(_total))
        << "percentage calculation > 100%. sum=" << _sum << " > total=" << _total;

    return (_sum / _total) * one_hundred;
}

template <typename Tp, typename Fp>
typename statistics<Tp, Fp>::float_type
statistics<Tp, Fp>::get_percent(const statistics<Tp, Fp>& _rhs) const
{
    return get_percent(_rhs.get_sum());
}

using float_type        = double;
using stats_data_t      = statistics<uint64_t, float_type>;
using stats_map_t       = std::map<std::string_view, stats_data_t>;
using stats_pair_t      = std::pair<std::string_view, stats_data_t>;
using stats_entry_vec_t = std::vector<stats_pair_t>;

inline bool
default_stats_sorter(const stats_pair_t& lhs, const stats_pair_t& rhs)
{
    return (lhs.second.get_sum() > rhs.second.get_sum());
}

struct stats_entry_t
{
    using sort_predicate_t = bool (*)(const stats_pair_t&, const stats_pair_t&);

    stats_entry_t()                         = default;
    ~stats_entry_t()                        = default;
    stats_entry_t(const stats_entry_t&)     = default;
    stats_entry_t(stats_entry_t&&) noexcept = default;
    stats_entry_t& operator=(const stats_entry_t&) = default;
    stats_entry_t& operator=(stats_entry_t&&) noexcept = default;

    template <typename FuncT = sort_predicate_t>
    stats_entry_t& sort(FuncT&& _predicate = default_stats_sorter);

    explicit operator bool() const { return (total.get_count() > 0 && !entries.empty()); }

    stats_data_t      total   = {};
    stats_entry_vec_t entries = {};

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned int) const
    {
        total.serialize(ar, 0);
        auto _entries_map = std::map<std::string, stats_data_t>{};
        for(const auto& itr : entries)
            _entries_map.emplace(std::string{itr.first}, itr.second);
        ar(cereal::make_nvp("operations", _entries_map));
    }
};

template <typename FuncT>
stats_entry_t&
stats_entry_t::sort(FuncT&& _predicate)
{
    std::sort(entries.begin(), entries.end(), std::forward<FuncT>(_predicate));
    return *this;
}

using domain_stats_t     = std::pair<domain_type, stats_entry_t>;
using domain_stats_vec_t = std::vector<domain_stats_t>;

struct stats_formatter
{
    template <typename Tp>
    std::ostream& operator()(std::ostream& ofs, const Tp& _val) const;
};

struct percentage
{
    float_type           value = {};
    friend std::ostream& operator<<(std::ostream& os, percentage val)
    {
        return (stats_formatter{}(os, val) << val.value);
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

inline std::string
to_string(::rocprofiler::tool::percentage val)
{
    auto _ss = std::stringstream{};
    _ss << val;
    return _ss.str();
}
}  // namespace std
