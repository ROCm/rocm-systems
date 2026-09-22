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

#include "statistics.hpp"

namespace rocprofiler
{
namespace tool
{
template <typename Tp>
std::ostream&
stats_formatter::operator()(std::ostream& ofs, const Tp& _val) const
{
    using value_type = common::mpl::unqualified_type_t<Tp>;

    if constexpr(std::is_floating_point<value_type>::value)
    {
        constexpr value_type one_hundredth = 1.0e-2;
        if(_val > one_hundredth)
            ofs << std::setprecision(6) << std::fixed;
        else
            ofs << std::setprecision(8) << std::scientific;
    }
    else if constexpr(std::is_same<Tp, percentage>::value)
    {
        constexpr float_type one           = 1.0;
        constexpr float_type one_hundredth = 1.0e-2;
        if(_val.value >= one)
            ofs << std::setprecision(2) << std::fixed;
        else if(_val.value > one_hundredth)
            ofs << std::setprecision(4) << std::fixed;
        else
            ofs << std::setprecision(3) << std::scientific;
    }

    return ofs;
}

#define STATS_FORMATTER_INSTANTIATE_TEMPLATE(TYPE)                                                 \
    template std::ostream& stats_formatter::operator()(std::ostream&, const TYPE&) const;

STATS_FORMATTER_INSTANTIATE_TEMPLATE(std::string)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(std::string_view)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(percentage)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(float)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(double)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(uint8_t)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(uint16_t)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(uint32_t)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(uint64_t)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(int8_t)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(int16_t)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(int32_t)
STATS_FORMATTER_INSTANTIATE_TEMPLATE(int64_t)
}  // namespace tool
}  // namespace rocprofiler
