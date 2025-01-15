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

//  IMPORTANT NOTE:
//      this may be a header file but will lead to ODR violations if included
//      more than once. The reason it is in a header file is so that the python
//      bindings can also build these symbols. The absence of inline is intentional!

#include "library/coverage.hpp"

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace rocprofsys
{
namespace coverage
{
namespace
{
template <typename Tp, typename... Args>
inline std::set<Tp, Args...>
get_uncovered(const std::set<Tp, Args...>& _covered,
              const std::set<Tp, Args...>& _possible)
{
    std::set<Tp, Args...> _v{};
    for(auto&& itr : _possible)
    {
        if(_covered.count(itr) == 0) _v.emplace(itr);
    }
    return _v;
}
//
template <typename Tp, typename... Args>
inline std::vector<Tp, Args...>
get_uncovered(const std::vector<Tp, Args...>& _covered,
              const std::vector<Tp, Args...>& _possible)
{
    std::vector<Tp, Args...> _v{};
    for(auto&& itr : _possible)
    {
        if(!std::any_of(_covered.begin(), _covered.end(),
                        [itr](auto&& _entry) { return _entry == itr; }))
            _v.emplace_back(itr);
    }
    return _v;
}
}  // namespace

code_coverage::data&
code_coverage::data::operator+=(const data& rhs)
{
    for(auto&& itr : rhs.addresses)
        addresses.emplace(itr);
    for(auto&& itr : rhs.modules)
        modules.emplace(itr);
    for(auto&& itr : rhs.modules)
        modules.emplace(itr);
    return *this;
}

code_coverage::data
code_coverage::data::operator+(const data& rhs) const
{
    return data{ *this } += rhs;
}

double
code_coverage::operator()(Category _c) const
{
    switch(_c)
    {
        case STANDARD: return static_cast<double>(count) / static_cast<double>(size);
        case ADDRESS:
            return static_cast<double>(covered.addresses.size()) /
                   static_cast<double>(possible.addresses.size());
        case MODULE:
            return static_cast<double>(covered.modules.size()) /
                   static_cast<double>(possible.modules.size());
        case FUNCTION:
            return static_cast<double>(covered.functions.size()) /
                   static_cast<double>(possible.functions.size());
    }
    return 0.0;
}

code_coverage::int_set_t
code_coverage::get_uncovered_addresses() const
{
    return get_uncovered(covered.addresses, possible.addresses);
}

code_coverage::str_set_t
code_coverage::get_uncovered_modules() const
{
    return get_uncovered(covered.modules, possible.modules);
}

code_coverage::str_set_t
code_coverage::get_uncovered_functions() const
{
    return get_uncovered(covered.functions, possible.functions);
}

//--------------------------------------------------------------------------------------//

coverage_data&
coverage_data::operator+=(const coverage_data& rhs)
{
    count += rhs.count;
    return *this;
}

coverage_data
coverage_data::operator+(const coverage_data& rhs) const
{
    return coverage_data{ *this } += rhs;
}

bool
coverage_data::operator==(const coverage_data& rhs) const
{
    return std::tie(module, function, address) ==
           std::tie(rhs.module, rhs.function, rhs.address);
}

bool
coverage_data::operator==(const data_tuple_t& rhs) const
{
    return std::tie(module, function, address) ==
           std::tie(std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs));
}

bool
coverage_data::operator!=(const coverage_data& rhs) const
{
    return !(*this == rhs);
}

bool
coverage_data::operator<(const coverage_data& rhs) const
{
    if(count != rhs.count) return count < rhs.count;
    if(module != rhs.module) return module < rhs.module;
    if(function != rhs.function) return function < rhs.function;
    if(address != rhs.address) return address < rhs.address;
    if(line != rhs.line) return line < rhs.line;
    return source < rhs.source;
}

bool
coverage_data::operator<=(const coverage_data& rhs) const
{
    return (*this == rhs || *this < rhs);
}

bool
coverage_data::operator>(const coverage_data& rhs) const
{
    return (*this != rhs && !(*this < rhs));
}

bool
coverage_data::operator>=(const coverage_data& rhs) const
{
    return !(*this < rhs);
}
//
}  // namespace coverage
}  // namespace rocprofsys
