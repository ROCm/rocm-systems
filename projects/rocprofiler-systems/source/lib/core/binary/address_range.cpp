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

#include "binary/address_range.hpp"
#include "debug.hpp"

namespace rocprofsys
{
namespace binary
{
address_range::address_range(uintptr_t _v)
: low{ _v }
, high{ _v }
{}

address_range::address_range(uintptr_t _low, uintptr_t _high)
: low{ _low }
, high{ _high }
{
    TIMEMORY_REQUIRE(high >= low)
        << "Error! address_range high must be >= low. low=" << as_hex(low)
        << ", high=" << as_hex(high) << "\n";
}

bool
address_range::is_range() const
{
    return (low < high);
}

std::string
address_range::as_string(int _depth) const
{
    std::stringstream _ss{};
    _ss << std::hex;
    _ss << std::setw(2 * _depth) << "";
    _ss.fill('0');
    _ss << "0x" << std::setw(16) << low << "-" << "0x" << std::setw(16) << high;
    return _ss.str();
}

uintptr_t
address_range::size() const
{
    return (low == high) ? 1 : (high > low) ? (high - low + 1) : (low - high + 1);
}

bool
address_range::is_valid() const
{
    return (low <= high && (low + 1) > 1);
}

bool
address_range::contains(uintptr_t _v) const
{
    return (is_range()) ? (low <= _v && high > _v) : (_v == low);
}

bool
address_range::contains(address_range _v) const
{
    return (*this == _v) || (contains(_v.low) && (contains(_v.high) || _v.high == high));
}

bool
address_range::overlaps(address_range _v) const
{
    if(contains(_v)) return false;
    int64_t _lhs_diff = (high - low);
    int64_t _rhs_diff = (_v.high - _v.low);
    int64_t _diff     = (std::max(high, _v.high) - std::min(low, _v.low));
    return (_diff < (_lhs_diff + _rhs_diff));
}

bool
address_range::contiguous_with(address_range _v) const
{
    return (_v.low == high || low == _v.high);
}

bool
address_range::operator==(address_range _v) const
{
    // if arg is range and this is not range, call this function with arg
    // if(_v.is_range() && !is_range()) return false;
    // check if arg is in range
    // if(is_range() && !_v.is_range()) return false;
    // both are ranges or both are just address
    return std::tie(low, high) == std::tie(_v.low, _v.high);
}

bool
address_range::operator<(address_range _v) const
{
    if(is_range() && !_v.is_range())
    {
        return (low == _v.low) ? true : (low < _v.low);
    }
    else if(!is_range() && _v.is_range())
    {
        return (low == _v.low) ? false : (low < _v.low);
    }
    else if(!is_range() && !_v.is_range())
    {
        return (low < _v.low);
    }
    return std::tie(low, high) < std::tie(_v.low, _v.high);
    // if(_v.low == _v.high && _v.low >= low && _v.low < high) return false;
    // return (low == _v.low) ? (high > _v.high) : (low < _v.low);
}

bool
address_range::operator>(address_range _v) const
{
    return !(*this < _v) && !(*this == _v);
}

address_range&
address_range::operator+=(uintptr_t _v)
{
    if(is_valid())
    {
        low += _v;
        high += _v;
    }
    else
    {
        low  = _v;
        high = _v;
    }
    return *this;
}

address_range&
address_range::operator-=(uintptr_t _v)
{
    if(is_valid())
    {
        low -= _v;
        high -= _v;
    }
    else
    {
        low  = _v;
        high = _v;
    }
    return *this;
}

address_range&
address_range::operator+=(address_range _v)
{
    if(!contiguous_with(_v))
        throw exception<std::runtime_error>(
            "attempting to add two address ranges that are not contiguous");

    low  = std::min(low, _v.low);
    high = std::max(high, _v.high);
    return *this;
}

hash_value_t
address_range::hash() const
{
    return (is_range()) ? tim::get_hash_id(hash_value_t{ low }, high)
                        : hash_value_t{ low };
}
}  // namespace binary
}  // namespace rocprofsys
