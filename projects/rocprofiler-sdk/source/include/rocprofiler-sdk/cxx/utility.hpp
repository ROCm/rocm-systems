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
//

#pragma once

#include <cassert>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace rocprofiler
{
namespace sdk
{
namespace utility
{
template <typename Tp>
auto
as_hex(Tp val, size_t width = 0)
{
    uintptr_t _uintp_val = 0;
    if constexpr(std::is_pointer<Tp>::value)
        _uintp_val = reinterpret_cast<uintptr_t>(val);
    else
        _uintp_val = val;

    auto ss = std::stringstream{};
    ss << "0x" << std::hex << std::setfill('0') << std::setw(width) << _uintp_val;
    return ss.str();
}

template <typename Integral = uint64_t>
constexpr Integral
bit_mask(int first, int last)
{
    assert(last >= first && "Error: hsa_support::bit_mask -> invalid argument");
    size_t num_bits = last - first + 1;
    return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
                                               /* num_bits exceed the size of Integral */
                                               : ((Integral{1} << num_bits) - 1))
           << first;
}

/* Extract bits [last:first] from t.  */
template <typename Integral>
constexpr Integral
bit_extract(Integral x, int first, int last)
{
    return (x >> first) & bit_mask<Integral>(0, last - first);
}
}  // namespace utility
}  // namespace sdk
}  // namespace rocprofiler
