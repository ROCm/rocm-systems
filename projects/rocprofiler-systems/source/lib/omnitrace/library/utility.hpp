// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace omnitrace
{
namespace utility
{
/// provides an alternative thread index for when using threading::get_id() is not
/// desirable
inline auto
get_thread_index()
{
    static std::atomic<int64_t> _c{ 0 };
    static thread_local int64_t _v = _c++;
    return _v;
}

/// fills any array with the result of the functor
template <size_t N, typename FuncT>
inline auto
get_filled_array(FuncT&& _func)
{
    using Tp = std::decay_t<decltype(_func())>;
    std::array<Tp, N> _v{};
    for(auto& itr : _v)
        itr = std::move(_func());
    return _v;
}

/// returns a vector with a preallocated buffer
template <typename... Tp>
inline auto
get_reserved_vector(size_t _n)
{
    std::vector<Tp...> _v{};
    _v.reserve(_n);
    return _v;
}
}  // namespace utility
}  // namespace omnitrace
