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

#include <sstream>
#include <string>

#if !defined(OMNITRACE_FOLD_EXPRESSION)
#    define OMNITRACE_FOLD_EXPRESSION(...) ((__VA_ARGS__), ...)
#endif

namespace omnitrace
{
inline namespace common
{
namespace
{
template <typename DelimT, typename... Args>
auto
join(DelimT&& _delim, Args&&... _args)
{
    std::stringstream _ss{};
    OMNITRACE_FOLD_EXPRESSION(_ss << _delim << _args);
    if constexpr(std::is_same<DelimT, char>::value)
    {
        return _ss.str().substr(1);
    }
    else
    {
        return _ss.str().substr(std::string{ _delim }.length());
    }
}
}  // namespace
}  // namespace common
}  // namespace omnitrace
