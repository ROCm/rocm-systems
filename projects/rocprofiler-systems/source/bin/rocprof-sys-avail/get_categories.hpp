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

#include "common.hpp"

#include <timemory/utility/demangle.hpp>
#include <timemory/utility/type_list.hpp>

#include <algorithm>
#include <sstream>
#include <string>

template <typename... Tp>
auto
get_categories(type_list<Tp...>)
{
    auto _cleanup = [](std::string _type, const std::string& _pattern) {
        auto _pos = std::string::npos;
        while((_pos = _type.find(_pattern)) != std::string::npos)
            _type.erase(_pos, _pattern.length());
        return _type;
    };
    (void) _cleanup;  // unused but set if sizeof...(Tp) == 0

    auto _vec = str_vec_t{ _cleanup(demangle<Tp>(), "tim::")... };
    std::sort(_vec.begin(), _vec.end(), [](const auto& lhs, const auto& rhs) {
        // prioritize project category
        auto lpos = lhs.find("project::");
        auto rpos = rhs.find("project::");
        return (lpos == rpos) ? (lhs < rhs) : (lpos < rpos);
    });
    std::stringstream _ss{};
    for(auto&& itr : _vec)
    {
        _ss << ", " << itr;
    }
    std::string _v = _ss.str();
    if(!_v.empty()) return _v.substr(2);
    return _v;
}
