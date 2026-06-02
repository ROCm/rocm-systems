// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common.hpp"
#include "core/demangler.hpp"

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

    auto _vec = str_vec_t{ _cleanup(rocprofsys::utility::demangle<Tp>(), "tim::")... };
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
