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
#include "defines.hpp"

#include <timemory/components/types.hpp>
#include <timemory/enum.h>
#include <timemory/utility/types.hpp>

#include <set>
#include <string>

template <typename Type = void>
struct component_categories;

template <typename Type>
struct component_categories
{
    template <typename... Tp>
    void operator()(std::set<std::string>& _v, type_list<Tp...>) const
    {
        //
        auto _cleanup = [](std::string _type, const std::string& _pattern) {
            auto _pos = std::string::npos;
            while((_pos = _type.find(_pattern)) != std::string::npos)
                _type = _type.erase(_pos, _pattern.length());
            return _type;
        };
        (void) _cleanup;  // unused but set if sizeof...(Tp) == 0

        TIMEMORY_FOLD_EXPRESSION(_v.emplace(TIMEMORY_JOIN(
            "::", "component", _cleanup(tim::try_demangle<Tp>(), "tim::"))));
    }

    void operator()(std::set<std::string>& _v) const
    {
        if constexpr(!tim::concepts::is_placeholder<Type>::value)
            (*this)(_v, tim::trait::component_apis_t<Type>{});
    }
};

template <>
struct component_categories<void>
{
    template <size_t... Idx>
    void operator()(std::set<std::string>& _v, std::index_sequence<Idx...>) const
    {
        TIMEMORY_FOLD_EXPRESSION(component_categories<comp::enumerator_t<Idx>>{}(_v));
    }

    void operator()(std::set<std::string>& _v) const
    {
        (*this)(_v, std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
    }

    auto operator()() const
    {
        std::set<std::string> _categories{};
        (*this)(_categories);
        return _categories;
    }
};
