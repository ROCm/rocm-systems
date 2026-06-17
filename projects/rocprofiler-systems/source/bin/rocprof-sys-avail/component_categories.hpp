// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common.hpp"
#include "core/demangler.hpp"
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

        ROCPROFSYS_FOLD_EXPRESSION(_v.emplace(TIMEMORY_JOIN(
            "::", "component", _cleanup(rocprofsys::utility::demangle<Tp>(), "tim::"))));
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
        ROCPROFSYS_FOLD_EXPRESSION(component_categories<comp::enumerator_t<Idx>>{}(_v));
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
