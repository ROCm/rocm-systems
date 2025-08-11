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
#include "get_categories.hpp"
#include "info_type.hpp"

#include <timemory/components/metadata.hpp>
#include <timemory/components/properties.hpp>
#include <timemory/defines.h>
#include <timemory/enum.h>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/type_list.hpp>
#include <timemory/variadic/macros.hpp>

//--------------------------------------------------------------------------------------//

struct unknown
{};

template <typename T, typename U = typename T::value_type>
constexpr bool
available_value_type_alias(int)
{
    return true;
}

template <typename T, typename U = unknown>
constexpr bool
available_value_type_alias(long)
{
    return false;
}

template <typename Type, bool>
struct component_value_type;

template <typename Type>
struct component_value_type<Type, true>
{
    using type = typename Type::value_type;
};

template <typename Type>
struct component_value_type<Type, false>
{
    using type = unknown;
};

template <typename Type>
using component_value_type_t =
    typename component_value_type<Type, available_value_type_alias<Type>(0)>::type;

//--------------------------------------------------------------------------------------//

template <typename Type = void>
struct get_availability;

//--------------------------------------------------------------------------------------//

template <typename Type>
struct get_availability
{
    using this_type  = get_availability<Type>;
    using metadata_t = ::tim::component::metadata<Type>;
    using property_t = ::tim::component::properties<Type>;

    static info_type get_info();
    auto             operator()() const { return get_info(); }
};

//--------------------------------------------------------------------------------------//

template <typename... Types>
struct get_availability<type_list<Types...>>
{
    using data_type = std::vector<info_type>;

    static data_type get_info(data_type& _v)
    {
        TIMEMORY_FOLD_EXPRESSION(_v.emplace_back(get_availability<Types>::get_info()));
        return _v;
    }

    static data_type get_info()
    {
        data_type _v{};
        return get_info(_v);
    }

    template <typename... Args>
    decltype(auto) operator()(Args&&... _args)
    {
        return get_info(std::forward<Args>(_args)...);
    }
};

//--------------------------------------------------------------------------------------//

template <typename Type>
info_type
get_availability<Type>::get_info()
{
    using namespace tim;
    using value_type     = component_value_type_t<Type>;
    using category_types = typename trait::component_apis<Type>::type;

    auto _cleanup = [](std::string _type, const std::string& _pattern) {
        auto _pos = std::string::npos;
        while((_pos = _type.find(_pattern)) != std::string::npos)
            _type.erase(_pos, _pattern.length());
        return _type;
    };
    auto _replace = [](std::string _type, const std::string& _pattern,
                       const std::string& _with) {
        auto _pos = std::string::npos;
        while((_pos = _type.find(_pattern)) != std::string::npos)
            _type.replace(_pos, _pattern.length(), _with);
        return _type;
    };

    bool has_metadata   = metadata_t::specialized();
    bool has_properties = property_t::specialized();
    bool is_available   = trait::is_available<Type>::value;
    bool file_output    = trait::generates_output<Type>::value;
    auto name           = component::metadata<Type>::name();
    auto label          = (file_output)
                              ? ((has_metadata) ? metadata_t::label() : Type::get_label())
                              : std::string("");
    auto description =
        (has_metadata) ? metadata_t::description() : Type::get_description();
    auto     data_type = demangle<value_type>();
    string_t enum_type = property_t::enum_string();
    string_t id_type   = property_t::id();
    auto     ids_set   = property_t::ids();

    if(!has_properties)
    {
        enum_type = "";
        id_type   = "";
        ids_set.clear();
    }
    string_t ids_str = {};
    {
        auto     itr = ids_set.begin();
        string_t db  = (markdown) ? "`\"" : (csv) ? "" : "\"";
        string_t de  = (markdown) ? "\"`" : (csv) ? "" : "\"";
        if(has_metadata) description += ". " + metadata_t::extra_description();
        description += ".";
        while(itr->empty())
            ++itr;
        if(itr != ids_set.end())
            ids_str = TIMEMORY_JOIN("", TIMEMORY_JOIN("", db, *itr++, de));
        for(; itr != ids_set.end(); ++itr)
        {
            if(!itr->empty())
                ids_str = TIMEMORY_JOIN(", ", ids_str, TIMEMORY_JOIN("", db, *itr, de));
        }
    }

    string_t categories = get_categories(category_types{});
    description         = _replace(_replace(description, ". .", "."), "..", ".");
    data_type           = _replace(_cleanup(data_type, "::__1"), "> >", ">>");
    return info_type{ name, is_available,
                      str_vec_t{ data_type, enum_type, id_type, ids_str, label,
                                 description, categories } };
}

//--------------------------------------------------------------------------------------//

template <>
struct get_availability<void>
{
    template <typename... Tp, typename... Args>
    decltype(auto) operator()(tim::type_list<Tp...>, Args&&... _args) const
    {
        return get_availability<tim::type_list<Tp...>>{}(std::forward<Args>(_args)...);
    }

    template <typename Tp, typename... Args>
    decltype(auto) operator()(Args&&... _args) const
    {
        return get_availability<tim::type_list<Tp>>{}(std::forward<Args>(_args)...);
    }
};
