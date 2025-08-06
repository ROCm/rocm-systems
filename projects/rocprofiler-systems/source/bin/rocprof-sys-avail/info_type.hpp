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

#include <timemory/enum.h>
#include <timemory/utility/macros.hpp>

#include <utility>

struct info_type : info_type_base
{
    TIMEMORY_DEFAULT_OBJECT(info_type)

    template <typename... Args>
    info_type(Args&&... _args)
    : info_type_base{ std::forward<Args>(_args)... }
    {}

    const auto& name() const { return std::get<0>(*this); }
    auto        is_available() const { return std::get<1>(*this); }
    const auto& info() const { return std::get<2>(*this); }
    const auto& data_type() const { return info().at(0); }
    const auto& enum_type() const { return info().at(1); }
    const auto& id_type() const { return info().at(2); }
    const auto& id_strings() const { return info().at(3); }
    const auto& label() const { return info().at(4); }
    const auto& description() const { return info().at(5); }
    const auto& categories() const { return info().at(6); }

    bool valid() const { return !name().empty() && info().size() >= 6; }

    bool operator<(const info_type& rhs) const { return name() < rhs.name(); }
    bool operator!=(const info_type& rhs) const { return !(*this == rhs); }
    bool operator==(const info_type& rhs) const
    {
        if(info().size() != rhs.info().size()) return false;
        for(size_t i = 0; i < info().size(); ++i)
        {
            if(info().at(i) != rhs.info().at(i)) return false;
        }
        return name() == rhs.name() && is_available() == rhs.is_available();
    }
};

template <size_t EndV>
std::vector<info_type>
get_component_info();

extern template std::vector<info_type>
get_component_info<TIMEMORY_NATIVE_COMPONENTS_END>();

extern template std::vector<info_type>
get_component_info<TIMEMORY_COMPONENTS_END>();
