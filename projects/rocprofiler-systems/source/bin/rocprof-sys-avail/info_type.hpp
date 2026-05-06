// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common.hpp"

#include <timemory/enum.h>
#include <timemory/utility/macros.hpp>

#include <utility>

struct info_type : info_type_base
{
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
