// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"

#include "logger/debug.hpp"
#include "rocprofiler-systems/categories.h"

#include <fmt/format.h>

#include <array>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace rocprofsys
{
namespace
{
template <size_t... Idx>
constexpr auto
make_category_table(std::index_sequence<Idx...>)
{
    return std::array<category_names, sizeof...(Idx)>{ category_names{
        .name        = tim::trait::name<category_type_id_t<Idx>>::value,
        .description = tim::trait::name<category_type_id_t<Idx>>::description }... };
}

// Index 0 holds the first id after ROCPROFSYS_CATEGORY_NONE.
constexpr auto k_category_table = make_category_table(
    utility::make_index_sequence_range<1, ROCPROFSYS_CATEGORY_LAST>{});
}  // namespace

category_names
category_info(std::size_t category_id)
{
    if(category_id == ROCPROFSYS_CATEGORY_NONE || category_id >= ROCPROFSYS_CATEGORY_LAST)
    {
        throw std::out_of_range(
            fmt::format("category id {} has no category", category_id));
    }
    return k_category_table[category_id - 1];
}
}  // namespace rocprofsys

namespace rocprofsys::categories
{
namespace
{
template <typename Tp>
void
configure_categories(bool _enable, const std::set<std::string>& _categories)
{
    auto _name = trait::name<Tp>::value;
    if(_categories.count(_name) > 0)
    {
        LOG_DEBUG("{} category: {}", _enable ? "Enabling" : "Disabling", _name);
        trait::runtime_enabled<Tp>::set(_enable);
    }
}

template <size_t... Idx>
void
configure_categories(bool _enable, const std::set<std::string>& _categories,
                     std::index_sequence<Idx...>)
{
    (configure_categories<category_type_id_t<Idx>>(_enable, _categories), ...);
}

void
configure_categories(bool _enable, const std::set<std::string>& _categories)
{
    LOG_DEBUG("{} categories...", (_enable) ? "Enabling" : "Disabling");

    configure_categories(
        _enable, _categories,
        utility::make_index_sequence_range<1, ROCPROFSYS_CATEGORY_LAST>{});
}
}  // namespace

void
enable_categories(const std::set<std::string>& _categories)
{
    configure_categories(
        true, _categories,
        utility::make_index_sequence_range<1, ROCPROFSYS_CATEGORY_LAST>{});
}

void
disable_categories(const std::set<std::string>& _categories)
{
    configure_categories(
        false, _categories,
        utility::make_index_sequence_range<1, ROCPROFSYS_CATEGORY_LAST>{});
}

void
setup()
{
    // disable specified categories
    disable_categories();
}

void
shutdown()
{
    disable_categories(config::get_enabled_categories());
}
}  // namespace rocprofsys::categories
