// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"

#include "logger/debug.hpp"

#include <set>
#include <string>

namespace rocprofsys
{
namespace categories
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
    // disable user-disabled categories. TRACE_DELAY/DURATION gating is
    // owned by the time_window trigger wired into the control session
    // (see library.cpp); recording paths are paused via subsystem subscribers
    // when the session pauses, so no per-category trait flipping is needed
    // here.
    disable_categories();
}

void
shutdown()
{
    disable_categories(config::get_enabled_categories());
}
}  // namespace categories
}  // namespace rocprofsys
