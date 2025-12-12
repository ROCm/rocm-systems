// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/constraint.hpp"
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
    // disable specified categories
    disable_categories();

    auto _trace_specs = constraint::get_trace_specs();

    if(!_trace_specs.empty())
    {
        auto _trace_stages = constraint::get_trace_stages();

        _trace_stages.init = [](const constraint::spec& _spec) {
            if(_spec.delay > 1.0e-3) disable_categories(config::get_enabled_categories());
            return get_state() < State::Finalized;
        };

        _trace_stages.start = [](const constraint::spec&) {
            enable_categories(config::get_enabled_categories());
            return get_state() < State::Finalized;
        };

        _trace_stages.stop = [](const constraint::spec&) {
            // only disable categories if not finalized since this might run in background
            // during finalization and disable output of data in those categories
            if(get_state() < State::Finalized)
                disable_categories(config::get_enabled_categories());
            return get_state() < State::Finalized;
        };

        auto _promise = std::promise<void>();
        std::thread{ [_trace_specs, _trace_stages](std::promise<void>* _prom) {
                        // ensure all categories are disabled before proceeding
                        // if a delay is requested
                        if(_trace_specs.front().delay > 1.0e-3)
                            disable_categories(config::get_enabled_categories());
                        _prom->set_value();
                        for(const auto& itr : _trace_specs)
                            itr(_trace_stages);
                    },
                     &_promise }
            .detach();

        _promise.get_future().wait_for(std::chrono::seconds{ 1 });
    }
}

void
shutdown()
{
    disable_categories(config::get_enabled_categories());
}
}  // namespace categories
}  // namespace rocprofsys
