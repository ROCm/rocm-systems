// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common.hpp"
#include "core/config.hpp"
#include "core/state.hpp"
#include "library/components/category_region.hpp"
#include "library/components/pthread_mutex_gotcha.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <timemory/backends/threading.hpp>
#include <timemory/components/gotcha/backends.hpp>
#include <timemory/components/gotcha/components.hpp>
#include <timemory/mpl/macros.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/process/threading.hpp>
#include <timemory/settings/settings.hpp>
#include <timemory/utility/types.hpp>

namespace rocprofsys
{
struct default_pthread_mutex_policy
{
    using gotcha_data_t = tim::component::gotcha_data;
    using component_t   = component::pthread_mutex_gotcha<default_pthread_mutex_policy>;
    using gotcha_t =
        comp::gotcha<component_t::gotcha_capacity, std::tuple<>, component_t>;
    using bundle_t = component::category_region<category::pthread>;

    static bool settings_enabled() { return tim::settings::enabled(); }
    static bool get_use_causal() { return ::rocprofsys::get_use_causal(); }
    static bool get_trace_locks() { return config::get_trace_thread_locks(); }
    static bool get_trace_rwlocks() { return config::get_trace_thread_rwlocks(); }
    static bool get_trace_spin_locks() { return config::get_trace_thread_spin_locks(); }
    static bool get_trace_barriers() { return config::get_trace_thread_barriers(); }
    static bool get_trace_join() { return config::get_trace_thread_join(); }

    static bool inactive_state() { return get_state() != State::Active; }

    static bool is_disabled_check()
    {
        static thread_local const auto& t_info = thread_info::get();
        return (!t_info || t_info->is_offset ||
                get_state() != ::rocprofsys::State::Active ||
                get_thread_state() != ThreadState::Enabled);
    }

    static uintptr_t get_thread_id() { return threading::get_id(); }

    template <typename... Args>
    static void audit_incoming(std::string_view id, Args&&... args)
    {
        bundle_t::audit(id, tim::audit::incoming{}, std::forward<Args>(args)...);
    }

    static void audit_outgoing(std::string_view id, int ret)
    {
        bundle_t::audit(id, tim::audit::outgoing{}, ret);
    }
};

namespace component
{
using pthread_mutex_gotcha_t = default_pthread_mutex_policy::gotcha_t;
}  // namespace component
}  // namespace rocprofsys

ROCPROFSYS_DEFINE_CONCRETE_TRAIT(fast_gotcha, component::pthread_mutex_gotcha_t,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(static_data, component::pthread_mutex_gotcha_t,
                                 true_type)

namespace tim::policy
{
using pthread_mutex_gotcha = ::rocprofsys::component::pthread_mutex_gotcha<
    ::rocprofsys::default_pthread_mutex_policy>;
using ::rocprofsys::component::pthread_mutex_gotcha_t;

template <>
struct static_data<pthread_mutex_gotcha, pthread_mutex_gotcha_t> : std::true_type
{
    template <size_t N>
    pthread_mutex_gotcha& operator()(
        std::integral_constant<size_t, N> /*unused*/,
        const ::tim::component::gotcha_data& component_gotcha_data) const
    {
        using thread_data_t                = rocprofsys::thread_data<pthread_mutex_gotcha,
                                                                     std::integral_constant<size_t, N>>;
        static thread_local auto& instance = thread_data_t::instance(
            rocprofsys::construct_on_thread{}, component_gotcha_data);
        return *instance;
    }
};
}  // namespace tim::policy
