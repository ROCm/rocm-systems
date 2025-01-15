// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/config.hpp"
#include "core/defines.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "library/causal/data.hpp"
#include "library/runtime.hpp"
#include "library/tracing.hpp"
#include "library/tracing/annotation.hpp"

#include <timemory/components/gotcha/backends.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/utility/types.hpp>

#include <string_view>

namespace tim
{
namespace quirk
{
struct causal : concepts::quirk_type
{};

struct perfetto : concepts::quirk_type
{};

struct timemory : concepts::quirk_type
{};
}  // namespace quirk
}  // namespace tim

namespace rocprofsys
{
namespace component
{
using tim::is_one_of;
using tim::type_list;

// these categories increment push/pop counts, which are used for sanity checks since
// they should ALWAYS be popped if they were pushed
using tracing_count_categories_t =
    type_list<category::host, category::mpi, category::pthread, category::rocm_hip_api,
              category::rocm_hsa_api, category::rocm_rccl>;

// convert these categories to throughput points
using causal_throughput_categories_t =
    type_list<category::host, category::kokkos, category::ompt, category::rocm_hip_api,
              category::rocm_hsa_api, category::rocm_rccl, category::rocm_marker_api>;

// define this outside of category region functions so that the
// static thread_local is global instead of per-template instantiation
inline ThreadState
get_thread_status()
{
    static thread_local auto _thread_init_once = std::once_flag{};
    std::call_once(_thread_init_once, tracing::thread_init);

    return get_thread_state();
}

// timemory component which calls rocprof-sys functions
// (used in gotcha wrappers)
template <typename CategoryT>
struct category_region : comp::base<category_region<CategoryT>, void>
{
    using gotcha_data_t = tim::component::gotcha_data;

    static constexpr auto category_name = trait::name<CategoryT>::value;

    static std::string label()
    {
        return JOIN('_', "rocprofsys", category_name, "region");
    }

    template <typename... OptsT, typename... Args>
    static void start(std::string_view name, Args&&...);

    template <typename... OptsT, typename... Args>
    static void stop(std::string_view name, Args&&...);

    template <typename... OptsT, typename... Args>
    static void mark(std::string_view name, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(const gotcha_data_t&, audit::incoming, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(const gotcha_data_t&, audit::outgoing, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(std::string_view, audit::incoming, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(std::string_view, audit::outgoing, Args&&...);

    template <typename... OptsT, typename... Args>
    static void audit(quirk::config<OptsT...>, Args&&...);
};

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::start(std::string_view name, Args&&... args)
{
    // skip if category is disabled
    if(tracing::category_push_disabled<CategoryT>()) return;

    // unconditionally return if thread is disabled or finalized
    if(get_thread_state() == ThreadState::Disabled) return;
    if(get_state() >= State::Finalized) return;

    if(name.empty()) return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    // the expectation here is that if the state is not active then the call
    // to rocprofsys_init_tooling_hidden will activate all the appropriate
    // tooling one time and as it exits set it to active and return true.
    if(get_state() != State::Active && !rocprofsys_init_tooling_hidden()) return;

    if(get_thread_status() == ThreadState::Disabled) return;

    constexpr bool _ct_use_timemory =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::timemory, type_list<OptsT...>>::value);

    constexpr bool _ct_use_perfetto =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::perfetto, type_list<OptsT...>>::value);

    constexpr bool _ct_use_causal =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::causal, type_list<OptsT...>>::value);

    ROCPROFSYS_CONDITIONAL_PRINT(
        tracing::debug_push,
        "[%s][PID=%i][state=%s][thread_state=%s] rocprofsys_push_region(%s)\n",
        category_name, process::get_id(), std::to_string(get_state()).c_str(),
        std::to_string(get_thread_state()).c_str(), name.data());

    if constexpr(is_one_of<CategoryT, tracing_count_categories_t>::value)
    {
        ++tracing::push_count();
    }

    auto _hash = tim::add_hash_id(name);
    name       = tim::get_hash_identifier_fast(_hash);

    if constexpr(_ct_use_causal)
    {
        if constexpr(!is_one_of<CategoryT, causal_throughput_categories_t>::value)
        {
            if(get_use_causal()) causal::push_progress_point(name);
        }
    }

    if constexpr(_ct_use_timemory)
    {
        if(get_use_timemory())
        {
            tracing::push_timemory(CategoryT{}, name, std::forward<Args>(args)...);
        }
    }

    if constexpr(_ct_use_perfetto)
    {
        if(get_use_perfetto())
        {
            tracing::push_perfetto(CategoryT{}, name.data(), std::forward<Args>(args)...);
        }
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::stop(std::string_view name, Args&&... args)
{
    // skip if category is disabled
    if(tracing::category_pop_disabled<CategoryT>()) return;

    if(get_thread_state() == ThreadState::Disabled) return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    constexpr bool _ct_use_timemory =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::timemory, type_list<OptsT...>>::value);

    constexpr bool _ct_use_perfetto =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::perfetto, type_list<OptsT...>>::value);

    constexpr bool _ct_use_causal =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::causal, type_list<OptsT...>>::value);

    ROCPROFSYS_CONDITIONAL_PRINT(
        tracing::debug_pop,
        "[%s][PID=%i][state=%s][thread_state=%s] rocprofsys_pop_region(%s)\n",
        category_name, process::get_id(), std::to_string(get_state()).c_str(),
        std::to_string(get_thread_state()).c_str(), name.data());

    // only execute when active
    if(get_state() == State::Active)
    {
        if constexpr(is_one_of<CategoryT, tracing_count_categories_t>::value)
        {
            ++tracing::pop_count();
        }

        if constexpr(_ct_use_perfetto)
        {
            if(get_use_perfetto())
            {
                tracing::pop_perfetto(CategoryT{}, name.data(),
                                      std::forward<Args>(args)...);
            }
        }

        if constexpr(_ct_use_timemory)
        {
            if(get_use_timemory())
            {
                tracing::pop_timemory(CategoryT{}, name, std::forward<Args>(args)...);
            }
        }

        if constexpr(_ct_use_causal)
        {
            if constexpr(is_one_of<CategoryT, causal_throughput_categories_t>::value)
            {
                if(get_use_causal()) causal::mark_progress_point(name);
            }
            else
            {
                if(get_use_causal()) causal::pop_progress_point(name);
            }
        }
    }
    else
    {
        static auto _debug = get_debug_env();
        ROCPROFSYS_CONDITIONAL_BASIC_PRINT(
            _debug, "[%s] rocprofsys_pop_region(%s) ignored :: state = %s\n",
            category_name, name.data(), std::to_string(get_state()).c_str());
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::mark(std::string_view name, Args&&...)
{
    constexpr bool _ct_use_causal =
        (sizeof...(OptsT) == 0 || is_one_of<quirk::causal, type_list<OptsT...>>::value);

    if constexpr(!_ct_use_causal) return;

    // skip if category is disabled
    if(tracing::category_mark_disabled<CategoryT>()) return;

    // the expectation here is that if the state is not active then the call
    // to rocprofsys_init_tooling_hidden will activate all the appropriate
    // tooling one time and as it exits set it to active and return true.
    if(get_state() != State::Active && !rocprofsys_init_tooling_hidden()) return;

    // unconditionally return if thread is disabled or finalized
    if(get_thread_state() >= ThreadState::Completed) return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    if(get_use_causal())
    {
        ROCPROFSYS_CONDITIONAL_PRINT(
            tracing::debug_mark,
            "[%s][PID=%i][state=%s][thread_state=%s] rocprofsys_progress(%s)\n",
            category_name, process::get_id(), std::to_string(get_state()).c_str(),
            std::to_string(get_thread_state()).c_str(), name.data());

        causal::mark_progress_point(name);
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(const gotcha_data_t& _data, audit::incoming,
                                  Args&&... _args)
{
    start<OptsT...>(_data.tool_id.c_str(), [&](::perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
        {
            int64_t _n = 0;
            ROCPROFSYS_FOLD_EXPRESSION(tracing::add_perfetto_annotation(
                ctx, tim::try_demangle<std::remove_reference_t<Args>>(), _args, _n++));
        }
    });
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(const gotcha_data_t& _data, audit::outgoing,
                                  Args&&... _args)
{
    stop<OptsT...>(_data.tool_id.c_str(), [&](::perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
            tracing::add_perfetto_annotation(ctx, "return", JOIN(", ", _args...));
    });
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(std::string_view _name, audit::incoming,
                                  Args&&... _args)
{
    start<OptsT...>(_name.data(), [&](::perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
        {
            int64_t _n = 0;
            ROCPROFSYS_FOLD_EXPRESSION(tracing::add_perfetto_annotation(
                ctx, tim::try_demangle<std::remove_reference_t<Args>>(), _args, _n++));
        }
    });
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(std::string_view _name, audit::outgoing,
                                  Args&&... _args)
{
    stop<OptsT...>(_name.data(), [&](::perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
            tracing::add_perfetto_annotation(ctx, "return", JOIN(", ", _args...));
    });
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(quirk::config<OptsT...>, Args&&... _args)
{
    audit<OptsT...>(std::forward<Args>(_args)...);
}

template <typename CategoryT>
struct local_category_region : comp::base<local_category_region<CategoryT>, void>
{
    using impl_type = category_region<CategoryT>;

    static constexpr auto category_name = impl_type::category_name;
    static std::string    label() { return impl_type::label(); }

    template <typename... OptsT, typename... Args>
    auto start(Args&&... args)
    {
        if(m_prefix.empty()) return;
        return impl_type::template start<OptsT...>(m_prefix, std::forward<Args>(args)...);
    }

    template <typename... OptsT, typename... Args>
    auto stop(Args&&... args)
    {
        if(m_prefix.empty()) return;
        return impl_type::template stop<OptsT...>(m_prefix, std::forward<Args>(args)...);
    }

    template <typename... OptsT, typename... Args>
    auto mark(Args&&... args)
    {
        if(m_prefix.empty()) return;
        return impl_type::template mark<OptsT...>(m_prefix, std::forward<Args>(args)...);
    }

    template <typename... OptsT, typename... Args>
    auto audit(Args&&... args)
        -> decltype(impl_type::template audit<OptsT...>(std::declval<std::string_view>(),
                                                        std::forward<Args>(args)...))
    {
        if(m_prefix.empty()) return;
        return impl_type::template audit<OptsT...>(m_prefix, std::forward<Args>(args)...);
    }

    template <typename... OptsT, typename... Args>
    auto audit(quirk::config<OptsT...>, Args&&... args)
    {
        if(m_prefix.empty()) return;
        return impl_type::template audit<OptsT...>(quirk::config<OptsT...>{}, m_prefix,
                                                   std::forward<Args>(args)...);
    }

    void set_prefix(std::string_view _v) { m_prefix = _v; }

private:
    std::string_view m_prefix = {};
};
}  // namespace component
}  // namespace rocprofsys
