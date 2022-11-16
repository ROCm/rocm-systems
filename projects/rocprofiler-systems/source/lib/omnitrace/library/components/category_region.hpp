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

#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/defines.hpp"
#include "library/runtime.hpp"
#include "library/timemory.hpp"
#include "library/tracing.hpp"
#include "library/tracing/annotation.hpp"

#include <timemory/components/gotcha/backends.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/utility/types.hpp>

#include <string_view>

namespace tim
{
namespace quirk
{
struct perfetto : concepts::quirk_type
{};

struct timemory : concepts::quirk_type
{};
}  // namespace quirk
}  // namespace tim

namespace omnitrace
{
namespace component
{
// timemory component which calls omnitrace functions
// (used in gotcha wrappers)
template <typename CategoryT>
struct category_region : comp::base<category_region<CategoryT>, void>
{
    using gotcha_data_t = tim::component::gotcha_data;

    static constexpr auto category_name = trait::name<CategoryT>::value;

    static std::string label() { return JOIN('_', "omnitrace", category_name, "region"); }

    template <typename... OptsT, typename... Args>
    static void start(std::string_view name, Args&&...);

    template <typename... OptsT, typename... Args>
    static void stop(std::string_view name, Args&&...);

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
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    // unconditionally return if thread is disabled or finalized
    if(get_thread_state() == ThreadState::Disabled) return;
    if(get_state() == State::Finalized) return;

    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    // the expectation here is that if the state is not active then the call
    // to omnitrace_init_tooling_hidden will activate all the appropriate
    // tooling one time and as it exits set it to active and return true.
    if(get_state() != State::Active && !omnitrace_init_tooling_hidden()) return;

    tracing::thread_init();

    // thread initialization may have disabled the thread
    if(get_thread_state() == ThreadState::Disabled) return;

    tracing::thread_init_sampling();

    constexpr bool _ct_use_timemory =
        (sizeof...(OptsT) == 0 ||
         tim::is_one_of<quirk::timemory, tim::type_list<OptsT...>>::value);

    constexpr bool _ct_use_perfetto =
        (sizeof...(OptsT) == 0 ||
         tim::is_one_of<quirk::perfetto, tim::type_list<OptsT...>>::value);

    OMNITRACE_CONDITIONAL_PRINT(tracing::debug_push,
                                "[%s][PID=%i][state=%s] omnitrace_push_region(%s)\n",
                                category_name, process::get_id(),
                                std::to_string(get_state()).c_str(), name.data());

    if constexpr(tim::is_one_of<CategoryT, tim::type_list<category::host>>::value)
    {
        ++tracing::push_count();
    }

    if constexpr(_ct_use_perfetto)
    {
        if(get_use_perfetto())
        {
            tracing::push_perfetto(CategoryT{}, name.data(), std::forward<Args>(args)...);
        }
    }

    if constexpr(_ct_use_timemory)
    {
        if(get_use_timemory())
        {
            tracing::push_timemory(CategoryT{}, name.data(), std::forward<Args>(args)...);
        }
    }

    if constexpr(tim::is_one_of<CategoryT, tim::type_list<category::host>>::value)
    {
        using Device = critical_trace::Device;
        using Phase  = critical_trace::Phase;

        if(get_use_critical_trace())
        {
            uint64_t _cid                       = 0;
            uint64_t _parent_cid                = 0;
            uint32_t _depth                     = 0;
            std::tie(_cid, _parent_cid, _depth) = create_cpu_cid_entry();
            auto _ts                            = comp::wall_clock::record();
            add_critical_trace<Device::CPU, Phase::BEGIN>(
                threading::get_id(), _cid, 0, _parent_cid, _ts, 0, 0, 0,
                critical_trace::add_hash_id(name.data()), _depth);
        }
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::stop(std::string_view name, Args&&... args)
{
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    if(get_thread_state() == ThreadState::Disabled) return;

    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    constexpr bool _ct_use_timemory =
        (sizeof...(OptsT) == 0 ||
         tim::is_one_of<quirk::timemory, tim::type_list<OptsT...>>::value);

    constexpr bool _ct_use_perfetto =
        (sizeof...(OptsT) == 0 ||
         tim::is_one_of<quirk::perfetto, tim::type_list<OptsT...>>::value);

    OMNITRACE_CONDITIONAL_PRINT(tracing::debug_pop,
                                "[%s][PID=%i][state=%s] omnitrace_pop_region(%s)\n",
                                category_name, process::get_id(),
                                std::to_string(get_state()).c_str(), name.data());

    // only execute when active
    if(get_state() == State::Active)
    {
        if constexpr(tim::is_one_of<CategoryT, tim::type_list<category::host>>::value)
        {
            ++tracing::pop_count();
        }

        if constexpr(_ct_use_timemory)
        {
            if(get_use_timemory())
            {
                tracing::pop_timemory(CategoryT{}, name.data(),
                                      std::forward<Args>(args)...);
            }
        }

        if constexpr(_ct_use_perfetto)
        {
            if(get_use_perfetto())
            {
                tracing::pop_perfetto(CategoryT{}, name.data(),
                                      std::forward<Args>(args)...);
            }
        }

        if constexpr(tim::is_one_of<CategoryT, tim::type_list<category::host>>::value)
        {
            using Device = critical_trace::Device;
            using Phase  = critical_trace::Phase;

            if(get_use_critical_trace())
            {
                if(get_cpu_cid_stack() && !get_cpu_cid_stack()->empty())
                {
                    auto _cid = get_cpu_cid_stack()->back();
                    if(get_cpu_cid_parents()->find(_cid) != get_cpu_cid_parents()->end())
                    {
                        uint64_t _parent_cid          = 0;
                        uint32_t _depth               = 0;
                        auto     _ts                  = comp::wall_clock::record();
                        std::tie(_parent_cid, _depth) = get_cpu_cid_parents()->at(_cid);
                        add_critical_trace<Device::CPU, Phase::END>(
                            threading::get_id(), _cid, 0, _parent_cid, _ts, _ts, 0, 0,
                            critical_trace::add_hash_id(name.data()), _depth);
                    }
                }
            }
        }
    }
    else
    {
        static auto _debug = get_debug_env();
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            _debug, "[%s] omnitrace_pop_region(%s) ignored :: state = %s\n",
            category_name, name.data(), std::to_string(get_state()).c_str());
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(const gotcha_data_t& _data, audit::incoming,
                                  Args&&... _args)
{
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    start<OptsT...>(_data.tool_id.c_str(), [&](perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
        {
            int64_t _n = 0;
            OMNITRACE_FOLD_EXPRESSION(tracing::add_perfetto_annotation(
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
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    stop<OptsT...>(_data.tool_id.c_str(), [&](perfetto::EventContext ctx) {
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
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    start<OptsT...>(_name.data(), [&](perfetto::EventContext ctx) {
        if(config::get_perfetto_annotations())
        {
            int64_t _n = 0;
            OMNITRACE_FOLD_EXPRESSION(tracing::add_perfetto_annotation(
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
    // skip if category is disabled
    if(!trait::runtime_enabled<CategoryT>::get()) return;

    stop<OptsT...>(_name.data(), [&](perfetto::EventContext ctx) {
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
}  // namespace omnitrace
