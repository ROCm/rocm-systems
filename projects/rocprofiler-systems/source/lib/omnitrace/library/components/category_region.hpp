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
#include "library/defines.hpp"
#include "library/timemory.hpp"
#include "library/tracing.hpp"

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
namespace audit = ::tim::audit;

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
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    // unconditionally return if finalized
    if(get_state() == State::Finalized)
    {
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            tracing::debug_user, "omnitrace_push_region(%s) called during finalization\n",
            name.data());
        return;
    }

    OMNITRACE_CONDITIONAL_BASIC_PRINT(tracing::debug_push, "omnitrace_push_region(%s)\n",
                                      name.data());

    // the expectation here is that if the state is not active then the call
    // to omnitrace_init_tooling_hidden will activate all the appropriate
    // tooling one time and as it exits set it to active and return true.
    if(get_state() != State::Active && !omnitrace_init_tooling_hidden())
    {
        static auto _debug = get_debug_env() || get_debug_init();
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            _debug, "omnitrace_push_region(%s) ignored :: not active. state = %s\n",
            name.data(), std::to_string(get_state()).c_str());
        return;
    }

    OMNITRACE_DEBUG("[%s] omnitrace_push_region(%s)\n", category_name, name.data());

    auto _use_timemory = get_use_timemory();
    auto _use_perfetto = get_use_perfetto();

    constexpr bool _ct_use_timemory =
        (sizeof...(OptsT) == 0 ||
         tim::is_one_of<quirk::timemory, tim::type_list<OptsT...>>::value);

    constexpr bool _ct_use_perfetto =
        (sizeof...(OptsT) == 0 ||
         tim::is_one_of<quirk::perfetto, tim::type_list<OptsT...>>::value);

    if(_use_timemory || _use_perfetto) tracing::thread_init();

    if(_use_perfetto)
    {
        if constexpr(_ct_use_perfetto)
        {
            tracing::push_perfetto(CategoryT{}, name.data(), std::forward<Args>(args)...);
        }
    }
    if(_use_timemory)
    {
        if constexpr(_ct_use_timemory)
        {
            tracing::push_timemory(name.data(), std::forward<Args>(args)...);
        }
    }
    if(_use_timemory || _use_perfetto) tracing::thread_init_sampling();
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::stop(std::string_view name, Args&&... args)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

    constexpr bool _ct_use_timemory =
        (sizeof...(OptsT) == 0 ||
         tim::is_one_of<quirk::timemory, tim::type_list<OptsT...>>::value);

    constexpr bool _ct_use_perfetto =
        (sizeof...(OptsT) == 0 ||
         tim::is_one_of<quirk::perfetto, tim::type_list<OptsT...>>::value);

    // only execute when active
    if(get_state() == State::Active)
    {
        OMNITRACE_CONDITIONAL_PRINT(tracing::debug_pop || get_debug(),
                                    "omnitrace_pop_region(%s)\n", name.data());
        if(get_use_timemory())
        {
            if constexpr(_ct_use_timemory)
            {
                tracing::pop_timemory(name.data(), std::forward<Args>(args)...);
            }
        }
        if(get_use_perfetto())
        {
            if constexpr(_ct_use_perfetto)
            {
                tracing::pop_perfetto(CategoryT{}, name.data(),
                                      std::forward<Args>(args)...);
            }
        }
    }
    else
    {
        static auto _debug = get_debug_env();
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            _debug, "omnitrace_pop_region(%s) ignored :: state = %s\n", name.data(),
            std::to_string(get_state()).c_str());
    }
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(const gotcha_data_t& _data, audit::incoming,
                                  Args&&... _args)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    start<OptsT...>(
        _data.tool_id.c_str(), "args",
        JOIN(", ",
             JOIN('=', tim::try_demangle<std::remove_reference_t<Args>>(), _args)...));
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(const gotcha_data_t& _data, audit::outgoing,
                                  Args&&... _args)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    stop<OptsT...>(_data.tool_id.c_str(), "return", JOIN(", ", _args...));
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(std::string_view _name, audit::incoming,
                                  Args&&... _args)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    start<OptsT...>(
        _name.data(), "args",
        JOIN(", ",
             JOIN('=', tim::try_demangle<std::remove_reference_t<Args>>(), _args)...));
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(std::string_view _name, audit::outgoing,
                                  Args&&... _args)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    stop<OptsT...>(_name.data(), "return", JOIN(", ", _args...));
}

template <typename CategoryT>
template <typename... OptsT, typename... Args>
void
category_region<CategoryT>::audit(quirk::config<OptsT...>, Args&&... _args)
{
    audit<OptsT...>(std::forward<Args>(_args)...);
}
}  // namespace component
}  // namespace omnitrace
