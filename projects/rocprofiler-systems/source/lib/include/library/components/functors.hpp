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

#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "library/timemory.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/utility/macros.hpp"

#include <type_traits>
#include <utility>

namespace omnitrace
{
namespace component
{
template <bool Cond, typename Tp>
using enable_if_t = typename std::enable_if<Cond, Tp>::type;

template <typename... Tp>
static auto get_default_functor(tim::type_list<Tp...>)
{
    return [](Tp...) {};
};

// timemory component which calls omnitrace functions
// (used in gotcha wrappers)
template <typename ApiT, typename StartFuncT, typename StopFuncT>
struct functors : comp::base<functors<ApiT, StartFuncT, StopFuncT>, void>
{
    using this_type = functors<ApiT, StartFuncT, StopFuncT>;
    using base_type = comp::base<functors<ApiT, StartFuncT, StopFuncT>, void>;
    using pair_type = std::pair<StartFuncT, StopFuncT>;

    static constexpr bool begin_supports_cstr =
        std::is_invocable<StartFuncT, const char*>::value;
    static constexpr bool end_supports_cstr =
        std::is_invocable<StopFuncT, const char*>::value;

    static constexpr bool begin_supports_void = std::is_invocable<StartFuncT>::value;
    static constexpr bool end_supports_void   = std::is_invocable<StopFuncT>::value;

    static void        preinit();
    static void        configure(StartFuncT&& _beg, StopFuncT&& _end);
    static std::string label();

    template <typename... Args, enable_if_t<((sizeof...(Args) > 0) &&
                                             std::is_invocable_v<StartFuncT, Args...>),
                                            int> = 0>
    static auto start(Args&&... _args)
    {
        get_functors().first(std::forward<Args>(_args)...);
    }

    template <typename... Args, enable_if_t<((sizeof...(Args) > 0) &&
                                             std::is_invocable_v<StopFuncT, Args...>),
                                            int> = 0>
    static auto stop(Args&&... _args)
    {
        get_functors().second(std::forward<Args>(_args)...);
    }

    TIMEMORY_DEFAULT_OBJECT(functors)

    template <typename Tp = this_type, enable_if_t<Tp::begin_supports_cstr, int> = 0>
    void start()
    {
        get_functors().first(m_prefix);
    }

    template <typename Tp = this_type, enable_if_t<Tp::end_supports_cstr, int> = 0>
    void stop()
    {
        get_functors().second(m_prefix);
    }

    template <typename Tp = this_type, enable_if_t<Tp::begin_supports_void, int> = 0>
    void start()
    {
        get_functors().first();
    }

    template <typename Tp = this_type, enable_if_t<Tp::end_supports_void, int> = 0>
    void stop()
    {
        get_functors().second();
    }

    template <typename Tp = this_type,
              enable_if_t<Tp::begin_supports_cstr || Tp::end_supports_cstr, int> = 0>
    void set_prefix(const char* _v)
    {
        m_prefix = _v;
    }

private:
    static bool&      is_configured();
    static pair_type& get_functors();

private:
    const char* m_prefix = nullptr;
};

template <typename ApiT, typename StartFuncT, typename StopFuncT>
void
functors<ApiT, StartFuncT, StopFuncT>::preinit()
{
    using start_args_t    = typename tim::mpl::function_traits<StartFuncT>::args_type;
    using stop_args_t     = typename tim::mpl::function_traits<StopFuncT>::args_type;
    get_functors().first  = get_default_functor(start_args_t{});
    get_functors().second = get_default_functor(stop_args_t{});
}

template <typename ApiT, typename StartFuncT, typename StopFuncT>
void
functors<ApiT, StartFuncT, StopFuncT>::configure(StartFuncT&& _beg, StopFuncT&& _end)
{
    is_configured()       = true;
    get_functors().first  = std::forward<StartFuncT>(_beg);
    get_functors().second = std::forward<StopFuncT>(_end);
}

template <typename ApiT, typename StartFuncT, typename StopFuncT>
std::string
functors<ApiT, StartFuncT, StopFuncT>::label()
{
    return trait::name<this_type>::value;
}

template <typename ApiT, typename StartFuncT, typename StopFuncT>
bool&
functors<ApiT, StartFuncT, StopFuncT>::is_configured()
{
    static bool _v = false;
    return _v;
}

template <typename ApiT, typename StartFuncT, typename StopFuncT>
typename functors<ApiT, StartFuncT, StopFuncT>::pair_type&
functors<ApiT, StartFuncT, StopFuncT>::get_functors()
{
    static auto _v = pair_type{};
    return _v;
}
}  // namespace component
}  // namespace omnitrace
