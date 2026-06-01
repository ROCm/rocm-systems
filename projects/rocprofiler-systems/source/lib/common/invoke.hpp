// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "common/join.hpp"
#include <cstdint>

#include <atomic>
#include <cstring>
#include <functional>
#include <string>
#include <unistd.h>

#if !defined(ROCPROFSYS_COMMON_LIBRARY_NAME)
#    define ROCPROFSYS_COMMON_LIBRARY_NAME "common"
#    error ROCPROFSYS_COMMON_LIBRARY_NAME must be defined
#endif

#if !defined(ROCPROFSYS_COMMON_LIBRARY_LOG_START)
#    define ROCPROFSYS_COMMON_LIBRARY_LOG_START
#endif

#if !defined(ROCPROFSYS_COMMON_LIBRARY_LOG_END)
#    define ROCPROFSYS_COMMON_LIBRARY_LOG_END
#endif

namespace rocprofsys
{
inline namespace common
{
namespace
{
template <typename FuncT, typename... Args>
inline auto
invoke(const char* _name, FuncT&& _func, Args... _args) ROCPROFSYS_HIDDEN_API;

inline std::int32_t&
get_guard()
{
    static thread_local std::int32_t _v = 0;
    return _v;
}

inline std::int64_t
get_thread_index()
{
    static std::atomic<std::int64_t> _c{ 0 };
    static thread_local auto         _v = _c++;
    return _v;
}

template <typename... Args>
auto
ignore(const char* _name, int _verbose, int _value, const char* _reason, Args... _args)
{
    if(_verbose >= _value)
    {
        fflush(stderr);
        fprintf(stderr,
                "[rocprof-sys][" ROCPROFSYS_COMMON_LIBRARY_NAME
                "][%i][%li] %s(%s) was ignored :: %s\n",
                getpid(), get_thread_index(), _name,
                join(QuoteStrings{}, ", ", _args...).c_str(), _reason);
        fflush(stderr);
    }
}

template <typename FuncT, typename... Args>
auto
invoke(const char* _name, int _verbose, bool& _toggle, FuncT&& _func, Args... _args)
{
    if(_func)
    {
        struct decrement_guard
        {
            // decrement the guard as it exits the scope
            ~decrement_guard() { --get_guard(); }
        } _unlk{};

        // if _lk is ever greater than zero on the same thread, this
        // means a function within the current function is calling
        // our instrumentation so we ignore the call
        std::int32_t _lk = get_guard()++;
        if(_lk == 0)
        {
            _toggle = !_toggle;
            if(_verbose >= 3)
            {
                fflush(stderr);
                ROCPROFSYS_COMMON_LIBRARY_LOG_START
                fprintf(stderr,
                        "[rocprof-sys][" ROCPROFSYS_COMMON_LIBRARY_NAME
                        "][%i][%li][%i] %s(%s)\n",
                        getpid(), get_thread_index(), _lk, _name,
                        join(QuoteStrings{}, ", ", _args...).c_str());
                ROCPROFSYS_COMMON_LIBRARY_LOG_END
                fflush(stderr);
            }
            return std::invoke(std::forward<FuncT>(_func), _args...);
        }
        else if(_verbose >= 2)
        {
            fflush(stderr);
            ROCPROFSYS_COMMON_LIBRARY_LOG_START
            fprintf(stderr,
                    "[rocprof-sys][" ROCPROFSYS_COMMON_LIBRARY_NAME
                    "][%i][%li] %s(%s) was guarded :: value = %i\n",
                    getpid(), get_thread_index(), _name,
                    join(QuoteStrings{}, ", ", _args...).c_str(), _lk);
            ROCPROFSYS_COMMON_LIBRARY_LOG_END
            fflush(stderr);
        }
    }
    else if(_verbose >= 0)
    {
        ROCPROFSYS_COMMON_LIBRARY_LOG_START
        fprintf(stderr,
                "[rocprof-sys][" ROCPROFSYS_COMMON_LIBRARY_NAME
                "][%i][%li] %s(%s) ignored :: null function pointer\n",
                getpid(), get_thread_index(), _name,
                join(QuoteStrings{}, ", ", _args...).c_str());
        ROCPROFSYS_COMMON_LIBRARY_LOG_END
    }

    using return_type = decltype(std::invoke(std::forward<FuncT>(_func), _args...));
    if constexpr(!std::is_void<return_type>::value) return return_type();
}
}  // namespace
}  // namespace common
}  // namespace rocprofsys
