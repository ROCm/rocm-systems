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

#include "library/defines.hpp"

#include <timemory/api.hpp>
#include <timemory/backends/dmp.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/utility/utility.hpp>

#include <cstdio>
#include <cstring>
#include <string_view>
#include <utility>

namespace omnitrace
{
inline namespace config
{
bool
get_debug();

bool
get_debug_tid();

bool
get_debug_pid();

bool
get_critical_trace_debug();
}  // namespace config

namespace debug
{
namespace
{
template <typename T, size_t... Idx>
auto get_chars(std::index_sequence<Idx...>)
{
    static const char _v[sizeof...(Idx) + 1] = { T::get()[Idx]..., '\0' };
    return _v;
}

template <typename T>
auto
get_chars()
{
    return get_chars<T>(typename T::sequence{});
}
}  // namespace
}  // namespace debug
}  // namespace omnitrace

#define OMNITRACE_VAR_NAME_COMBINE(X, Y) X##Y
#define OMNITRACE_LINESTR                TIMEMORY_STRINGIZE(__LINE__)
#define OMNITRACE_VARIABLE(LABEL)        OMNITRACE_VAR_NAME_COMBINE(_omni_var_, LABEL)

#if defined(TIMEMORY_USE_MPI)
#    define OMNITRACE_PROCESS_IDENTIFIER static_cast<int>(::tim::dmp::rank())
#elif defined(TIMEMORY_USE_MPI_HEADERS)
#    define OMNITRACE_PROCESS_IDENTIFIER                                                 \
        (::tim::dmp::is_initialized()) ? static_cast<int>(::tim::dmp::rank())            \
                                       : static_cast<int>(::tim::process::get_id())
#else
#    define OMNITRACE_PROCESS_IDENTIFIER static_cast<int>(::tim::process::get_id())
#endif

#define OMNITRACE_FUNCTION                                                               \
    std::string{ __FUNCTION__ }                                                          \
        .substr(0, std::string_view{ __FUNCTION__ }.find("_hidden"))                     \
        .c_str()

#define OMNITRACE_CT_FUNCTION(VAR, STR)                                                  \
    static constexpr auto OMNITRACE_VARIABLE(__LINE__) = std::string_view{ STR };        \
    VAR = OMNITRACE_CT_FUNCTION_IMPL(OMNITRACE_VARIABLE(__LINE__));

#define OMNITRACE_CT_FUNCTION_IMPL(STR)                                                  \
    []() {                                                                               \
        struct wrapper                                                                   \
        {                                                                                \
            static constexpr const char* get() { return STR.data(); }                    \
            using sequence =                                                             \
                std::make_index_sequence<std::min(STR.find("_hidden"), STR.length())>;   \
        };                                                                               \
        return ::omnitrace::debug::get_chars<wrapper>();                                 \
    }();

#define OMNITRACE_CONDITIONAL_PRINT(COND, ...)                                           \
    if((COND) && ::omnitrace::config::get_debug_tid() &&                                 \
       ::omnitrace::config::get_debug_pid())                                             \
    {                                                                                    \
        fflush(stderr);                                                                  \
        tim::auto_lock_t _lk{ tim::type_mutex<decltype(std::cerr)>() };                  \
        fprintf(stderr, "[omnitrace][%i][%li] ", OMNITRACE_PROCESS_IDENTIFIER,           \
                tim::threading::get_id());                                               \
        fprintf(stderr, __VA_ARGS__);                                                    \
        fflush(stderr);                                                                  \
    }

#define OMNITRACE_CONDITIONAL_BASIC_PRINT(COND, ...)                                     \
    if((COND) && ::omnitrace::config::get_debug_tid() &&                                 \
       ::omnitrace::config::get_debug_pid())                                             \
    {                                                                                    \
        fflush(stderr);                                                                  \
        tim::auto_lock_t _lk{ tim::type_mutex<decltype(std::cerr)>() };                  \
        fprintf(stderr, "[omnitrace] ");                                                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        fflush(stderr);                                                                  \
    }

#define OMNITRACE_CONDITIONAL_PRINT_F(COND, ...)                                         \
    if((COND) && ::omnitrace::config::get_debug_tid() &&                                 \
       ::omnitrace::config::get_debug_pid())                                             \
    {                                                                                    \
        fflush(stderr);                                                                  \
        tim::auto_lock_t _lk{ tim::type_mutex<decltype(std::cerr)>() };                  \
        fprintf(stderr, "[omnitrace][%i][%li][%s] ", OMNITRACE_PROCESS_IDENTIFIER,       \
                tim::threading::get_id(), OMNITRACE_FUNCTION);                           \
        fprintf(stderr, __VA_ARGS__);                                                    \
        fflush(stderr);                                                                  \
    }

#define OMNITRACE_CONDITIONAL_BASIC_PRINT_F(COND, ...)                                   \
    if((COND) && ::omnitrace::config::get_debug_tid() &&                                 \
       ::omnitrace::config::get_debug_pid())                                             \
    {                                                                                    \
        fflush(stderr);                                                                  \
        tim::auto_lock_t _lk{ tim::type_mutex<decltype(std::cerr)>() };                  \
        fprintf(stderr, "[omnitrace][%s] ", OMNITRACE_FUNCTION);                         \
        fprintf(stderr, __VA_ARGS__);                                                    \
        fflush(stderr);                                                                  \
    }

#define OMNITRACE_CONDITIONAL_THROW(COND, ...)                                           \
    if(COND)                                                                             \
    {                                                                                    \
        char _msg_buffer[2048];                                                          \
        snprintf(_msg_buffer, 2048, "[omnitrace][%i][%li][%s] ",                         \
                 OMNITRACE_PROCESS_IDENTIFIER, tim::threading::get_id(),                 \
                 OMNITRACE_FUNCTION);                                                    \
        auto len = strlen(_msg_buffer);                                                  \
        snprintf(_msg_buffer + len, 2048 - len, __VA_ARGS__);                            \
        throw std::runtime_error(_msg_buffer);                                           \
    }

#define OMNITRACE_CONDITIONAL_BASIC_THROW(COND, ...)                                     \
    if(COND)                                                                             \
    {                                                                                    \
        char _msg_buffer[2048];                                                          \
        snprintf(_msg_buffer, 2048, "[omnitrace][%s] ", OMNITRACE_FUNCTION);             \
        auto len = strlen(_msg_buffer);                                                  \
        snprintf(_msg_buffer + len, 2048 - len, __VA_ARGS__);                            \
        throw std::runtime_error(_msg_buffer);                                           \
    }

#define OMNITRACE_CI_THROW(COND, ...)                                                    \
    OMNITRACE_CONDITIONAL_THROW(::omnitrace::get_is_continuous_integration() && (COND),  \
                                __VA_ARGS__)

#define OMNITRACE_CI_BASIC_THROW(COND, ...)                                              \
    OMNITRACE_CONDITIONAL_BASIC_THROW(                                                   \
        ::omnitrace::get_is_continuous_integration() && (COND), __VA_ARGS__)

#define OMNITRACE_STRINGIZE(...) #__VA_ARGS__
#define OMNITRACE_ESC(...)       __VA_ARGS__

//--------------------------------------------------------------------------------------//
//
//  Debug macros
//
//--------------------------------------------------------------------------------------//
#define OMNITRACE_DEBUG(...)                                                             \
    OMNITRACE_CONDITIONAL_PRINT(::omnitrace::get_debug(), __VA_ARGS__)

#define OMNITRACE_BASIC_DEBUG(...)                                                       \
    OMNITRACE_CONDITIONAL_BASIC_PRINT(::omnitrace::get_debug_env(), __VA_ARGS__)

#define OMNITRACE_DEBUG_F(...)                                                           \
    OMNITRACE_CONDITIONAL_PRINT_F(::omnitrace::get_debug(), __VA_ARGS__)

#define OMNITRACE_BASIC_DEBUG_F(...)                                                     \
    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(::omnitrace::get_debug_env(), __VA_ARGS__)

#define OMNITRACE_CT_DEBUG(...)                                                          \
    OMNITRACE_CONDITIONAL_PRINT(::omnitrace::get_critical_trace_debug(), __VA_ARGS__)

#define OMNITRACE_CT_DEBUG_F(...)                                                        \
    OMNITRACE_CONDITIONAL_PRINT_F(::omnitrace::get_critical_trace_debug(), __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Verbose macros
//
//--------------------------------------------------------------------------------------//
#define OMNITRACE_VERBOSE(LEVEL, ...)                                                    \
    OMNITRACE_CONDITIONAL_PRINT(                                                         \
        ::omnitrace::get_debug() || ::omnitrace::get_verbose() >= LEVEL, __VA_ARGS__)

#define OMNITRACE_BASIC_VERBOSE(LEVEL, ...)                                              \
    OMNITRACE_CONDITIONAL_BASIC_PRINT(::omnitrace::get_debug_env() ||                    \
                                          ::omnitrace::get_verbose_env() >= LEVEL,       \
                                      __VA_ARGS__)

#define OMNITRACE_VERBOSE_F(LEVEL, ...)                                                  \
    OMNITRACE_CONDITIONAL_PRINT_F(                                                       \
        ::omnitrace::get_debug() || ::omnitrace::get_verbose() >= LEVEL, __VA_ARGS__)

#define OMNITRACE_BASIC_VERBOSE_F(LEVEL, ...)                                            \
    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(::omnitrace::get_debug_env() ||                  \
                                            ::omnitrace::get_verbose_env() >= LEVEL,     \
                                        __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Basic print macros (basic means it will not provide PID/RANK or TID) and will not
//  initialize the settings.
//
//--------------------------------------------------------------------------------------//

#define OMNITRACE_BASIC_PRINT(...) OMNITRACE_CONDITIONAL_BASIC_PRINT(true, __VA_ARGS__)

#define OMNITRACE_BASIC_PRINT_F(...) OMNITRACE_CONDITIONAL_BASIC_PRINT(true, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Print macros. Will provide PID/RANK and TID (will initialize settings)
//
//--------------------------------------------------------------------------------------//

#define OMNITRACE_PRINT(...) OMNITRACE_CONDITIONAL_PRINT(true, __VA_ARGS__)

#define OMNITRACE_PRINT_F(...) OMNITRACE_CONDITIONAL_PRINT_F(true, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Throw macros
//
//--------------------------------------------------------------------------------------//

#define OMNITRACE_THROW(...) OMNITRACE_CONDITIONAL_THROW(true, __VA_ARGS__)

#define OMNITRACE_BASIC_THROW(...) OMNITRACE_CONDITIONAL_BASIC_THROW(true, __VA_ARGS__)

#include <string>

namespace std
{
inline std::string
to_string(bool _v)
{
    return (_v) ? "true" : "false";
}
}  // namespace std
