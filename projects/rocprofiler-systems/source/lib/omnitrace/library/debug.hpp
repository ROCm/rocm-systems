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
#include <timemory/backends/threading.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/locking.hpp>
#include <timemory/utility/utility.hpp>

#include <array>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <utility>

namespace omnitrace
{
inline namespace config
{
bool
get_debug() OMNITRACE_HOT;

int
get_verbose() OMNITRACE_HOT;

bool
get_debug_env() OMNITRACE_HOT;

int
get_verbose_env() OMNITRACE_HOT;

bool
get_is_continuous_integration() OMNITRACE_HOT;

bool
get_debug_tid() OMNITRACE_HOT;

bool
get_debug_pid() OMNITRACE_HOT;

bool
get_critical_trace_debug() OMNITRACE_HOT;
}  // namespace config

namespace debug
{
inline void
flush()
{
    fflush(stdout);
    std::cout << std::flush;
    fflush(stderr);
    std::cerr << std::flush;
}
//
struct lock
{
    lock();
    ~lock();

private:
    tim::auto_lock_t m_lk;
};
//
template <typename Arg, typename... Args>
bool
is_bracket(Arg&& _arg, Args&&...)
{
    if constexpr(::tim::concepts::is_string_type<Arg>::value)
        return (::std::string_view{ _arg }.empty()) ? false : _arg[0] == '[';
    else
        return false;
}
//
namespace
{
template <typename T, size_t... Idx>
auto
get_chars(T&& _c, std::index_sequence<Idx...>)
{
    return std::array<const char, sizeof...(Idx) + 1>{ std::forward<T>(_c)[Idx]...,
                                                       '\0' };
}
}  // namespace
}  // namespace debug
}  // namespace omnitrace

#if !defined(OMNITRACE_DEBUG_BUFFER_LEN)
#    define OMNITRACE_DEBUG_BUFFER_LEN 2048
#endif

#if !defined(OMNITRACE_PROCESS_IDENTIFIER)
#    if defined(TIMEMORY_USE_MPI)
#        define OMNITRACE_PROCESS_IDENTIFIER static_cast<int>(::tim::dmp::rank())
#    elif defined(TIMEMORY_USE_MPI_HEADERS)
#        define OMNITRACE_PROCESS_IDENTIFIER                                             \
            (::tim::dmp::is_initialized()) ? static_cast<int>(::tim::dmp::rank())        \
                                           : static_cast<int>(::tim::process::get_id())
#    else
#        define OMNITRACE_PROCESS_IDENTIFIER static_cast<int>(::tim::process::get_id())
#    endif
#endif

#if !defined(OMNITRACE_THREAD_IDENTIFIER)
#    define OMNITRACE_THREAD_IDENTIFIER ::tim::threading::get_id()
#endif

#if defined(__clang__) || (__GNUC__ < 9)
#    define OMNITRACE_FUNCTION                                                           \
        std::string{ __FUNCTION__ }                                                      \
            .substr(0, std::string_view{ __FUNCTION__ }.find("_hidden"))                 \
            .c_str()
#    define OMNITRACE_PRETTY_FUNCTION                                                    \
        std::string{ __PRETTY_FUNCTION__ }                                               \
            .substr(0, std::string_view{ __PRETTY_FUNCTION__ }.find("_hidden"))          \
            .c_str()
#else
#    define OMNITRACE_FUNCTION                                                           \
        ::omnitrace::debug::get_chars(                                                   \
            std::string_view{ __FUNCTION__ },                                            \
            std::make_index_sequence<std::min(                                           \
                std::string_view{ __FUNCTION__ }.find("_hidden"),                        \
                std::string_view{ __FUNCTION__ }.length())>{})                           \
            .data()
#    define OMNITRACE_PRETTY_FUNCTION                                                    \
        ::omnitrace::debug::get_chars(                                                   \
            std::string_view{ __PRETTY_FUNCTION__ },                                     \
            std::make_index_sequence<std::min(                                           \
                std::string_view{ __PRETTY_FUNCTION__ }.find("_hidden"),                 \
                std::string_view{ __PRETTY_FUNCTION__ }.length())>{})                    \
            .data()
#endif

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_PRINT(COND, ...)                                           \
    if((COND) && ::omnitrace::config::get_debug_tid() &&                                 \
       ::omnitrace::config::get_debug_pid())                                             \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _lk{};                                                  \
        fprintf(stderr, "[omnitrace][%i][%li]%s", OMNITRACE_PROCESS_IDENTIFIER,          \
                OMNITRACE_THREAD_IDENTIFIER,                                             \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_BASIC_PRINT(COND, ...)                                     \
    if((COND) && ::omnitrace::config::get_debug_tid() &&                                 \
       ::omnitrace::config::get_debug_pid())                                             \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _lk{};                                                  \
        fprintf(stderr, "[omnitrace]%s",                                                 \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_PRINT_F(COND, ...)                                         \
    if((COND) && ::omnitrace::config::get_debug_tid() &&                                 \
       ::omnitrace::config::get_debug_pid())                                             \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _lk{};                                                  \
        fprintf(stderr, "[omnitrace][%i][%li][%s]%s", OMNITRACE_PROCESS_IDENTIFIER,      \
                OMNITRACE_THREAD_IDENTIFIER, OMNITRACE_FUNCTION,                         \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_BASIC_PRINT_F(COND, ...)                                   \
    if((COND) && ::omnitrace::config::get_debug_tid() &&                                 \
       ::omnitrace::config::get_debug_pid())                                             \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _lk{};                                                  \
        fprintf(stderr, "[omnitrace][%s]%s", OMNITRACE_FUNCTION,                         \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ::omnitrace::debug::flush();                                                     \
    }

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_THROW(COND, ...)                                           \
    if(COND)                                                                             \
    {                                                                                    \
        char _msg_buffer[OMNITRACE_DEBUG_BUFFER_LEN];                                    \
        snprintf(_msg_buffer, OMNITRACE_DEBUG_BUFFER_LEN, "[omnitrace][%i][%li][%s]%s",  \
                 OMNITRACE_PROCESS_IDENTIFIER, OMNITRACE_THREAD_IDENTIFIER,              \
                 OMNITRACE_FUNCTION,                                                     \
                 ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        auto len = strlen(_msg_buffer);                                                  \
        snprintf(_msg_buffer + len, OMNITRACE_DEBUG_BUFFER_LEN - len, __VA_ARGS__);      \
        throw std::runtime_error(_msg_buffer);                                           \
    }

#define OMNITRACE_CONDITIONAL_BASIC_THROW(COND, ...)                                     \
    if(COND)                                                                             \
    {                                                                                    \
        char _msg_buffer[OMNITRACE_DEBUG_BUFFER_LEN];                                    \
        snprintf(_msg_buffer, OMNITRACE_DEBUG_BUFFER_LEN, "[omnitrace][%s]%s",           \
                 OMNITRACE_FUNCTION,                                                     \
                 ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        auto len = strlen(_msg_buffer);                                                  \
        snprintf(_msg_buffer + len, OMNITRACE_DEBUG_BUFFER_LEN - len, __VA_ARGS__);      \
        throw std::runtime_error(_msg_buffer);                                           \
    }

#define OMNITRACE_CI_THROW(COND, ...)                                                    \
    OMNITRACE_CONDITIONAL_THROW(::omnitrace::get_is_continuous_integration() && (COND),  \
                                __VA_ARGS__)

#define OMNITRACE_CI_BASIC_THROW(COND, ...)                                              \
    OMNITRACE_CONDITIONAL_BASIC_THROW(                                                   \
        ::omnitrace::get_is_continuous_integration() && (COND), __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_FAIL(COND, ...)                                            \
    if(COND)                                                                             \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        fprintf(stderr, "[omnitrace][%i][%li]%s", OMNITRACE_PROCESS_IDENTIFIER,          \
                OMNITRACE_THREAD_IDENTIFIER,                                             \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ::omnitrace::set_state(::omnitrace::State::Finalized);                           \
        ::tim::disable_signal_detection();                                               \
        ::tim::print_demangled_backtrace<64>();                                          \
        ::std::exit(EXIT_FAILURE);                                                       \
    }

#define OMNITRACE_CONDITIONAL_BASIC_FAIL(COND, ...)                                      \
    if(COND)                                                                             \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        fprintf(stderr, "[omnitrace]%s",                                                 \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ::omnitrace::set_state(::omnitrace::State::Finalized);                           \
        ::tim::disable_signal_detection();                                               \
        ::tim::print_demangled_backtrace<64>();                                          \
        ::std::exit(EXIT_FAILURE);                                                       \
    }

#define OMNITRACE_CONDITIONAL_FAIL_F(COND, ...)                                          \
    if(COND)                                                                             \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        fprintf(stderr, "[omnitrace][%i][%li][%s]%s", OMNITRACE_PROCESS_IDENTIFIER,      \
                OMNITRACE_THREAD_IDENTIFIER, OMNITRACE_FUNCTION,                         \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ::omnitrace::set_state(::omnitrace::State::Finalized);                           \
        ::tim::disable_signal_detection();                                               \
        ::tim::print_demangled_backtrace<64>();                                          \
        ::std::exit(EXIT_FAILURE);                                                       \
    }

#define OMNITRACE_CONDITIONAL_BASIC_FAIL_F(COND, ...)                                    \
    if(COND)                                                                             \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        fprintf(stderr, "[omnitrace][%s]%s", OMNITRACE_FUNCTION,                         \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ::omnitrace::set_state(::omnitrace::State::Finalized);                           \
        ::tim::disable_signal_detection();                                               \
        ::tim::print_demangled_backtrace<64>();                                          \
        ::std::exit(EXIT_FAILURE);                                                       \
    }

#define OMNITRACE_CI_FAIL(COND, ...)                                                     \
    OMNITRACE_CONDITIONAL_FAIL(::omnitrace::get_is_continuous_integration() && (COND),   \
                               __VA_ARGS__)

#define OMNITRACE_CI_BASIC_FAIL(COND, ...)                                               \
    OMNITRACE_CONDITIONAL_BASIC_FAIL(                                                    \
        ::omnitrace::get_is_continuous_integration() && (COND), __VA_ARGS__)

//--------------------------------------------------------------------------------------//

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
        ::omnitrace::get_debug() || (::omnitrace::get_verbose() >= LEVEL), __VA_ARGS__)

#define OMNITRACE_BASIC_VERBOSE(LEVEL, ...)                                              \
    OMNITRACE_CONDITIONAL_BASIC_PRINT(::omnitrace::get_debug_env() ||                    \
                                          (::omnitrace::get_verbose_env() >= LEVEL),     \
                                      __VA_ARGS__)

#define OMNITRACE_VERBOSE_F(LEVEL, ...)                                                  \
    OMNITRACE_CONDITIONAL_PRINT_F(                                                       \
        ::omnitrace::get_debug() || (::omnitrace::get_verbose() >= LEVEL), __VA_ARGS__)

#define OMNITRACE_BASIC_VERBOSE_F(LEVEL, ...)                                            \
    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(::omnitrace::get_debug_env() ||                  \
                                            (::omnitrace::get_verbose_env() >= LEVEL),   \
                                        __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Basic print macros (basic means it will not provide PID/RANK or TID) and will not
//  initialize the settings.
//
//--------------------------------------------------------------------------------------//

#define OMNITRACE_BASIC_PRINT(...) OMNITRACE_CONDITIONAL_BASIC_PRINT(true, __VA_ARGS__)

#define OMNITRACE_BASIC_PRINT_F(...)                                                     \
    OMNITRACE_CONDITIONAL_BASIC_PRINT_F(true, __VA_ARGS__)

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

//--------------------------------------------------------------------------------------//
//
//  Fail macros
//
//--------------------------------------------------------------------------------------//

#define OMNITRACE_FAIL(...) OMNITRACE_CONDITIONAL_FAIL(true, __VA_ARGS__)

#define OMNITRACE_FAIL_F(...) OMNITRACE_CONDITIONAL_FAIL_F(true, __VA_ARGS__)

#define OMNITRACE_BASIC_FAIL(...) OMNITRACE_CONDITIONAL_BASIC_FAIL(true, __VA_ARGS__)

#define OMNITRACE_BASIC_FAIL_F(...) OMNITRACE_CONDITIONAL_BASIC_FAIL_F(true, __VA_ARGS__)

#include <string>

namespace std
{
inline std::string
to_string(bool _v)
{
    return (_v) ? "true" : "false";
}
}  // namespace std
