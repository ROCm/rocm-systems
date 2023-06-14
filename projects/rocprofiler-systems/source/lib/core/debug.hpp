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

#include "defines.hpp"
#include "exception.hpp"
#include "locking.hpp"

#include <timemory/api.hpp>
#include <timemory/backends/dmp.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/log/logger.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/signals/signal_handlers.hpp>
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
struct source_location
{
    std::string_view function = {};
    std::string_view file     = {};
    int              line     = 0;
};
//
void
set_source_location(source_location&&);
//
FILE*
get_file();
//
void
close_file();
//
int64_t
get_tid();
//
inline void
flush()
{
    fprintf(stdout, "%s", ::tim::log::color::end());
    fflush(stdout);
    std::cout << ::tim::log::color::end() << std::flush;
    fprintf(::omnitrace::debug::get_file(), "%s", ::tim::log::color::end());
    fflush(::omnitrace::debug::get_file());
    std::cerr << ::tim::log::color::end() << std::flush;
}
//
struct lock
{
    lock();
    ~lock();

private:
    locking::atomic_lock m_lk;
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

namespace binary
{
struct address_range;
}

using address_range_t = binary::address_range;

template <typename Tp>
std::string
as_hex(Tp, size_t _wdith = 16);

template <>
std::string as_hex<address_range_t>(address_range_t, size_t);

extern template std::string as_hex<int32_t>(int32_t, size_t);
extern template std::string as_hex<uint32_t>(uint32_t, size_t);
extern template std::string as_hex<int64_t>(int64_t, size_t);
extern template std::string as_hex<uint64_t>(uint64_t, size_t);
extern template std::string
as_hex<void*>(void*, size_t);
}  // namespace omnitrace

#if !defined(OMNITRACE_DEBUG_BUFFER_LEN)
#    define OMNITRACE_DEBUG_BUFFER_LEN 1024
#endif

#if !defined(OMNITRACE_DEBUG_PROCESS_IDENTIFIER)
#    if defined(TIMEMORY_USE_MPI)
#        define OMNITRACE_DEBUG_PROCESS_IDENTIFIER static_cast<int>(::tim::dmp::rank())
#    elif defined(TIMEMORY_USE_MPI_HEADERS)
#        define OMNITRACE_DEBUG_PROCESS_IDENTIFIER                                       \
            (::tim::dmp::is_initialized()) ? static_cast<int>(::tim::dmp::rank())        \
                                           : static_cast<int>(::tim::process::get_id())
#    else
#        define OMNITRACE_DEBUG_PROCESS_IDENTIFIER                                       \
            static_cast<int>(::tim::process::get_id())
#    endif
#endif

#if !defined(OMNITRACE_DEBUG_THREAD_IDENTIFIER)
#    define OMNITRACE_DEBUG_THREAD_IDENTIFIER ::omnitrace::debug::get_tid()
#endif

#if !defined(OMNITRACE_SOURCE_LOCATION)
#    define OMNITRACE_SOURCE_LOCATION                                                    \
        ::omnitrace::debug::source_location { __PRETTY_FUNCTION__, __FILE__, __LINE__ }
#endif

#if !defined(OMNITRACE_RECORD_SOURCE_LOCATION)
#    define OMNITRACE_RECORD_SOURCE_LOCATION                                             \
        ::omnitrace::debug::set_source_location(OMNITRACE_SOURCE_LOCATION)
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

#define OMNITRACE_FPRINTF_STDERR_COLOR(COLOR)                                            \
    fprintf(::omnitrace::debug::get_file(), "%s", ::tim::log::color::COLOR())

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_PRINT_COLOR(COLOR, COND, ...)                              \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(COLOR);                                           \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%li]%s",                \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_DEBUG_THREAD_IDENTIFIER,   \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_PRINT_COLOR_F(COLOR, COND, ...)                            \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(COLOR);                                           \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%li][%s]%s",            \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_DEBUG_THREAD_IDENTIFIER,   \
                OMNITRACE_FUNCTION,                                                      \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_PRINT_COLOR(COLOR, ...)                                                \
    OMNITRACE_CONDITIONAL_PRINT_COLOR(COLOR, true, __VA_ARGS__)

#define OMNITRACE_PRINT_COLOR_F(COLOR, ...)                                              \
    OMNITRACE_CONDITIONAL_PRINT_COLOR_F(COLOR, true, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_PRINT(COND, ...)                                           \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(info);                                            \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%li]%s",                \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_DEBUG_THREAD_IDENTIFIER,   \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_BASIC_PRINT(COND, ...)                                     \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(info);                                            \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i]%s",                     \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER,                                      \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_PRINT_F(COND, ...)                                         \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(info);                                            \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%li][%s]%s",            \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_DEBUG_THREAD_IDENTIFIER,   \
                OMNITRACE_FUNCTION,                                                      \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_BASIC_PRINT_F(COND, ...)                                   \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(info);                                            \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%s]%s",                 \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_FUNCTION,                  \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_WARN(COND, ...)                                            \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(warning);                                         \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%li]%s",                \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_DEBUG_THREAD_IDENTIFIER,   \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_BASIC_WARN(COND, ...)                                      \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(warning);                                         \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i]%s",                     \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER,                                      \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_WARN_F(COND, ...)                                          \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(warning);                                         \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%li][%s]%s",            \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_DEBUG_THREAD_IDENTIFIER,   \
                OMNITRACE_FUNCTION,                                                      \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

#define OMNITRACE_CONDITIONAL_BASIC_WARN_F(COND, ...)                                    \
    if(OMNITRACE_UNLIKELY((COND) && ::omnitrace::config::get_debug_tid() &&              \
                          ::omnitrace::config::get_debug_pid()))                         \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::debug::lock _debug_lk{};                                            \
        OMNITRACE_FPRINTF_STDERR_COLOR(warning);                                         \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%s]%s",                 \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_FUNCTION,                  \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
    }

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_THROW_E(COND, TYPE, ...)                                   \
    if(OMNITRACE_UNLIKELY((COND)))                                                       \
    {                                                                                    \
        char _msg_buffer[OMNITRACE_DEBUG_BUFFER_LEN];                                    \
        snprintf(_msg_buffer, OMNITRACE_DEBUG_BUFFER_LEN, "[omnitrace][%i][%li][%s]%s",  \
                 OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_DEBUG_THREAD_IDENTIFIER,  \
                 OMNITRACE_FUNCTION,                                                     \
                 ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        auto len = strlen(_msg_buffer);                                                  \
        snprintf(_msg_buffer + len, OMNITRACE_DEBUG_BUFFER_LEN - len, __VA_ARGS__);      \
        throw ::omnitrace::exception<TYPE>(                                              \
            ::tim::log::string(::tim::log::color::fatal(), _msg_buffer));                \
    }

#define OMNITRACE_CONDITIONAL_BASIC_THROW_E(COND, TYPE, ...)                             \
    if(OMNITRACE_UNLIKELY((COND)))                                                       \
    {                                                                                    \
        char _msg_buffer[OMNITRACE_DEBUG_BUFFER_LEN];                                    \
        snprintf(_msg_buffer, OMNITRACE_DEBUG_BUFFER_LEN, "[omnitrace][%i][%s]%s",       \
                 OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_FUNCTION,                 \
                 ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        auto len = strlen(_msg_buffer);                                                  \
        snprintf(_msg_buffer + len, OMNITRACE_DEBUG_BUFFER_LEN - len, __VA_ARGS__);      \
        throw ::omnitrace::exception<TYPE>(                                              \
            ::tim::log::string(::tim::log::color::fatal(), _msg_buffer));                \
    }

#define OMNITRACE_CI_THROW_E(COND, TYPE, ...)                                            \
    OMNITRACE_CONDITIONAL_THROW_E(                                                       \
        ::omnitrace::get_is_continuous_integration() && (COND), TYPE, __VA_ARGS__)

#define OMNITRACE_CI_BASIC_THROW_E(COND, TYPE, ...)                                      \
    OMNITRACE_CONDITIONAL_BASIC_THROW_E(                                                 \
        ::omnitrace::get_is_continuous_integration() && (COND), TYPE, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_THROW(COND, ...)                                           \
    OMNITRACE_CONDITIONAL_THROW_E((COND), std::runtime_error, __VA_ARGS__)

#define OMNITRACE_CONDITIONAL_BASIC_THROW(COND, ...)                                     \
    OMNITRACE_CONDITIONAL_BASIC_THROW_E((COND), std::runtime_error, __VA_ARGS__)

#define OMNITRACE_CI_THROW(COND, ...)                                                    \
    OMNITRACE_CI_THROW_E((COND), std::runtime_error, __VA_ARGS__)

#define OMNITRACE_CI_BASIC_THROW(COND, ...)                                              \
    OMNITRACE_CI_BASIC_THROW_E((COND), std::runtime_error, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_FAILURE(COND, METHOD, ...)                                 \
    if(OMNITRACE_UNLIKELY((COND)))                                                       \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        OMNITRACE_FPRINTF_STDERR_COLOR(fatal);                                           \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%li]%s",                \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_DEBUG_THREAD_IDENTIFIER,   \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::set_state(::omnitrace::State::Finalized);                           \
        timemory_print_demangled_backtrace<64>();                                        \
        METHOD;                                                                          \
    }

#define OMNITRACE_CONDITIONAL_BASIC_FAILURE(COND, METHOD, ...)                           \
    if(OMNITRACE_UNLIKELY((COND)))                                                       \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        OMNITRACE_FPRINTF_STDERR_COLOR(fatal);                                           \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i]%s",                     \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER,                                      \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::set_state(::omnitrace::State::Finalized);                           \
        timemory_print_demangled_backtrace<64>();                                        \
        METHOD;                                                                          \
    }

#define OMNITRACE_CONDITIONAL_FAILURE_F(COND, METHOD, ...)                               \
    if(OMNITRACE_UNLIKELY((COND)))                                                       \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        OMNITRACE_FPRINTF_STDERR_COLOR(fatal);                                           \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%li][%s]%s",            \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_DEBUG_THREAD_IDENTIFIER,   \
                OMNITRACE_FUNCTION,                                                      \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::set_state(::omnitrace::State::Finalized);                           \
        timemory_print_demangled_backtrace<64>();                                        \
        METHOD;                                                                          \
    }

#define OMNITRACE_CONDITIONAL_BASIC_FAILURE_F(COND, METHOD, ...)                         \
    if(OMNITRACE_UNLIKELY((COND)))                                                       \
    {                                                                                    \
        ::omnitrace::debug::flush();                                                     \
        OMNITRACE_FPRINTF_STDERR_COLOR(fatal);                                           \
        fprintf(::omnitrace::debug::get_file(), "[omnitrace][%i][%s]%s",                 \
                OMNITRACE_DEBUG_PROCESS_IDENTIFIER, OMNITRACE_FUNCTION,                  \
                ::omnitrace::debug::is_bracket(__VA_ARGS__) ? "" : " ");                 \
        fprintf(::omnitrace::debug::get_file(), __VA_ARGS__);                            \
        ::omnitrace::debug::flush();                                                     \
        ::omnitrace::set_state(::omnitrace::State::Finalized);                           \
        timemory_print_demangled_backtrace<64>();                                        \
        METHOD;                                                                          \
    }

#define OMNITRACE_CI_FAILURE(COND, METHOD, ...)                                          \
    OMNITRACE_CONDITIONAL_FAILURE(                                                       \
        ::omnitrace::get_is_continuous_integration() && (COND), METHOD, __VA_ARGS__)

#define OMNITRACE_CI_BASIC_FAILURE(COND, METHOD, ...)                                    \
    OMNITRACE_CONDITIONAL_BASIC_FAILURE(                                                 \
        ::omnitrace::get_is_continuous_integration() && (COND), METHOD, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_FAIL(COND, ...)                                            \
    OMNITRACE_CONDITIONAL_FAILURE(COND, OMNITRACE_ESC(::std::exit(EXIT_FAILURE)),        \
                                  __VA_ARGS__)

#define OMNITRACE_CONDITIONAL_BASIC_FAIL(COND, ...)                                      \
    OMNITRACE_CONDITIONAL_BASIC_FAILURE(COND, OMNITRACE_ESC(::std::exit(EXIT_FAILURE)),  \
                                        __VA_ARGS__)

#define OMNITRACE_CONDITIONAL_FAIL_F(COND, ...)                                          \
    OMNITRACE_CONDITIONAL_FAILURE_F(COND, OMNITRACE_ESC(::std::exit(EXIT_FAILURE)),      \
                                    __VA_ARGS__)

#define OMNITRACE_CONDITIONAL_BASIC_FAIL_F(COND, ...)                                    \
    OMNITRACE_CONDITIONAL_BASIC_FAILURE_F(                                               \
        COND, OMNITRACE_ESC(::std::exit(EXIT_FAILURE)), __VA_ARGS__)

#define OMNITRACE_CI_FAIL(COND, ...)                                                     \
    OMNITRACE_CI_FAILURE(COND, OMNITRACE_ESC(::std::exit(EXIT_FAILURE)), __VA_ARGS__)

#define OMNITRACE_CI_BASIC_FAIL(COND, ...)                                               \
    OMNITRACE_CI_BASIC_FAILURE(COND, OMNITRACE_ESC(::std::exit(EXIT_FAILURE)),           \
                               __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define OMNITRACE_CONDITIONAL_ABORT(COND, ...)                                           \
    OMNITRACE_CONDITIONAL_FAILURE(COND, OMNITRACE_ESC(::std::abort()), __VA_ARGS__)

#define OMNITRACE_CONDITIONAL_BASIC_ABORT(COND, ...)                                     \
    OMNITRACE_CONDITIONAL_BASIC_FAILURE(COND, OMNITRACE_ESC(::std::abort()), __VA_ARGS__)

#define OMNITRACE_CONDITIONAL_ABORT_F(COND, ...)                                         \
    OMNITRACE_CONDITIONAL_FAILURE_F(COND, OMNITRACE_ESC(::std::abort()), __VA_ARGS__)

#define OMNITRACE_CONDITIONAL_BASIC_ABORT_F(COND, ...)                                   \
    OMNITRACE_CONDITIONAL_BASIC_FAILURE_F(COND, OMNITRACE_ESC(::std::abort()),           \
                                          __VA_ARGS__)

#define OMNITRACE_CI_ABORT(COND, ...)                                                    \
    OMNITRACE_CI_FAILURE(COND, OMNITRACE_ESC(::std::abort()), __VA_ARGS__)

#define OMNITRACE_CI_BASIC_ABORT(COND, ...)                                              \
    OMNITRACE_CI_BASIC_FAILURE(COND, OMNITRACE_ESC(::std::abort()), __VA_ARGS__)

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
//  Warning macros
//
//--------------------------------------------------------------------------------------//

#define OMNITRACE_WARNING(LEVEL, ...)                                                    \
    OMNITRACE_CONDITIONAL_WARN(                                                          \
        ::omnitrace::get_debug() || (::omnitrace::get_verbose() >= LEVEL), __VA_ARGS__)

#define OMNITRACE_BASIC_WARNING(LEVEL, ...)                                              \
    OMNITRACE_CONDITIONAL_BASIC_WARN(::omnitrace::get_debug_env() ||                     \
                                         (::omnitrace::get_verbose_env() >= LEVEL),      \
                                     __VA_ARGS__)

#define OMNITRACE_WARNING_F(LEVEL, ...)                                                  \
    OMNITRACE_CONDITIONAL_WARN_F(                                                        \
        ::omnitrace::get_debug() || (::omnitrace::get_verbose() >= LEVEL), __VA_ARGS__)

#define OMNITRACE_BASIC_WARNING_F(LEVEL, ...)                                            \
    OMNITRACE_CONDITIONAL_BASIC_WARN_F(::omnitrace::get_debug_env() ||                   \
                                           (::omnitrace::get_verbose_env() >= LEVEL),    \
                                       __VA_ARGS__)

#define OMNITRACE_WARNING_IF(COND, ...) OMNITRACE_CONDITIONAL_WARN((COND), __VA_ARGS__)

#define OMNITRACE_WARNING_IF_F(COND, ...)                                                \
    OMNITRACE_CONDITIONAL_WARN_F((COND), __VA_ARGS__)

#define OMNITRACE_WARNING_OR_CI_THROW(LEVEL, ...)                                        \
    {                                                                                    \
        if(OMNITRACE_UNLIKELY(::omnitrace::get_is_continuous_integration()))             \
        {                                                                                \
            OMNITRACE_CI_THROW(true, __VA_ARGS__);                                       \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            OMNITRACE_CONDITIONAL_WARN(::omnitrace::get_debug() ||                       \
                                           (::omnitrace::get_verbose() >= LEVEL),        \
                                       __VA_ARGS__)                                      \
        }                                                                                \
    }

#define OMNITRACE_REQUIRE(...) TIMEMORY_REQUIRE(__VA_ARGS__)
#define OMNITRACE_PREFER(COND)                                                           \
    (OMNITRACE_LIKELY(COND))                         ? ::tim::log::base()                \
    : (::omnitrace::get_is_continuous_integration()) ? TIMEMORY_FATAL                    \
                                                     : TIMEMORY_WARNING

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

//--------------------------------------------------------------------------------------//
//
//  Abort macros
//
//--------------------------------------------------------------------------------------//

#define OMNITRACE_ABORT(...) OMNITRACE_CONDITIONAL_ABORT(true, __VA_ARGS__)

#define OMNITRACE_ABORT_F(...) OMNITRACE_CONDITIONAL_ABORT_F(true, __VA_ARGS__)

#define OMNITRACE_BASIC_ABORT(...) OMNITRACE_CONDITIONAL_BASIC_ABORT(true, __VA_ARGS__)

#define OMNITRACE_BASIC_ABORT_F(...)                                                     \
    OMNITRACE_CONDITIONAL_BASIC_ABORT_F(true, __VA_ARGS__)

#include <string>

namespace std
{
inline std::string
to_string(bool _v)
{
    return (_v) ? "true" : "false";
}
}  // namespace std
