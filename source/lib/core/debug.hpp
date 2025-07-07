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

namespace rocprofsys
{
inline namespace config
{
bool
get_debug() ROCPROFSYS_HOT;

int
get_verbose() ROCPROFSYS_HOT;

bool
get_debug_env() ROCPROFSYS_HOT;

int
get_verbose_env() ROCPROFSYS_HOT;

bool
get_is_continuous_integration() ROCPROFSYS_HOT;

bool
get_debug_tid() ROCPROFSYS_HOT;

bool
get_debug_pid() ROCPROFSYS_HOT;
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
    fprintf(::rocprofsys::debug::get_file(), "%s", ::tim::log::color::end());
    fflush(::rocprofsys::debug::get_file());
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
}  // namespace rocprofsys

#if !defined(ROCPROFSYS_DEBUG_BUFFER_LEN)
#    define ROCPROFSYS_DEBUG_BUFFER_LEN 1024
#endif

#if !defined(ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER)
#    if defined(ROCPROFSYS_USE_MPI)
#        define ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER static_cast<int>(::tim::dmp::rank())
#    elif defined(ROCPROFSYS_USE_MPI_HEADERS)
#        define ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER                                      \
            (::tim::dmp::is_initialized()) ? static_cast<int>(::tim::dmp::rank())        \
                                           : static_cast<int>(::tim::process::get_id())
#    else
#        define ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER                                      \
            static_cast<int>(::tim::process::get_id())
#    endif
#endif

#if !defined(ROCPROFSYS_DEBUG_THREAD_IDENTIFIER)
#    define ROCPROFSYS_DEBUG_THREAD_IDENTIFIER ::rocprofsys::debug::get_tid()
#endif

#if !defined(ROCPROFSYS_SOURCE_LOCATION)
#    define ROCPROFSYS_SOURCE_LOCATION                                                   \
        ::rocprofsys::debug::source_location { __PRETTY_FUNCTION__, __FILE__, __LINE__ }
#endif

#if !defined(ROCPROFSYS_RECORD_SOURCE_LOCATION)
#    define ROCPROFSYS_RECORD_SOURCE_LOCATION                                            \
        ::rocprofsys::debug::set_source_location(ROCPROFSYS_SOURCE_LOCATION)
#endif

#if defined(__clang__) || (__GNUC__ < 9)
#    define ROCPROFSYS_FUNCTION                                                          \
        std::string{ __FUNCTION__ }                                                      \
            .substr(0, std::string_view{ __FUNCTION__ }.find("_hidden"))                 \
            .c_str()
#    define ROCPROFSYS_PRETTY_FUNCTION                                                   \
        std::string{ __PRETTY_FUNCTION__ }                                               \
            .substr(0, std::string_view{ __PRETTY_FUNCTION__ }.find("_hidden"))          \
            .c_str()
#else
#    define ROCPROFSYS_FUNCTION                                                          \
        ::rocprofsys::debug::get_chars(                                                  \
            std::string_view{ __FUNCTION__ },                                            \
            std::make_index_sequence<std::min(                                           \
                std::string_view{ __FUNCTION__ }.find("_hidden"),                        \
                std::string_view{ __FUNCTION__ }.length())>{})                           \
            .data()
#    define ROCPROFSYS_PRETTY_FUNCTION                                                   \
        ::rocprofsys::debug::get_chars(                                                  \
            std::string_view{ __PRETTY_FUNCTION__ },                                     \
            std::make_index_sequence<std::min(                                           \
                std::string_view{ __PRETTY_FUNCTION__ }.find("_hidden"),                 \
                std::string_view{ __PRETTY_FUNCTION__ }.length())>{})                    \
            .data()
#endif

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_FPRINTF_STDERR_COLOR(COLOR)                                           \
    fprintf(::rocprofsys::debug::get_file(), "%s", ::tim::log::color::COLOR())

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_PRINT_COLOR(COLOR, COND, ...)                             \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(COLOR);                                          \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%li]%s",             \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_DEBUG_THREAD_IDENTIFIER, \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

#define ROCPROFSYS_CONDITIONAL_PRINT_COLOR_F(COLOR, COND, ...)                           \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(COLOR);                                          \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%li][%s]%s",         \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_DEBUG_THREAD_IDENTIFIER, \
                ROCPROFSYS_FUNCTION,                                                     \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

#define ROCPROFSYS_PRINT_COLOR(COLOR, ...)                                               \
    ROCPROFSYS_CONDITIONAL_PRINT_COLOR(COLOR, true, __VA_ARGS__)

#define ROCPROFSYS_PRINT_COLOR_F(COLOR, ...)                                             \
    ROCPROFSYS_CONDITIONAL_PRINT_COLOR_F(COLOR, true, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_PRINT(COND, ...)                                          \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(info);                                           \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%li]%s",             \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_DEBUG_THREAD_IDENTIFIER, \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

#define ROCPROFSYS_CONDITIONAL_BASIC_PRINT(COND, ...)                                    \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(info);                                           \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i]%s",                  \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER,                                     \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

#define ROCPROFSYS_CONDITIONAL_PRINT_F(COND, ...)                                        \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(info);                                           \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%li][%s]%s",         \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_DEBUG_THREAD_IDENTIFIER, \
                ROCPROFSYS_FUNCTION,                                                     \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

#define ROCPROFSYS_CONDITIONAL_BASIC_PRINT_F(COND, ...)                                  \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(info);                                           \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%s]%s",              \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_FUNCTION,                \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_WARN(COND, ...)                                           \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(warning);                                        \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%li]%s",             \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_DEBUG_THREAD_IDENTIFIER, \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

#define ROCPROFSYS_CONDITIONAL_BASIC_WARN(COND, ...)                                     \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(warning);                                        \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i]%s",                  \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER,                                     \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

#define ROCPROFSYS_CONDITIONAL_WARN_F(COND, ...)                                         \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(warning);                                        \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%li][%s]%s",         \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_DEBUG_THREAD_IDENTIFIER, \
                ROCPROFSYS_FUNCTION,                                                     \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

#define ROCPROFSYS_CONDITIONAL_BASIC_WARN_F(COND, ...)                                   \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&            \
                           ::rocprofsys::config::get_debug_pid()))                       \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::debug::lock _debug_lk{};                                           \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(warning);                                        \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%s]%s",              \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_FUNCTION,                \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
    }

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_THROW_E(COND, TYPE, ...)                                  \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                      \
    {                                                                                    \
        char _msg_buffer[ROCPROFSYS_DEBUG_BUFFER_LEN];                                   \
        bool _print_backtrace = ::rocprofsys::get_debug() ||                             \
                                ::rocprofsys::get_verbose() >= 2 ||                      \
                                ::rocprofsys::get_is_continuous_integration();           \
        snprintf(_msg_buffer, ROCPROFSYS_DEBUG_BUFFER_LEN,                               \
                 "[rocprof-sys][%i][%li][%s]%s", ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER,    \
                 ROCPROFSYS_DEBUG_THREAD_IDENTIFIER, ROCPROFSYS_FUNCTION,                \
                 ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");               \
        auto len = strlen(_msg_buffer);                                                  \
        snprintf(_msg_buffer + len, ROCPROFSYS_DEBUG_BUFFER_LEN - len, __VA_ARGS__);     \
        if(!_print_backtrace)                                                            \
            throw ::rocprofsys::exception<TYPE>(                                         \
                ::tim::log::string(::tim::log::color::fatal(), _msg_buffer), false);     \
        else                                                                             \
            throw ::rocprofsys::exception<TYPE>(                                         \
                ::tim::log::string(::tim::log::color::fatal(), _msg_buffer));            \
    }

#define ROCPROFSYS_CONDITIONAL_BASIC_THROW_E(COND, TYPE, ...)                            \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                      \
    {                                                                                    \
        char _msg_buffer[ROCPROFSYS_DEBUG_BUFFER_LEN];                                   \
        snprintf(_msg_buffer, ROCPROFSYS_DEBUG_BUFFER_LEN, "[rocprof-sys][%i][%s]%s",    \
                 ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_FUNCTION,               \
                 ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");               \
        auto len = strlen(_msg_buffer);                                                  \
        snprintf(_msg_buffer + len, ROCPROFSYS_DEBUG_BUFFER_LEN - len, __VA_ARGS__);     \
        throw ::rocprofsys::exception<TYPE>(                                             \
            ::tim::log::string(::tim::log::color::fatal(), _msg_buffer));                \
    }

#define ROCPROFSYS_CI_THROW_E(COND, TYPE, ...)                                           \
    ROCPROFSYS_CONDITIONAL_THROW_E(                                                      \
        ::rocprofsys::get_is_continuous_integration() && (COND), TYPE, __VA_ARGS__)

#define ROCPROFSYS_CI_BASIC_THROW_E(COND, TYPE, ...)                                     \
    ROCPROFSYS_CONDITIONAL_BASIC_THROW_E(                                                \
        ::rocprofsys::get_is_continuous_integration() && (COND), TYPE, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_THROW(COND, ...)                                          \
    ROCPROFSYS_CONDITIONAL_THROW_E((COND), std::runtime_error, __VA_ARGS__)

#define ROCPROFSYS_CONDITIONAL_BASIC_THROW(COND, ...)                                    \
    ROCPROFSYS_CONDITIONAL_BASIC_THROW_E((COND), std::runtime_error, __VA_ARGS__)

#define ROCPROFSYS_CI_THROW(COND, ...)                                                   \
    ROCPROFSYS_CI_THROW_E((COND), std::runtime_error, __VA_ARGS__)

#define ROCPROFSYS_CI_BASIC_THROW(COND, ...)                                             \
    ROCPROFSYS_CI_BASIC_THROW_E((COND), std::runtime_error, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_FAILURE(COND, METHOD, ...)                                \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                      \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(fatal);                                          \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%li]%s",             \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_DEBUG_THREAD_IDENTIFIER, \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);                         \
        timemory_print_demangled_backtrace<64>();                                        \
        METHOD;                                                                          \
    }

#define ROCPROFSYS_CONDITIONAL_BASIC_FAILURE(COND, METHOD, ...)                          \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                      \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(fatal);                                          \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i]%s",                  \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER,                                     \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);                         \
        timemory_print_demangled_backtrace<64>();                                        \
        METHOD;                                                                          \
    }

#define ROCPROFSYS_CONDITIONAL_FAILURE_F(COND, METHOD, ...)                              \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                      \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(fatal);                                          \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%li][%s]%s",         \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_DEBUG_THREAD_IDENTIFIER, \
                ROCPROFSYS_FUNCTION,                                                     \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);                         \
        timemory_print_demangled_backtrace<64>();                                        \
        METHOD;                                                                          \
    }

#define ROCPROFSYS_CONDITIONAL_BASIC_FAILURE_F(COND, METHOD, ...)                        \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                      \
    {                                                                                    \
        ::rocprofsys::debug::flush();                                                    \
        ROCPROFSYS_FPRINTF_STDERR_COLOR(fatal);                                          \
        fprintf(::rocprofsys::debug::get_file(), "[rocprof-sys][%i][%s]%s",              \
                ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER, ROCPROFSYS_FUNCTION,                \
                ::rocprofsys::debug::is_bracket(__VA_ARGS__) ? "" : " ");                \
        fprintf(::rocprofsys::debug::get_file(), __VA_ARGS__);                           \
        ::rocprofsys::debug::flush();                                                    \
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);                         \
        timemory_print_demangled_backtrace<64>();                                        \
        METHOD;                                                                          \
    }

#define ROCPROFSYS_CI_FAILURE(COND, METHOD, ...)                                         \
    ROCPROFSYS_CONDITIONAL_FAILURE(                                                      \
        ::rocprofsys::get_is_continuous_integration() && (COND), METHOD, __VA_ARGS__)

#define ROCPROFSYS_CI_BASIC_FAILURE(COND, METHOD, ...)                                   \
    ROCPROFSYS_CONDITIONAL_BASIC_FAILURE(                                                \
        ::rocprofsys::get_is_continuous_integration() && (COND), METHOD, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_FAIL(COND, ...)                                           \
    ROCPROFSYS_CONDITIONAL_FAILURE(COND, ROCPROFSYS_ESC(::std::exit(EXIT_FAILURE)),      \
                                   __VA_ARGS__)

#define ROCPROFSYS_CONDITIONAL_BASIC_FAIL(COND, ...)                                     \
    ROCPROFSYS_CONDITIONAL_BASIC_FAILURE(                                                \
        COND, ROCPROFSYS_ESC(::std::exit(EXIT_FAILURE)), __VA_ARGS__)

#define ROCPROFSYS_CONDITIONAL_FAIL_F(COND, ...)                                         \
    ROCPROFSYS_CONDITIONAL_FAILURE_F(COND, ROCPROFSYS_ESC(::std::exit(EXIT_FAILURE)),    \
                                     __VA_ARGS__)

#define ROCPROFSYS_CONDITIONAL_BASIC_FAIL_F(COND, ...)                                   \
    ROCPROFSYS_CONDITIONAL_BASIC_FAILURE_F(                                              \
        COND, ROCPROFSYS_ESC(::std::exit(EXIT_FAILURE)), __VA_ARGS__)

#define ROCPROFSYS_CI_FAIL(COND, ...)                                                    \
    ROCPROFSYS_CI_FAILURE(COND, ROCPROFSYS_ESC(::std::exit(EXIT_FAILURE)), __VA_ARGS__)

#define ROCPROFSYS_CI_BASIC_FAIL(COND, ...)                                              \
    ROCPROFSYS_CI_BASIC_FAILURE(COND, ROCPROFSYS_ESC(::std::exit(EXIT_FAILURE)),         \
                                __VA_ARGS__)

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_ABORT(COND, ...)                                          \
    ROCPROFSYS_CONDITIONAL_FAILURE(COND, ROCPROFSYS_ESC(::std::abort()), __VA_ARGS__)

#define ROCPROFSYS_CONDITIONAL_BASIC_ABORT(COND, ...)                                    \
    ROCPROFSYS_CONDITIONAL_BASIC_FAILURE(COND, ROCPROFSYS_ESC(::std::abort()),           \
                                         __VA_ARGS__)

#define ROCPROFSYS_CONDITIONAL_ABORT_F(COND, ...)                                        \
    ROCPROFSYS_CONDITIONAL_FAILURE_F(COND, ROCPROFSYS_ESC(::std::abort()), __VA_ARGS__)

#define ROCPROFSYS_CONDITIONAL_BASIC_ABORT_F(COND, ...)                                  \
    ROCPROFSYS_CONDITIONAL_BASIC_FAILURE_F(COND, ROCPROFSYS_ESC(::std::abort()),         \
                                           __VA_ARGS__)

#define ROCPROFSYS_CI_ABORT(COND, ...)                                                   \
    ROCPROFSYS_CI_FAILURE(COND, ROCPROFSYS_ESC(::std::abort()), __VA_ARGS__)

#define ROCPROFSYS_CI_BASIC_ABORT(COND, ...)                                             \
    ROCPROFSYS_CI_BASIC_FAILURE(COND, ROCPROFSYS_ESC(::std::abort()), __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Debug macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_DEBUG(...)                                                            \
    ROCPROFSYS_CONDITIONAL_PRINT(::rocprofsys::get_debug(), __VA_ARGS__)

#define ROCPROFSYS_BASIC_DEBUG(...)                                                      \
    ROCPROFSYS_CONDITIONAL_BASIC_PRINT(::rocprofsys::get_debug_env(), __VA_ARGS__)

#define ROCPROFSYS_DEBUG_F(...)                                                          \
    ROCPROFSYS_CONDITIONAL_PRINT_F(::rocprofsys::get_debug(), __VA_ARGS__)

#define ROCPROFSYS_BASIC_DEBUG_F(...)                                                    \
    ROCPROFSYS_CONDITIONAL_BASIC_PRINT_F(::rocprofsys::get_debug_env(), __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Verbose macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_VERBOSE(LEVEL, ...)                                                   \
    ROCPROFSYS_CONDITIONAL_PRINT(::rocprofsys::get_debug() ||                            \
                                     (::rocprofsys::get_verbose() >= LEVEL),             \
                                 __VA_ARGS__)

#define ROCPROFSYS_BASIC_VERBOSE(LEVEL, ...)                                             \
    ROCPROFSYS_CONDITIONAL_BASIC_PRINT(::rocprofsys::get_debug_env() ||                  \
                                           (::rocprofsys::get_verbose_env() >= LEVEL),   \
                                       __VA_ARGS__)

#define ROCPROFSYS_VERBOSE_F(LEVEL, ...)                                                 \
    ROCPROFSYS_CONDITIONAL_PRINT_F(::rocprofsys::get_debug() ||                          \
                                       (::rocprofsys::get_verbose() >= LEVEL),           \
                                   __VA_ARGS__)

#define ROCPROFSYS_BASIC_VERBOSE_F(LEVEL, ...)                                           \
    ROCPROFSYS_CONDITIONAL_BASIC_PRINT_F(::rocprofsys::get_debug_env() ||                \
                                             (::rocprofsys::get_verbose_env() >= LEVEL), \
                                         __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Warning macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_WARNING(LEVEL, ...)                                                   \
    ROCPROFSYS_CONDITIONAL_WARN(::rocprofsys::get_debug() ||                             \
                                    (::rocprofsys::get_verbose() >= LEVEL),              \
                                __VA_ARGS__)

#define ROCPROFSYS_BASIC_WARNING(LEVEL, ...)                                             \
    ROCPROFSYS_CONDITIONAL_BASIC_WARN(::rocprofsys::get_debug_env() ||                   \
                                          (::rocprofsys::get_verbose_env() >= LEVEL),    \
                                      __VA_ARGS__)

#define ROCPROFSYS_WARNING_F(LEVEL, ...)                                                 \
    ROCPROFSYS_CONDITIONAL_WARN_F(::rocprofsys::get_debug() ||                           \
                                      (::rocprofsys::get_verbose() >= LEVEL),            \
                                  __VA_ARGS__)

#define ROCPROFSYS_BASIC_WARNING_F(LEVEL, ...)                                           \
    ROCPROFSYS_CONDITIONAL_BASIC_WARN_F(::rocprofsys::get_debug_env() ||                 \
                                            (::rocprofsys::get_verbose_env() >= LEVEL),  \
                                        __VA_ARGS__)

#define ROCPROFSYS_WARNING_IF(COND, ...) ROCPROFSYS_CONDITIONAL_WARN((COND), __VA_ARGS__)

#define ROCPROFSYS_WARNING_IF_F(COND, ...)                                               \
    ROCPROFSYS_CONDITIONAL_WARN_F((COND), __VA_ARGS__)

#define ROCPROFSYS_WARNING_OR_CI_THROW(LEVEL, ...)                                       \
    {                                                                                    \
        if(ROCPROFSYS_UNLIKELY(::rocprofsys::get_is_continuous_integration()))           \
        {                                                                                \
            ROCPROFSYS_CI_THROW(true, __VA_ARGS__);                                      \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            ROCPROFSYS_CONDITIONAL_WARN(::rocprofsys::get_debug() ||                     \
                                            (::rocprofsys::get_verbose() >= LEVEL),      \
                                        __VA_ARGS__)                                     \
        }                                                                                \
    }

#define ROCPROFSYS_REQUIRE(...) TIMEMORY_REQUIRE(__VA_ARGS__)
#define ROCPROFSYS_PREFER(COND)                                                          \
    ((ROCPROFSYS_LIKELY(COND))                                                           \
         ? ::tim::log::base()                                                            \
         : ((::rocprofsys::get_is_continuous_integration()) ? TIMEMORY_FATAL             \
                                                            : TIMEMORY_WARNING))

//--------------------------------------------------------------------------------------//
//
//  Basic print macros (basic means it will not provide PID/RANK or TID) and will not
//  initialize the settings.
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_BASIC_PRINT(...) ROCPROFSYS_CONDITIONAL_BASIC_PRINT(true, __VA_ARGS__)

#define ROCPROFSYS_BASIC_PRINT_F(...)                                                    \
    ROCPROFSYS_CONDITIONAL_BASIC_PRINT_F(true, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Print macros. Will provide PID/RANK and TID (will initialize settings)
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_PRINT(...) ROCPROFSYS_CONDITIONAL_PRINT(true, __VA_ARGS__)

#define ROCPROFSYS_PRINT_F(...) ROCPROFSYS_CONDITIONAL_PRINT_F(true, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Throw macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_THROW(...) ROCPROFSYS_CONDITIONAL_THROW(true, __VA_ARGS__)

#define ROCPROFSYS_BASIC_THROW(...) ROCPROFSYS_CONDITIONAL_BASIC_THROW(true, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Fail macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_FAIL(...) ROCPROFSYS_CONDITIONAL_FAIL(true, __VA_ARGS__)

#define ROCPROFSYS_FAIL_F(...) ROCPROFSYS_CONDITIONAL_FAIL_F(true, __VA_ARGS__)

#define ROCPROFSYS_BASIC_FAIL(...) ROCPROFSYS_CONDITIONAL_BASIC_FAIL(true, __VA_ARGS__)

#define ROCPROFSYS_BASIC_FAIL_F(...)                                                     \
    ROCPROFSYS_CONDITIONAL_BASIC_FAIL_F(true, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Abort macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_ABORT(...) ROCPROFSYS_CONDITIONAL_ABORT(true, __VA_ARGS__)

#define ROCPROFSYS_ABORT_F(...) ROCPROFSYS_CONDITIONAL_ABORT_F(true, __VA_ARGS__)

#define ROCPROFSYS_BASIC_ABORT(...) ROCPROFSYS_CONDITIONAL_BASIC_ABORT(true, __VA_ARGS__)

#define ROCPROFSYS_BASIC_ABORT_F(...)                                                    \
    ROCPROFSYS_CONDITIONAL_BASIC_ABORT_F(true, __VA_ARGS__)

#include <string>

namespace std
{
inline std::string
to_string(bool _v)
{
    return (_v) ? "true" : "false";
}
}  // namespace std
