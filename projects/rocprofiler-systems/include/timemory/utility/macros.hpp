//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.
//

#pragma once

#include "timemory/macros/attributes.hpp"

#include <cstdint>
#include <cstdio>
#include <iosfwd>
#include <string>
#include <utility>

#if defined(TIMEMORY_CORE_SOURCE)
#    define TIMEMORY_UTILITY_SOURCE
#elif defined(TIMEMORY_USE_CORE_EXTERN)
#    define TIMEMORY_USE_UTILITY_EXTERN
#endif
//
#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_UTILITY_EXTERN)
#    define TIMEMORY_USE_UTILITY_EXTERN
#endif

//======================================================================================//
//
//      Quick way to create a globally accessible setting
//
//======================================================================================//

#if !defined(CREATE_STATIC_VARIABLE_ACCESSOR)
#    define CREATE_STATIC_VARIABLE_ACCESSOR(TYPE, FUNC_NAME, VARIABLE)                   \
        static TYPE& FUNC_NAME()                                                         \
        {                                                                                \
            static TYPE _instance = Type::VARIABLE;                                      \
            return _instance;                                                            \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(CREATE_STATIC_FUNCTION_ACCESSOR)
#    define CREATE_STATIC_FUNCTION_ACCESSOR(TYPE, FUNC_NAME, VARIABLE)                   \
        static TYPE& FUNC_NAME()                                                         \
        {                                                                                \
            static TYPE _instance = Type::VARIABLE();                                    \
            return _instance;                                                            \
        }
#endif

//======================================================================================//
//
//      DEBUG
//
//======================================================================================//

// stringify some macro -- uses TIMEMORY_STRINGIZE2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#if !defined(TIMEMORY_STRINGIZE)
#    define TIMEMORY_STRINGIZE(X) TIMEMORY_STRINGIZE2(X)
#endif

// actual stringifying
#if !defined(TIMEMORY_STRINGIZE2)
#    define TIMEMORY_STRINGIZE2(X) #    X
#endif

#if !defined(TIMEMORY_TRUNCATED_FILE_STRING)
#    define TIMEMORY_TRUNCATED_FILE_STRING(FILE)                                         \
        []() {                                                                           \
            std::string _f{ FILE };                                                      \
            auto        _pos = _f.find("/timemory/");                                    \
            if(_pos != std::string::npos)                                                \
            {                                                                            \
                return _f.substr(_pos + 1);                                              \
            }                                                                            \
            return _f;                                                                   \
        }()
#endif

#if !defined(TIMEMORY_FILE_LINE_FUNC_STRING)
#    define TIMEMORY_FILE_LINE_FUNC_STRING                                               \
        std::string                                                                      \
        {                                                                                \
            std::string{ "[" } + TIMEMORY_TRUNCATED_FILE_STRING(__FILE__) + ":" +        \
                std::to_string(__LINE__) + "@'" + __FUNCTION__ + "']"                    \
        }
#endif

#if !defined(TIMEMORY_PID_TID_STRING)
#    define TIMEMORY_PID_TID_STRING                                                      \
        std::string                                                                      \
        {                                                                                \
            std::string{ "[pid=" } + std::to_string(::tim::process::get_id()) +          \
                std::string{ "][tid=" } + std::to_string(::tim::threading::get_id()) +   \
                "]"                                                                      \
        }
#endif

template <typename Arg>
TIMEMORY_NOINLINE auto
timemory_proxy_value(Arg&& arg, int) -> decltype(arg.proxy_value())
{
    return arg.proxy_value();
}

template <typename Arg>
TIMEMORY_NOINLINE auto
timemory_proxy_value(Arg&& arg, long)
{
    return std::forward<Arg>(arg);
}

template <typename... Args>
TIMEMORY_NOINLINE void
timemory_print_here(const char* _pid_tid, const char* _file, int _line, const char* _func,
                    Args&&... args)
{
    fprintf(stderr, "%s[%s:%i@'%s']> ", _pid_tid, _file, _line, _func);
    fprintf(stderr, timemory_proxy_value(std::forward<Args>(args), 0)...);
    fprintf(stderr, "...\n");
    fflush(stderr);
}

#if !defined(PRINT_HERE)
#    define PRINT_HERE(...)                                                              \
        timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                             \
                            TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(), __LINE__,  \
                            __FUNCTION__, __VA_ARGS__)
#endif

#if !defined(DEBUG_PRINT_HERE)
#    if defined(DEBUG)
#        define DEBUG_PRINT_HERE(...)                                                    \
            if(::tim::settings::debug())                                                 \
            {                                                                            \
                timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                     \
                                    TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),    \
                                    __LINE__, __FUNCTION__, __VA_ARGS__);                \
            }
#    else
#        define DEBUG_PRINT_HERE(...)
#    endif
#endif

#if !defined(VERBOSE_PRINT_HERE)
#    define VERBOSE_PRINT_HERE(VERBOSE_LEVEL, ...)                                       \
        if(::tim::settings::verbose() >= VERBOSE_LEVEL)                                  \
        {                                                                                \
            timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                         \
                                TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),        \
                                __LINE__, __FUNCTION__, __VA_ARGS__);                    \
        }
#endif

#if !defined(CONDITIONAL_PRINT_HERE)
#    define CONDITIONAL_PRINT_HERE(CONDITION, ...)                                       \
        if(CONDITION)                                                                    \
        {                                                                                \
            timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                         \
                                TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),        \
                                __LINE__, __FUNCTION__, __VA_ARGS__);                    \
        }
#endif

#if !defined(TIMEMORY_CONDITIONAL_BACKTRACE)
#    define TIMEMORY_CONDITIONAL_BACKTRACE(CONDITION, DEPTH)                             \
        if(CONDITION)                                                                    \
        {                                                                                \
            ::tim::print_backtrace<DEPTH>(std::cerr, TIMEMORY_PID_TID_STRING,            \
                                          TIMEMORY_FILE_LINE_FUNC_STRING);               \
        }
#endif

#if !defined(TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE)
#    define TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(CONDITION, DEPTH)                   \
        if(CONDITION)                                                                    \
        {                                                                                \
            ::tim::print_demangled_backtrace<DEPTH>(std::cerr, TIMEMORY_PID_TID_STRING,  \
                                                    TIMEMORY_FILE_LINE_FUNC_STRING);     \
        }
#endif

#if !defined(PRETTY_PRINT_HERE)
#    if defined(TIMEMORY_GNU_COMPILER) || defined(TIMEMORY_CLANG_COMPILER)
#        define PRETTY_PRINT_HERE(...)                                                   \
            timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                         \
                                TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),        \
                                __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)
#    else
#        define PRETTY_PRINT_HERE(...)                                                   \
            timemory_print_here(TIMEMORY_PID_TID_STRING.c_str(),                         \
                                TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),        \
                                __LINE__, __FUNCTION__, __VA_ARGS__)
#    endif
#endif

#if !defined(TIMEMORY_CONDITIONAL_BACKTRACE)
#    define TIMEMORY_CONDITIONAL_BACKTRACE(CONDITION, DEPTH)                             \
        if(CONDITION)                                                                    \
        {                                                                                \
            ::tim::print_backtrace<DEPTH>(std::cerr, TIMEMORY_PID_TID_STRING);           \
        }
#endif

#if !defined(TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE)
#    define TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(CONDITION, DEPTH)                   \
        if(CONDITION)                                                                    \
        {                                                                                \
            ::tim::print_demangled_backtrace<DEPTH>(std::cerr, TIMEMORY_PID_TID_STRING); \
        }
#endif

#if defined(DEBUG)

template <typename... Args>
inline void
__LOG(std::string file, int line, const char* msg, Args&&... args)
{
    auto _pos = file.find("/timemory/");
    if(_pos == std::string::npos)
        _pos = file.find_last_of('/');
    if(_pos != std::string::npos)
        file = file.substr(_pos);
    fprintf(stderr, "[Log @ %s:%i]> ", file.c_str(), line);
    fprintf(stderr, msg, std::forward<Args>(args)...);
    fprintf(stderr, "\n");
}

//--------------------------------------------------------------------------------------//

inline void
__LOG(std::string file, int line, const char* msg)
{
    if(file.find('/') != std::string::npos)
        file = file.substr(file.find_last_of('/'));
    fprintf(stderr, "[Log @ %s:%i]> %s\n", file.c_str(), line, msg);
}

//--------------------------------------------------------------------------------------//
// auto insert the file and line
#    define _LOG(...) __LOG(__FILE__, __LINE__, __VA_ARGS__)

//--------------------------------------------------------------------------------------//

template <typename... Args>
inline void
_DBG(const char* msg, Args&&... args)
{
    fprintf(stderr, msg, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

inline void
_DBG(const char* msg)
{
    fprintf(stderr, "%s", msg);
}

#else
#    define _LOG(...)                                                                    \
        {}
#    define _DBG(...)                                                                    \
        {}
#endif

//======================================================================================//
//
// Define macros for utility
//
//======================================================================================//
//
#if defined(TIMEMORY_UTILITY_SOURCE)
#    define TIMEMORY_UTILITY_LINKAGE(...) __VA_ARGS__
#    define TIMEMORY_UTILITY_INLINE
#elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_UTILITY_EXTERN)
#    define TIMEMORY_UTILITY_LINKAGE(...) __VA_ARGS__
#    define TIMEMORY_UTILITY_INLINE
#else
#    define TIMEMORY_UTILITY_LINKAGE(...) inline __VA_ARGS__
#    define TIMEMORY_UTILITY_INLINE inline
#    define TIMEMORY_UTILITY_HEADER_MODE
#endif
