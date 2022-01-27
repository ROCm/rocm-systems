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

namespace omnitrace
{
bool
get_debug();

bool
get_critical_trace_debug();
}  // namespace omnitrace

#if defined(TIMEMORY_USE_MPI)
#    define OMNITRACE_CONDITIONAL_PRINT(COND, ...)                                       \
        if(COND)                                                                         \
        {                                                                                \
            fflush(stderr);                                                              \
            tim::auto_lock_t _lk{ tim::type_mutex<decltype(std::cerr)>() };              \
            fprintf(stderr, "[omnitrace][%i][%li] ", static_cast<int>(tim::dmp::rank()), \
                    tim::threading::get_id());                                           \
            fprintf(stderr, __VA_ARGS__);                                                \
            fflush(stderr);                                                              \
        }
#else
#    define OMNITRACE_CONDITIONAL_PRINT(COND, ...)                                       \
        if(COND)                                                                         \
        {                                                                                \
            fflush(stderr);                                                              \
            tim::auto_lock_t _lk{ tim::type_mutex<decltype(std::cerr)>() };              \
            fprintf(stderr, "[omnitrace][%i][%li] ",                                     \
                    static_cast<int>(tim::process::get_id()), tim::threading::get_id()); \
            fprintf(stderr, __VA_ARGS__);                                                \
            fflush(stderr);                                                              \
        }
#endif

#define OMNITRACE_CONDITIONAL_BASIC_PRINT(COND, ...)                                     \
    if(COND)                                                                             \
    {                                                                                    \
        fflush(stderr);                                                                  \
        tim::auto_lock_t _lk{ tim::type_mutex<decltype(std::cerr)>() };                  \
        fprintf(stderr, "[omnitrace] ");                                                 \
        fprintf(stderr, __VA_ARGS__);                                                    \
        fflush(stderr);                                                                  \
    }

#define OMNITRACE_DEBUG(...)                                                             \
    OMNITRACE_CONDITIONAL_PRINT(::omnitrace::get_debug(), __VA_ARGS__)
#define OMNITRACE_PRINT(...) OMNITRACE_CONDITIONAL_PRINT(true, __VA_ARGS__)
#define OMNITRACE_CT_DEBUG(...)                                                          \
    OMNITRACE_CONDITIONAL_PRINT(::omnitrace::get_critical_trace_debug(), __VA_ARGS__)
