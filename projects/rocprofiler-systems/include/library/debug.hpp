// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include <cstdio>
#include <timemory/api.hpp>
#include <timemory/backends/dmp.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/utility/utility.hpp>

bool
get_debug();

bool
get_critical_trace_debug();

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

#define OMNITRACE_DEBUG(...) OMNITRACE_CONDITIONAL_PRINT(get_debug(), __VA_ARGS__)
#define OMNITRACE_PRINT(...) OMNITRACE_CONDITIONAL_PRINT(true, __VA_ARGS__)
#define OMNITRACE_CT_DEBUG(...)                                                          \
    OMNITRACE_CONDITIONAL_PRINT(get_critical_trace_debug(), __VA_ARGS__)
