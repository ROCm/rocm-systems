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

#include "common/defines.h"
#include "omnitrace/user.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <gnu/libc-version.h>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(OMNITRACE_USE_OMPT)
#    define OMNITRACE_USE_OMPT 0
#endif

//--------------------------------------------------------------------------------------//
//
//      omnitrace symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    void omnitrace_init_library(void) OMNITRACE_PUBLIC_API;
    void omnitrace_init(const char*, bool, const char*) OMNITRACE_PUBLIC_API;
    void omnitrace_finalize(void) OMNITRACE_PUBLIC_API;
    void omnitrace_set_env(const char* env_name,
                           const char* env_val) OMNITRACE_PUBLIC_API;
    void omnitrace_set_mpi(bool use, bool attached) OMNITRACE_PUBLIC_API;
    void omnitrace_push_trace(const char* name) OMNITRACE_PUBLIC_API;
    void omnitrace_pop_trace(const char* name) OMNITRACE_PUBLIC_API;
    void omnitrace_push_region(const char*) OMNITRACE_PUBLIC_API;
    void omnitrace_pop_region(const char*) OMNITRACE_PUBLIC_API;

#if defined(OMNITRACE_DL_SOURCE) && (OMNITRACE_DL_SOURCE > 0)
    int omnitrace_user_start_trace_dl(void) OMNITRACE_HIDDEN_API;
    int omnitrace_user_stop_trace_dl(void) OMNITRACE_HIDDEN_API;

    int omnitrace_user_start_thread_trace_dl(void) OMNITRACE_HIDDEN_API;
    int omnitrace_user_stop_thread_trace_dl(void) OMNITRACE_HIDDEN_API;

    int omnitrace_user_push_region_dl(const char*) OMNITRACE_HIDDEN_API;
    int omnitrace_user_pop_region_dl(const char*) OMNITRACE_HIDDEN_API;

    struct ompt_start_tool_result_t;

    ompt_start_tool_result_t* ompt_start_tool(unsigned int,
                                              const char*) OMNITRACE_PUBLIC_API;
#endif
}
