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

#define OMNITRACE_SOURCE 1

#include "omnitrace/user.h"

#include <cstdio>

namespace
{
using void_func_t   = void (*)(void);
using region_func_t = void (*)(const char*);

void_func_t   _start_trace        = nullptr;
void_func_t   _stop_trace         = nullptr;
void_func_t   _start_thread_trace = nullptr;
void_func_t   _stop_thread_trace  = nullptr;
region_func_t _push_region        = nullptr;
region_func_t _pop_region         = nullptr;
}  // namespace

extern "C"
{
    void omnitrace_user_start_trace(void)
    {
        if(_start_trace) (*_start_trace)();
    }

    void omnitrace_user_stop_trace(void)
    {
        if(_stop_trace) (*_stop_trace)();
    }

    void omnitrace_user_start_thread_trace(void)
    {
        if(_start_thread_trace) (*_start_thread_trace)();
    }

    void omnitrace_user_stop_thread_trace(void)
    {
        if(_stop_thread_trace) (*_stop_thread_trace)();
    }

    void omnitrace_user_push_region(const char* name)
    {
        if(_push_region) (*_push_region)(name);
    }

    void omnitrace_user_pop_region(const char* name)
    {
        if(_pop_region) (*_pop_region)(name);
    }

    void omnitrace_user_configure(void (*_a)(), void (*_b)(), void (*_c)(), void (*_d)(),
                                  void (*_e)(const char*), void (*_f)(const char*))
    {
        _start_trace        = _a;
        _stop_trace         = _b;
        _start_thread_trace = _c;
        _stop_thread_trace  = _d;
        _push_region        = _e;
        _pop_region         = _f;
    }
}
