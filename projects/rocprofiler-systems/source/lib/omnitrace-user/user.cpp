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
#include <cstdlib>

namespace
{
using trace_func_t  = int (*)(void);
using region_func_t = int (*)(const char*);

trace_func_t  _start_trace        = nullptr;
trace_func_t  _stop_trace         = nullptr;
trace_func_t  _start_thread_trace = nullptr;
trace_func_t  _stop_thread_trace  = nullptr;
region_func_t _push_region        = nullptr;
region_func_t _pop_region         = nullptr;

const char*
as_string(OMNITRACE_USER_BINDINGS _category)
{
    switch(_category)
    {
        case OMNITRACE_USER_START_STOP: return "OMNITRACE_USER_START_STOP";
        case OMNITRACE_USER_START_STOP_THREAD: return "OMNITRACE_USER_START_STOP_THREAD";
        case OMNITRACE_USER_REGION: return "OMNITRACE_USER_REGION";
        default:
        {
            fprintf(stderr, "[omnitrace][user] Unknown user binding category: %i\n",
                    static_cast<int>(_category));
            fflush(stderr);
            break;
        }
    }
    return "OMNITRACE_USER_BINDINGS_unknown";
}

template <typename... Args>
inline auto
invoke(int (*_func)(Args...), Args... args)
{
    if(!_func) return OMNITRACE_USER_ERROR_NO_BINDING;
    if((*_func)(args...) != 0) return OMNITRACE_USER_ERROR_INTERNAL;
    return OMNITRACE_USER_SUCCESS;
}
}  // namespace

extern "C"
{
    int omnitrace_user_start_trace(void) { return invoke(_start_trace); }
    int omnitrace_user_stop_trace(void) { return invoke(_stop_trace); }
    int omnitrace_user_start_thread_trace(void) { return invoke(_start_thread_trace); }
    int omnitrace_user_stop_thread_trace(void) { return invoke(_stop_thread_trace); }
    int omnitrace_user_push_region(const char* id) { return invoke(_push_region, id); }
    int omnitrace_user_pop_region(const char* id) { return invoke(_pop_region, id); }

    int omnitrace_user_configure(int category, void* begin_func, void* end_func)
    {
        switch(category)
        {
            case OMNITRACE_USER_START_STOP:
            {
                if(begin_func) _start_trace = reinterpret_cast<trace_func_t>(begin_func);
                if(end_func) _stop_trace = reinterpret_cast<trace_func_t>(end_func);
                break;
            }
            case OMNITRACE_USER_START_STOP_THREAD:
            {
                if(begin_func)
                    _start_thread_trace = reinterpret_cast<trace_func_t>(begin_func);
                if(end_func)
                    _stop_thread_trace = reinterpret_cast<trace_func_t>(end_func);
                break;
            }
            case OMNITRACE_USER_REGION:
            {
                if(begin_func) _push_region = reinterpret_cast<region_func_t>(begin_func);
                if(end_func) _pop_region = reinterpret_cast<region_func_t>(end_func);
                break;
            }
            default:
            {
                return OMNITRACE_USER_ERROR_INVALID_CATEGORY;
            }
        }

        return OMNITRACE_USER_SUCCESS;
    }

    int omnitrace_user_get_callbacks(int category, void** begin_func, void** end_func)
    {
        switch(category)
        {
            case OMNITRACE_USER_START_STOP:
            {
                if(begin_func) *begin_func = reinterpret_cast<void*>(_start_trace);
                if(end_func) *end_func = reinterpret_cast<void*>(_stop_trace);
                break;
            }
            case OMNITRACE_USER_START_STOP_THREAD:
            {
                if(begin_func) *begin_func = reinterpret_cast<void*>(_start_thread_trace);
                if(end_func) *end_func = reinterpret_cast<void*>(_stop_thread_trace);
                break;
            }
            case OMNITRACE_USER_REGION:
            {
                if(begin_func) *begin_func = reinterpret_cast<void*>(_push_region);
                if(end_func) *end_func = reinterpret_cast<void*>(_pop_region);
                break;
            }
            default:
            {
                return OMNITRACE_USER_ERROR_INVALID_CATEGORY;
            }
        }

        return OMNITRACE_USER_SUCCESS;
    }

    const char* omnitrace_user_error_string(int error_category)
    {
        switch(error_category)
        {
            case OMNITRACE_USER_SUCCESS: return "Success";
            case OMNITRACE_USER_ERROR_NO_BINDING: return "Function pointer not assigned";
            case OMNITRACE_USER_ERROR_INVALID_CATEGORY:
                return "Invalid user binding category";
            case OMNITRACE_USER_ERROR_INTERNAL:
                return "An unknown error occurred within omnitrace library";
            default: break;
        }
        return "No error";
    }
}
