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

#if !defined(OMNITRACE_USER_SOURCE)
#    define OMNITRACE_USER_SOURCE 1
#endif

#include "omnitrace/user.h"
#include "omnitrace/categories.h"
#include "omnitrace/types.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <optional>

using annotation_t = omnitrace_annotation_t;

namespace
{
using trace_func_t            = omnitrace_trace_func_t;
using region_func_t           = omnitrace_region_func_t;
using annotated_region_func_t = omnitrace_annotated_region_func_t;
using user_callbacks_t        = omnitrace_user_callbacks_t;

user_callbacks_t _callbacks = OMNITRACE_USER_CALLBACKS_INIT;

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
    int omnitrace_user_start_trace(void) { return invoke(_callbacks.start_trace); }

    int omnitrace_user_stop_trace(void) { return invoke(_callbacks.stop_trace); }

    int omnitrace_user_start_thread_trace(void)
    {
        return invoke(_callbacks.start_thread_trace);
    }

    int omnitrace_user_stop_thread_trace(void)
    {
        return invoke(_callbacks.stop_thread_trace);
    }

    int omnitrace_user_push_region(const char* id)
    {
        return invoke(_callbacks.push_region, id);
    }

    int omnitrace_user_pop_region(const char* id)
    {
        return invoke(_callbacks.pop_region, id);
    }

    int omnitrace_user_progress(const char* id)
    {
        return invoke(_callbacks.progress, id);
    }

    int omnitrace_user_push_annotated_region(const char* id, annotation_t* _annotations,
                                             size_t _annotation_count)
    {
        return invoke(_callbacks.push_annotated_region, id, _annotations,
                      _annotation_count);
    }

    int omnitrace_user_pop_annotated_region(const char* id, annotation_t* _annotations,
                                            size_t _annotation_count)
    {
        return invoke(_callbacks.pop_annotated_region, id, _annotations,
                      _annotation_count);
    }

    int omnitrace_user_annotated_progress(const char* id, annotation_t* _annotations,
                                          size_t _annotation_count)
    {
        return invoke(_callbacks.annotated_progress, id, _annotations, _annotation_count);
    }

    int omnitrace_user_configure(omnitrace_user_configure_mode_t mode,
                                 omnitrace_user_callbacks_t      inp,
                                 omnitrace_user_callbacks_t*     out)
    {
        auto _former = _callbacks;

        switch(mode)
        {
            case OMNITRACE_USER_REPLACE_CONFIG:
            {
                _callbacks = inp;
                break;
            }
            case OMNITRACE_USER_UNION_CONFIG:
            {
                auto _update = [](auto& _lhs, auto _rhs) {
                    if(_rhs) _lhs = _rhs;
                };

                user_callbacks_t _v = _callbacks;

                _update(_v.start_trace, inp.start_trace);
                _update(_v.stop_trace, inp.stop_trace);
                _update(_v.start_thread_trace, inp.start_thread_trace);
                _update(_v.stop_thread_trace, inp.stop_thread_trace);
                _update(_v.push_region, inp.push_region);
                _update(_v.pop_region, inp.pop_region);
                _update(_v.progress, inp.progress);
                _update(_v.push_annotated_region, inp.push_annotated_region);
                _update(_v.pop_annotated_region, inp.pop_annotated_region);
                _update(_v.annotated_progress, inp.annotated_progress);

                _callbacks = _v;
                break;
            }
            case OMNITRACE_USER_INTERSECT_CONFIG:
            {
                auto _update = [](auto& _lhs, auto _rhs) {
                    if(_lhs != _rhs) _lhs = nullptr;
                };

                user_callbacks_t _v = _callbacks;

                _update(_v.start_trace, inp.start_trace);
                _update(_v.stop_trace, inp.stop_trace);
                _update(_v.start_thread_trace, inp.start_thread_trace);
                _update(_v.stop_thread_trace, inp.stop_thread_trace);
                _update(_v.push_region, inp.push_region);
                _update(_v.pop_region, inp.pop_region);
                _update(_v.progress, inp.progress);
                _update(_v.push_annotated_region, inp.push_annotated_region);
                _update(_v.pop_annotated_region, inp.pop_annotated_region);
                _update(_v.annotated_progress, inp.annotated_progress);

                _callbacks = _v;
                break;
            }
            default:
            {
                if(out) *out = _former;
                return OMNITRACE_USER_ERROR_INVALID_CATEGORY;
            }
        }

        if(out) *out = _former;

        return OMNITRACE_USER_SUCCESS;
    }

    const char* omnitrace_user_error_string(int error_category)
    {
        switch(error_category)
        {
            case OMNITRACE_USER_SUCCESS: return "Success";
            case OMNITRACE_USER_ERROR_NO_BINDING: return "Function pointer not assigned";
            case OMNITRACE_USER_ERROR_BAD_VALUE: return "Invalid value was provided";
            case OMNITRACE_USER_ERROR_INVALID_CATEGORY:
                return "Invalid user binding category";
            case OMNITRACE_USER_ERROR_INTERNAL:
                return "An unknown error occurred within omnitrace library";
            default: break;
        }
        return "No error";
    }
}
