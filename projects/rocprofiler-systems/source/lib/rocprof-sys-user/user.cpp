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

#if !defined(ROCPROFSYS_USER_SOURCE)
#    define ROCPROFSYS_USER_SOURCE 1
#endif

#include "rocprofiler-systems/user.h"
#include "rocprofiler-systems/categories.h"
#include "rocprofiler-systems/types.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <optional>

using annotation_t = rocprofsys_annotation_t;

namespace
{
using trace_func_t            = rocprofsys_trace_func_t;
using region_func_t           = rocprofsys_region_func_t;
using annotated_region_func_t = rocprofsys_annotated_region_func_t;
using user_callbacks_t        = rocprofsys_user_callbacks_t;

user_callbacks_t _callbacks = ROCPROFSYS_USER_CALLBACKS_INIT;

template <typename... Args>
inline auto
invoke(int (*_func)(Args...), Args... args)
{
    if(!_func) return ROCPROFSYS_USER_ERROR_NO_BINDING;
    if((*_func)(args...) != 0) return ROCPROFSYS_USER_ERROR_INTERNAL;
    return ROCPROFSYS_USER_SUCCESS;
}
}  // namespace

extern "C"
{
    int rocprofsys_user_start_trace(void) { return invoke(_callbacks.start_trace); }

    int rocprofsys_user_stop_trace(void) { return invoke(_callbacks.stop_trace); }

    int rocprofsys_user_start_thread_trace(void)
    {
        return invoke(_callbacks.start_thread_trace);
    }

    int rocprofsys_user_stop_thread_trace(void)
    {
        return invoke(_callbacks.stop_thread_trace);
    }

    int rocprofsys_user_push_region(const char* id)
    {
        return invoke(_callbacks.push_region, id);
    }

    int rocprofsys_user_pop_region(const char* id)
    {
        return invoke(_callbacks.pop_region, id);
    }

    int rocprofsys_user_progress(const char* id)
    {
        return invoke(_callbacks.progress, id);
    }

    int rocprofsys_user_push_annotated_region(const char* id, annotation_t* _annotations,
                                              size_t _annotation_count)
    {
        return invoke(_callbacks.push_annotated_region, id, _annotations,
                      _annotation_count);
    }

    int rocprofsys_user_pop_annotated_region(const char* id, annotation_t* _annotations,
                                             size_t _annotation_count)
    {
        return invoke(_callbacks.pop_annotated_region, id, _annotations,
                      _annotation_count);
    }

    int rocprofsys_user_annotated_progress(const char* id, annotation_t* _annotations,
                                           size_t _annotation_count)
    {
        return invoke(_callbacks.annotated_progress, id, _annotations, _annotation_count);
    }

    int rocprofsys_user_configure(rocprofsys_user_configure_mode_t mode,
                                  rocprofsys_user_callbacks_t      inp,
                                  rocprofsys_user_callbacks_t*     out)
    {
        auto _former = _callbacks;

        switch(mode)
        {
            case ROCPROFSYS_USER_REPLACE_CONFIG:
            {
                _callbacks = inp;
                break;
            }
            case ROCPROFSYS_USER_UNION_CONFIG:
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
            case ROCPROFSYS_USER_INTERSECT_CONFIG:
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
                return ROCPROFSYS_USER_ERROR_INVALID_CATEGORY;
            }
        }

        if(out) *out = _former;

        return ROCPROFSYS_USER_SUCCESS;
    }

    const char* rocprofsys_user_error_string(int error_category)
    {
        switch(error_category)
        {
            case ROCPROFSYS_USER_SUCCESS: return "Success";
            case ROCPROFSYS_USER_ERROR_NO_BINDING: return "Function pointer not assigned";
            case ROCPROFSYS_USER_ERROR_BAD_VALUE: return "Invalid value was provided";
            case ROCPROFSYS_USER_ERROR_INVALID_CATEGORY:
                return "Invalid user binding category";
            case ROCPROFSYS_USER_ERROR_INTERNAL:
                return "An unknown error occurred within rocprof-sys library";
            default: break;
        }
        return "No error";
    }
}
