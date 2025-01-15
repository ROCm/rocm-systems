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

#ifndef ROCPROFSYS_TYPES_H_
#define ROCPROFSYS_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C"
{
#endif

    struct rocprofsys_annotation;
    typedef int (*rocprofsys_trace_func_t)(void);
    typedef int (*rocprofsys_region_func_t)(const char*);
    typedef int (*rocprofsys_annotated_region_func_t)(const char*, rocprofsys_annotation*,
                                                      size_t);

    /// @struct rocprofsys_user_callbacks
    /// @brief Struct containing the callbacks for the user API
    ///
    /// @typedef rocprofsys_user_callbacks rocprofsys_user_callbacks_t
    typedef struct rocprofsys_user_callbacks
    {
        rocprofsys_trace_func_t            start_trace;
        rocprofsys_trace_func_t            stop_trace;
        rocprofsys_trace_func_t            start_thread_trace;
        rocprofsys_trace_func_t            stop_thread_trace;
        rocprofsys_region_func_t           push_region;
        rocprofsys_region_func_t           pop_region;
        rocprofsys_region_func_t           progress;
        rocprofsys_annotated_region_func_t push_annotated_region;
        rocprofsys_annotated_region_func_t pop_annotated_region;
        rocprofsys_annotated_region_func_t annotated_progress;

        /// @var start_trace
        /// @brief callback for enabling tracing globally
        /// @var stop_trace
        /// @brief callback for disabling tracing globally
        /// @var start_thread_trace
        /// @brief callback for enabling tracing on current thread
        /// @var stop_thread_trace
        /// @brief callback for disabling tracing on current thread
        /// @var push_region
        /// @brief callback for starting a trace region
        /// @var pop_region
        /// @brief callback for ending a trace region
        /// @var progress
        /// @brief callback for marking an causal profiling event
        /// @var push_annotated_region
        /// @brief callback for starting a trace region + annotations
        /// @var pop_annotated_region
        /// @brief callback for ending a trace region + annotations
        /// @var annotated_progress
        /// @brief callback for marking an causal profiling event + annotations
    } rocprofsys_user_callbacks_t;

    /// @enum ROCPROFSYS_USER_CONFIGURE_MODE
    /// @brief Identifier for errors
    /// @typedef ROCPROFSYS_USER_CONFIGURE_MODE rocprofsys_user_configure_mode_t
    typedef enum ROCPROFSYS_USER_CONFIGURE_MODE
    {
        // clang-format off
        ROCPROFSYS_USER_UNION_CONFIG = 0,    ///< Replace the callbacks in the current config with the non-null callbacks in the provided config
        ROCPROFSYS_USER_REPLACE_CONFIG,      ///< Replace the entire config even if the provided config has null callbacks
        ROCPROFSYS_USER_INTERSECT_CONFIG,    ///< Produce a config which is the intersection of the current config and the provided config
        ROCPROFSYS_USER_CONFIGURE_MODE_LAST
        // clang-format on
    } rocprofsys_user_configure_mode_t;

    /// @enum ROCPROFSYS_USER_ERROR
    /// @brief Identifier for errors
    /// @typedef ROCPROFSYS_USER_ERROR rocprofsys_user_error_t
    ///
    typedef enum ROCPROFSYS_USER_ERROR
    {
        ROCPROFSYS_USER_SUCCESS = 0,             ///< No error
        ROCPROFSYS_USER_ERROR_NO_BINDING,        ///< Function pointer was not assigned
        ROCPROFSYS_USER_ERROR_BAD_VALUE,         ///< Provided value was invalid
        ROCPROFSYS_USER_ERROR_INVALID_CATEGORY,  ///< Invalid user binding category
        ROCPROFSYS_USER_ERROR_INTERNAL,          ///< Internal error occurred within
                                                 ///< librocprof-sys
        ROCPROFSYS_USER_ERROR_LAST
    } rocprofsys_user_error_t;

#if defined(__cplusplus)
}
#endif

#ifndef ROCPROFSYS_USER_CALLBACKS_INIT
#    define ROCPROFSYS_USER_CALLBACKS_INIT                                               \
        {                                                                                \
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL                   \
        }
#endif

#endif  // ROCPROFSYS_TYPES_H_
