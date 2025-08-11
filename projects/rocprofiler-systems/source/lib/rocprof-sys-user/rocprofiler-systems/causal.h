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

/** @file causal.h */

#ifndef ROCPROFSYS_CAUSAL_H_
#define ROCPROFSYS_CAUSAL_H_

/**
 * @defgroup ROCPROFSYS_CASUAL_GROUP ROCm Systems Profiler Causal Profiling Defines
 *
 * @{
 */

#if !defined(ROCPROFSYS_CAUSAL_ENABLED)
/** Preprocessor switch to enable/disable instrumentation for causal profiling */
#    define ROCPROFSYS_CAUSAL_ENABLED 1
#endif

#if ROCPROFSYS_CAUSAL_ENABLED > 0
#    include <rocprofiler-systems/user.h>

#    if !defined(ROCPROFSYS_CAUSAL_LABEL)
/** @cond ROCPROFSYS_HIDDEN_DEFINES */
#        define ROCPROFSYS_CAUSAL_STR2(x) #x
#        define ROCPROFSYS_CAUSAL_STR(x)  ROCPROFSYS_CAUSAL_STR2(x)
/** @endcond */
/** Default label for a causal progress point */
#        define ROCPROFSYS_CAUSAL_LABEL __FILE__ ":" ROCPROFSYS_CAUSAL_STR(__LINE__)
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_PROGRESS)
/** Adds a throughput progress point with label `<file>:<line>` */
#        define ROCPROFSYS_CAUSAL_PROGRESS                                               \
            rocprofsys_user_progress(ROCPROFSYS_CAUSAL_LABEL);
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_PROGRESS_NAMED)
/** Adds a throughput progress point with user defined label. Each instance should use a
 * unique label. */
#        define ROCPROFSYS_CAUSAL_PROGRESS_NAMED(LABEL) rocprofsys_user_progress(LABEL);
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_BEGIN)
/** Starts a latency progress point (region of interest) with user defined label. Each
 * instance should use a unique label. */
#        define ROCPROFSYS_CAUSAL_BEGIN(LABEL) rocprofsys_user_push_region(LABEL);
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_END)
/** End the latency progress point (region of interest) for the matching user defined
 * label. */
#        define ROCPROFSYS_CAUSAL_END(LABEL) rocprofsys_user_pop_region(LABEL);
#    endif
#else
#    if !defined(ROCPROFSYS_CAUSAL_PROGRESS)
#        define ROCPROFSYS_CAUSAL_PROGRESS
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_PROGRESS_NAMED)
#        define ROCPROFSYS_CAUSAL_PROGRESS_NAMED(LABEL)
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_BEGIN)
#        define ROCPROFSYS_CAUSAL_BEGIN(LABEL)
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_END)
#        define ROCPROFSYS_CAUSAL_END(LABEL)
#    endif
#endif

/** @} */

#endif  // ROCPROFSYS_CAUSAL_H_
