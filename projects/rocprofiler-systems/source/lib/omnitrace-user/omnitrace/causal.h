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

/** @file causal.h */

#ifndef OMNITRACE_CAUSAL_H_
#define OMNITRACE_CAUSAL_H_

/**
 * @defgroup OMNITRACE_CASUAL_GROUP OmniTrace Causal Profiling Defines
 *
 * @{
 */

#if !defined(OMNITRACE_CAUSAL_ENABLED)
/** Preprocessor switch to enable/disable instrumentation for causal profiling */
#    define OMNITRACE_CAUSAL_ENABLED 1
#endif

#if OMNITRACE_CAUSAL_ENABLED > 0
#    include <omnitrace/user.h>

#    if !defined(OMNITRACE_CAUSAL_LABEL)
/** @cond OMNITRACE_HIDDEN_DEFINES */
#        define OMNITRACE_CAUSAL_STR2(x) #        x
#        define OMNITRACE_CAUSAL_STR(x)  OMNITRACE_CAUSAL_STR2(x)
/** @endcond */
/** Default label for a causal progress point */
#        define OMNITRACE_CAUSAL_LABEL __FILE__ ":" OMNITRACE_CAUSAL_STR(__LINE__)
#    endif
#    if !defined(OMNITRACE_CAUSAL_PROGRESS)
/** Adds a throughput progress point with label `<file>:<line>` */
#        define OMNITRACE_CAUSAL_PROGRESS omnitrace_user_progress(OMNITRACE_CAUSAL_LABEL);
#    endif
#    if !defined(OMNITRACE_CAUSAL_PROGRESS_NAMED)
/** Adds a throughput progress point with user defined label. Each instance should use a
 * unique label. */
#        define OMNITRACE_CAUSAL_PROGRESS_NAMED(LABEL) omnitrace_user_progress(LABEL);
#    endif
#    if !defined(OMNITRACE_CAUSAL_BEGIN)
/** Starts a latency progress point (region of interest) with user defined label. Each
 * instance should use a unique label. */
#        define OMNITRACE_CAUSAL_BEGIN(LABEL) omnitrace_user_push_region(LABEL);
#    endif
#    if !defined(OMNITRACE_CAUSAL_END)
/** End the latency progress point (region of interest) for the matching user defined
 * label. */
#        define OMNITRACE_CAUSAL_END(LABEL) omnitrace_user_pop_region(LABEL);
#    endif
#else
#    if !defined(OMNITRACE_CAUSAL_PROGRESS)
#        define OMNITRACE_CAUSAL_PROGRESS
#    endif
#    if !defined(OMNITRACE_CAUSAL_PROGRESS_NAMED)
#        define OMNITRACE_CAUSAL_PROGRESS_NAMED(LABEL)
#    endif
#    if !defined(OMNITRACE_CAUSAL_BEGIN)
#        define OMNITRACE_CAUSAL_BEGIN(LABEL)
#    endif
#    if !defined(OMNITRACE_CAUSAL_END)
#        define OMNITRACE_CAUSAL_END(LABEL)
#    endif
#endif

/** @} */

#endif  // OMNITRACE_CAUSAL_H_
