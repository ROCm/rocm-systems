// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/** @file causal.h */

#ifndef ROCPROFSYS_CAUSAL_H_
#define ROCPROFSYS_CAUSAL_H_

/**
 * @defgroup ROCPROFSYS_CAUSAL_GROUP ROCm Systems Profiler Causal Profiling Defines
 *
 * @{
 */

#if !defined(ROCPROFSYS_CAUSAL_ENABLED)
/** Preprocessor switch to enable/disable instrumentation for causal profiling */
#    define ROCPROFSYS_CAUSAL_ENABLED 1
#endif

#if ROCPROFSYS_CAUSAL_ENABLED > 0
#    include <rocprofiler-systems/causal_api.h>  // NOLINT(misc-include-cleaner)

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
            rocprofsys_causal_progress(ROCPROFSYS_CAUSAL_LABEL);
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_PROGRESS_NAMED)
/** Adds a throughput progress point with user defined label. Each instance should use a
 * unique label. */
#        define ROCPROFSYS_CAUSAL_PROGRESS_NAMED(LABEL) rocprofsys_causal_progress(LABEL);
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_BEGIN)
/** Starts a latency progress point (region of interest) with user defined label. Each
 * instance should use a unique label. */
#        define ROCPROFSYS_CAUSAL_BEGIN(LABEL) rocprofsys_causal_begin(LABEL);
#    endif
#    if !defined(ROCPROFSYS_CAUSAL_END)
/** End the latency progress point (region of interest) for the matching user defined
 * label. */
#        define ROCPROFSYS_CAUSAL_END(LABEL) rocprofsys_causal_end(LABEL);
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
