// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/** @file causal_api.h
 *
 * Minimal, near-zero-overhead library backing the ROCPROFSYS_CAUSAL_* macros
 * (see causal.h). An application links this tiny library directly; every
 * function is a no-op until librocprof-sys-dl.so is separately preloaded and
 * registers the real callbacks via rocprofsys_causal_register_callbacks().
 * This lets causal progress points ship in production code without pulling in
 * the full profiler backend.
 *
 * There is no general-purpose instrumentation API here (that role is now
 * served by rocprofiler-sdk-roctx) -- only what causal profiling needs.
 */

#ifndef ROCPROFSYS_CAUSAL_API_H_
#define ROCPROFSYS_CAUSAL_API_H_

#if defined(ROCPROFSYS_CAUSAL_API_SOURCE) && (ROCPROFSYS_CAUSAL_API_SOURCE > 0)
#    if !defined(ROCPROFSYS_CAUSAL_PUBLIC_API)
#        define ROCPROFSYS_CAUSAL_PUBLIC_API __attribute__((visibility("default")))
#    endif
#else
#    if !defined(ROCPROFSYS_CAUSAL_PUBLIC_API)
#        define ROCPROFSYS_CAUSAL_PUBLIC_API
#    endif
#endif

#include "rocprofiler-systems/annotation.h"

#include <stddef.h>

#if defined(__cplusplus)
extern "C"
{
#endif

    // NOLINTNEXTLINE(modernize-use-using) -- C-style typedef for C ABI compatibility
    typedef int (*rocprofsys_causal_region_func_t)(const char*);
    // NOLINTNEXTLINE(modernize-use-using) -- C-style typedef for C ABI compatibility
    typedef int (*rocprofsys_causal_annotated_func_t)(const char*,
                                                      rocprofsys_annotation_t*, size_t);

    /// @struct rocprofsys_causal_callbacks
    /// @brief Callbacks invoked by the causal API functions below. Registered by
    /// librocprof-sys-dl when it is preloaded; left null (no-op) otherwise.
    ///
    /// @typedef rocprofsys_causal_callbacks rocprofsys_causal_callbacks_t
    // NOLINTNEXTLINE(modernize-use-using) -- C-style typedef for C ABI compatibility
    typedef struct rocprofsys_causal_callbacks
    {
        rocprofsys_causal_region_func_t    begin;
        rocprofsys_causal_region_func_t    end;
        rocprofsys_causal_region_func_t    progress;
        rocprofsys_causal_annotated_func_t annotated_progress;
    } rocprofsys_causal_callbacks_t;

    /// @brief Starts a latency progress point (region of interest) with the given label.
    extern int rocprofsys_causal_begin(const char*) ROCPROFSYS_CAUSAL_PUBLIC_API;

    /// @brief Ends the latency progress point for the matching label.
    extern int rocprofsys_causal_end(const char*) ROCPROFSYS_CAUSAL_PUBLIC_API;

    /// @brief Adds a throughput progress point with the given label.
    extern int rocprofsys_causal_progress(const char*) ROCPROFSYS_CAUSAL_PUBLIC_API;

    /// @brief Adds a throughput progress point with the given label and annotations.
    extern int rocprofsys_causal_annotated_progress(const char*, rocprofsys_annotation_t*,
                                                    size_t) ROCPROFSYS_CAUSAL_PUBLIC_API;

    /// @brief Registers the callbacks invoked by the functions above. Called by
    /// librocprof-sys-dl once it dlopen's this library; not intended for direct use.
    extern void rocprofsys_causal_register_callbacks(rocprofsys_causal_callbacks_t)
        ROCPROFSYS_CAUSAL_PUBLIC_API;

#if defined(__cplusplus)
}
#endif

#endif  // ROCPROFSYS_CAUSAL_API_H_
