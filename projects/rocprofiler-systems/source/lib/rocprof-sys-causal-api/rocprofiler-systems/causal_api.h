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
#    if !defined(ROCPROFSYS_PUBLIC_API)
#        define ROCPROFSYS_PUBLIC_API __attribute__((visibility("default")))
#    endif
#else
#    if !defined(ROCPROFSYS_PUBLIC_API)
#        define ROCPROFSYS_PUBLIC_API
#    endif
#endif

#include "rocprofiler-systems/annotation.h"

#include <stddef.h>

#if defined(__cplusplus)
extern "C"
{
#endif

    typedef int (*rocprofsys_causal_region_func_t)(const char*);
    typedef int (*rocprofsys_causal_annotated_func_t)(const char*,
                                                      rocprofsys_annotation_t*, size_t);

    /// @struct rocprofsys_causal_callbacks
    /// @brief Callbacks invoked by the causal API functions below. Registered by
    /// librocprof-sys-dl when it is preloaded; left null (no-op) otherwise.
    ///
    /// @typedef rocprofsys_causal_callbacks rocprofsys_causal_callbacks_t
    typedef struct rocprofsys_causal_callbacks
    {
        rocprofsys_causal_region_func_t    begin;
        rocprofsys_causal_region_func_t    end;
        rocprofsys_causal_region_func_t    progress;
        rocprofsys_causal_annotated_func_t annotated_progress;
    } rocprofsys_causal_callbacks_t;

#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
extern "C"
{
#endif

    /// @brief Starts a latency progress point (region of interest) with the given label.
    extern int rocprofsys_causal_begin(const char*) ROCPROFSYS_PUBLIC_API;

    /// @brief Ends the latency progress point for the matching label.
    extern int rocprofsys_causal_end(const char*) ROCPROFSYS_PUBLIC_API;

    /// @brief Adds a throughput progress point with the given label.
    extern int rocprofsys_causal_progress(const char*) ROCPROFSYS_PUBLIC_API;

    /// @brief Adds a throughput progress point with the given label and annotations.
    extern int rocprofsys_causal_annotated_progress(const char*, rocprofsys_annotation_t*,
                                                    size_t) ROCPROFSYS_PUBLIC_API;

    /// @brief Registers the callbacks invoked by the functions above. Called by
    /// librocprof-sys-dl once it dlopen's this library; not intended for direct use.
    extern void rocprofsys_causal_register_callbacks(rocprofsys_causal_callbacks_t)
        ROCPROFSYS_PUBLIC_API;

#if defined(__cplusplus)
}
#endif

#endif  // ROCPROFSYS_CAUSAL_API_H_
