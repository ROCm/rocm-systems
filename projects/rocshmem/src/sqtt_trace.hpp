/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef SQTT_TRACE_HPP_
#define SQTT_TRACE_HPP_

#include <hip/hip_runtime.h>

/**
 * @file sqtt_trace.hpp
 * @brief SQTT (SQ Thread Trace) instrumentation markers for ROCshmem
 *
 * These functions insert SQTT user event markers into the command stream
 * for performance profiling and analysis.
 */

#ifdef __HIP_DEVICE_COMPILE__

/**
 * @brief Insert an SQTT marker indicating entry into a code section
 * @param marker_name String identifier for the section being entered
 */
__device__ __forceinline__ void sqtt_marker_enter(const char* marker_name) {
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // Use s_code_end as a profiling marker
    // The marker ID is encoded in the immediate operand
    __builtin_amdgcn_s_code_end();
#endif
}

/**
 * @brief Insert an SQTT marker indicating exit from a code section
 * @param marker_name String identifier for the section being exited (should match entry)
 */
__device__ __forceinline__ void sqtt_marker_exit(const char* marker_name) {
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // Use s_code_end as a profiling marker
    __builtin_amdgcn_s_code_end();
#endif
}

#else  // __HIP_DEVICE_COMPILE__

// Host-side stubs (no-ops)
inline void sqtt_marker_enter(const char* marker_name) {}
inline void sqtt_marker_exit(const char* marker_name) {}

#endif  // __HIP_DEVICE_COMPILE__

#endif  // SQTT_TRACE_HPP_
