// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler/defines.h>

#define ROCPROFILER_VISIBILITY(MODE)  ROCPROFILER_ATTRIBUTE(visibility(MODE))
#define ROCPROFILER_INTERNAL_API      ROCPROFILER_VISIBILITY("internal")
#define ROCPROFILER_INLINE            ROCPROFILER_ATTRIBUTE(always_inline) inline
#define ROCPROFILER_NOINLINE          ROCPROFILER_ATTRIBUTE(noinline)
#define ROCPROFILER_HOT               ROCPROFILER_ATTRIBUTE(hot)
#define ROCPROFILER_COLD              ROCPROFILER_ATTRIBUTE(cold)
#define ROCPROFILER_CONST             ROCPROFILER_ATTRIBUTE(const)
#define ROCPROFILER_PURE              ROCPROFILER_ATTRIBUTE(pure)
#define ROCPROFILER_WEAK              ROCPROFILER_ATTRIBUTE(weak)
#define ROCPROFILER_PACKED            ROCPROFILER_ATTRIBUTE(__packed__)
#define ROCPROFILER_PACKED_ALIGN(VAL) ROCPROFILER_PACKED ROCPROFILER_ATTRIBUTE(__aligned__(VAL))
#define ROCPROFILER_LIKELY(...)       __builtin_expect((__VA_ARGS__), 1)
#define ROCPROFILER_UNLIKELY(...)     __builtin_expect((__VA_ARGS__), 0)

#if defined(ROCPROFILER_CI) && ROCPROFILER_CI > 0
#    if defined(NDEBUG)
#        undef NDEBUG
#    endif
#    if !defined(DEBUG)
#        define DEBUG 1
#    endif
#    if defined(__cplusplus)
#        include <cassert>
#    else
#        include <assert.h>
#    endif
#endif

#define ROCPROFILER_STRINGIZE(X)           ROCPROFILER_STRINGIZE2(X)
#define ROCPROFILER_STRINGIZE2(X)          #X
#define ROCPROFILER_VAR_NAME_COMBINE(X, Y) X##Y
#define ROCPROFILER_VARIABLE(X, Y)         ROCPROFILER_VAR_NAME_COMBINE(X, Y)
#define ROCPROFILER_LINESTR                ROCPROFILER_STRINGIZE(__LINE__)
#define ROCPROFILER_ESC(...)               __VA_ARGS__

#if defined(__cplusplus)
#    if !defined(ROCPROFILER_FOLD_EXPRESSION)
#        define ROCPROFILER_FOLD_EXPRESSION(...) ((__VA_ARGS__), ...)
#    endif
#endif

#define ROCPROFILER_COMPUTE_VERSION(MAJOR, MINOR, PATCH) ((10000 * MAJOR) + (100 * MINOR) + (PATCH))
