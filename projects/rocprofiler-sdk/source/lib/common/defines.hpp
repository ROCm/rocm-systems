// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/defines.h>

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

#define ROCPROFILER_COMPUTE_VERSION(MAJOR, MINOR, PATCH)                                           \
    ROCPROFILER_SDK_COMPUTE_VERSION(MAJOR, MINOR, PATCH)

// Below are used in HSA, HIP, and Marker API tracing
#define IMPL_DETAIL_EXPAND(X) X
#define IMPL_DETAIL_FOR_EACH_NARG(...)                                                             \
    IMPL_DETAIL_FOR_EACH_NARG_(__VA_ARGS__, IMPL_DETAIL_FOR_EACH_RSEQ_N())
#define IMPL_DETAIL_FOR_EACH_NARG_(...) IMPL_DETAIL_EXPAND(IMPL_DETAIL_FOR_EACH_ARG_N(__VA_ARGS__))
#define IMPL_DETAIL_FOR_EACH_ARG_N(                                                                \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...)                      \
    N
#define IMPL_DETAIL_FOR_EACH_RSEQ_N() 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
#define IMPL_DETAIL_CONCATENATE(X, Y) X##Y
#define IMPL_DETAIL_FOR_EACH_(N, MACRO, PREFIX, ...)                                               \
    IMPL_DETAIL_EXPAND(IMPL_DETAIL_CONCATENATE(MACRO, N)(PREFIX, __VA_ARGS__))
#define IMPL_DETAIL_FOR_EACH(MACRO, PREFIX, ...)                                                   \
    IMPL_DETAIL_FOR_EACH_(IMPL_DETAIL_FOR_EACH_NARG(__VA_ARGS__), MACRO, PREFIX, __VA_ARGS__)

#define ADDR_MEMBER_0(...)
#define ADDR_MEMBER_1(PREFIX, FIELD)      static_cast<void*>(&PREFIX.FIELD)
#define ADDR_MEMBER_2(PREFIX, A, B)       ADDR_MEMBER_1(PREFIX, A), ADDR_MEMBER_1(PREFIX, B)
#define ADDR_MEMBER_3(PREFIX, A, B, C)    ADDR_MEMBER_2(PREFIX, A, B), ADDR_MEMBER_1(PREFIX, C)
#define ADDR_MEMBER_4(PREFIX, A, B, C, D) ADDR_MEMBER_3(PREFIX, A, B, C), ADDR_MEMBER_1(PREFIX, D)
#define ADDR_MEMBER_5(PREFIX, A, B, C, D, E)                                                       \
    ADDR_MEMBER_4(PREFIX, A, B, C, D), ADDR_MEMBER_1(PREFIX, E)
#define ADDR_MEMBER_6(PREFIX, A, B, C, D, E, F)                                                    \
    ADDR_MEMBER_5(PREFIX, A, B, C, D, E), ADDR_MEMBER_1(PREFIX, F)
#define ADDR_MEMBER_7(PREFIX, A, B, C, D, E, F, G)                                                 \
    ADDR_MEMBER_6(PREFIX, A, B, C, D, E, F), ADDR_MEMBER_1(PREFIX, G)
#define ADDR_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H)                                              \
    ADDR_MEMBER_7(PREFIX, A, B, C, D, E, F, G), ADDR_MEMBER_1(PREFIX, H)
#define ADDR_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I)                                           \
    ADDR_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H), ADDR_MEMBER_1(PREFIX, I)
#define ADDR_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J)                                       \
    ADDR_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I), ADDR_MEMBER_1(PREFIX, J)
#define ADDR_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K)                                    \
    ADDR_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J), ADDR_MEMBER_1(PREFIX, K)
#define ADDR_MEMBER_12(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L)                                 \
    ADDR_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K), ADDR_MEMBER_1(PREFIX, L)
#define ADDR_MEMBER_13(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M)                              \
    ADDR_MEMBER_12(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L), ADDR_MEMBER_1(PREFIX, M)
#define ADDR_MEMBER_14(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M, N)                           \
    ADDR_MEMBER_13(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M), ADDR_MEMBER_1(PREFIX, N)
#define ADDR_MEMBER_15(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)                        \
    ADDR_MEMBER_14(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M, N), ADDR_MEMBER_1(PREFIX, O)

#define NAMED_MEMBER_0(...)
#define NAMED_MEMBER_1(PREFIX, FIELD)   std::make_pair(#FIELD, PREFIX.FIELD)
#define NAMED_MEMBER_2(PREFIX, A, B)    NAMED_MEMBER_1(PREFIX, A), NAMED_MEMBER_1(PREFIX, B)
#define NAMED_MEMBER_3(PREFIX, A, B, C) NAMED_MEMBER_2(PREFIX, A, B), NAMED_MEMBER_1(PREFIX, C)
#define NAMED_MEMBER_4(PREFIX, A, B, C, D)                                                         \
    NAMED_MEMBER_3(PREFIX, A, B, C), NAMED_MEMBER_1(PREFIX, D)
#define NAMED_MEMBER_5(PREFIX, A, B, C, D, E)                                                      \
    NAMED_MEMBER_4(PREFIX, A, B, C, D), NAMED_MEMBER_1(PREFIX, E)
#define NAMED_MEMBER_6(PREFIX, A, B, C, D, E, F)                                                   \
    NAMED_MEMBER_5(PREFIX, A, B, C, D, E), NAMED_MEMBER_1(PREFIX, F)
#define NAMED_MEMBER_7(PREFIX, A, B, C, D, E, F, G)                                                \
    NAMED_MEMBER_6(PREFIX, A, B, C, D, E, F), NAMED_MEMBER_1(PREFIX, G)
#define NAMED_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H)                                             \
    NAMED_MEMBER_7(PREFIX, A, B, C, D, E, F, G), NAMED_MEMBER_1(PREFIX, H)
#define NAMED_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I)                                          \
    NAMED_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H), NAMED_MEMBER_1(PREFIX, I)
#define NAMED_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J)                                      \
    NAMED_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I), NAMED_MEMBER_1(PREFIX, J)
#define NAMED_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K)                                   \
    NAMED_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J), NAMED_MEMBER_1(PREFIX, K)
#define NAMED_MEMBER_12(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L)                                \
    NAMED_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K), NAMED_MEMBER_1(PREFIX, L)
#define NAMED_MEMBER_13(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M)                             \
    NAMED_MEMBER_12(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L), NAMED_MEMBER_1(PREFIX, M)
#define NAMED_MEMBER_14(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M, N)                          \
    NAMED_MEMBER_13(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M), NAMED_MEMBER_1(PREFIX, N)
#define NAMED_MEMBER_15(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)                       \
    NAMED_MEMBER_14(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M, N), NAMED_MEMBER_1(PREFIX, O)

#define GET_ADDR_MEMBER_FIELDS(VAR, ...)  IMPL_DETAIL_FOR_EACH(ADDR_MEMBER_, VAR, __VA_ARGS__)
#define GET_NAMED_MEMBER_FIELDS(VAR, ...) IMPL_DETAIL_FOR_EACH(NAMED_MEMBER_, VAR, __VA_ARGS__)
