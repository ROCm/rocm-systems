// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#pragma once

#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"

//======================================================================================//
//
#if !defined(TIMEMORY_ATTRIBUTE)
#    if defined(TIMEMORY_MSVC_COMPILER)
#        define TIMEMORY_ATTRIBUTE(...) __declspec(__VA_ARGS__)
#    else
#        define TIMEMORY_ATTRIBUTE(...) __attribute__((__VA_ARGS__))
#    endif
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_ALWAYS_INLINE)
#    if defined(TIMEMORY_MSVC_COMPILER)
#        define TIMEMORY_ALWAYS_INLINE __forceinline
#    else
#        define TIMEMORY_ALWAYS_INLINE TIMEMORY_ATTRIBUTE(always_inline) inline
#    endif
#endif

#if !defined(TIMEMORY_INLINE)
#    define TIMEMORY_INLINE TIMEMORY_ALWAYS_INLINE
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_NODISCARD)
#    if defined(CXX17)
#        define TIMEMORY_NODISCARD [[nodiscard]]
#    else
#        define TIMEMORY_NODISCARD
#    endif
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_FLATTEN)
#    if !defined(TIMEMORY_MSVC_COMPILER)
#        define TIMEMORY_FLATTEN [[gnu::flatten]]
#    else
#        define TIMEMORY_FLATTEN
#    endif
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_HOT)
#    if !defined(TIMEMORY_MSVC_COMPILER)
#        define TIMEMORY_HOT TIMEMORY_ATTRIBUTE(hot)
#    else
#        define TIMEMORY_HOT
#    endif
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_COLD)
#    if !defined(TIMEMORY_MSVC_COMPILER)
#        define TIMEMORY_COLD TIMEMORY_ATTRIBUTE(cold)
#    else
#        define TIMEMORY_COLD
#    endif
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_CONST)
#    define TIMEMORY_CONST [[gnu::const]]
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DEPRECATED)
#    define TIMEMORY_DEPRECATED(...) [[gnu::deprecated(__VA_ARGS__)]]
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_EXTERN_VISIBLE)
#    define TIMEMORY_EXTERN_VISIBLE [[gnu::externally_visible]]
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_HIDDEN)
#    if !defined(TIMEMORY_MSVC_COMPILER)
#        define TIMEMORY_HIDDEN TIMEMORY_ATTRIBUTE(visibility("hidden"))
#    else
#        define TIMEMORY_HIDDEN
#    endif
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_ALIAS)
#    define TIMEMORY_ALIAS(...) [[gnu::alias(__VA_ARGS__)]]
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_NOINLINE)
#    define TIMEMORY_NOINLINE TIMEMORY_ATTRIBUTE(noinline)
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_NOCLONE)
#    if defined(TIMEMORY_GNU_COMPILER)
#        define TIMEMORY_NOCLONE TIMEMORY_ATTRIBUTE(noclone)
#    else
#        define TIMEMORY_NOCLONE
#    endif
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_HOT_INLINE)
#    define TIMEMORY_HOT_INLINE TIMEMORY_HOT TIMEMORY_INLINE
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DATA_ALIGNMENT)
#    define TIMEMORY_DATA_ALIGNMENT 8
#endif

#if !defined(TIMEMORY_PACKED_ALIGNMENT)
#    if defined(_WIN32)  // Windows 32- and 64-bit
#        define TIMEMORY_PACKED_ALIGNMENT __declspec(align(TIMEMORY_DATA_ALIGNMENT))
#    elif defined(__GNUC__)  // GCC
#        define TIMEMORY_PACKED_ALIGNMENT                                                \
            __attribute__((__packed__)) __attribute__((aligned(TIMEMORY_DATA_ALIGNMENT)))
#    else  // all other compilers
#        define TIMEMORY_PACKED_ALIGNMENT
#    endif
#endif

//======================================================================================//
//  device decorators
//
#if defined(__CUDACC__) || defined(__HIPCC__)
#    define TIMEMORY_LAMBDA __host__ __device__
#    define TIMEMORY_HOST_LAMBDA __host__
#    define TIMEMORY_HOST_FUNCTION __host__
#    define TIMEMORY_DEVICE_LAMBDA __device__
#    define TIMEMORY_DEVICE_FUNCTION __device__
#    define TIMEMORY_GLOBAL_FUNCTION __global__
#    define TIMEMORY_HOST_DEVICE_FUNCTION __host__ __device__
#    define TIMEMORY_DEVICE_INLINE __device__ __inline__
#    define TIMEMORY_GLOBAL_INLINE __global__ __inline__
#    define TIMEMORY_HOST_DEVICE_INLINE __host__ __device__ __inline__
#else
#    define TIMEMORY_LAMBDA
#    define TIMEMORY_HOST_LAMBDA
#    define TIMEMORY_HOST_FUNCTION
#    define TIMEMORY_DEVICE_LAMBDA
#    define TIMEMORY_DEVICE_FUNCTION
#    define TIMEMORY_GLOBAL_FUNCTION
#    define TIMEMORY_HOST_DEVICE_FUNCTION
#    define TIMEMORY_DEVICE_INLINE inline
#    define TIMEMORY_GLOBAL_INLINE inline
#    define TIMEMORY_HOST_DEVICE_INLINE inline
#endif
