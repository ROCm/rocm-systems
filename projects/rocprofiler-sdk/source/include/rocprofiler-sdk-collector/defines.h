// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @defgroup SYMBOL_VERSIONING_GROUP Symbol Versions
 *
 * @brief The names used for the shared library versioned symbols.
 *
 * @{
 */

/**
 * @brief The function was introduced in version 0.0 of the interface and has the
 * symbol version string of ``"ROCPROFILER_SDK_COLLECTOR_0.0"``.
 */
#define ROCPROFILER_SDK_COLLECTOR_VERSION_0_0

/** @} */

#if !defined(ROCPROFILER_COLLECTOR_ATTRIBUTE)
#    if defined(_MSC_VER)
#        define ROCPROFILER_COLLECTOR_ATTRIBUTE(...) __declspec(__VA_ARGS__)
#    else
#        define ROCPROFILER_COLLECTOR_ATTRIBUTE(...) __attribute__((__VA_ARGS__))
#    endif
#endif

#if !defined(ROCPROFILER_COLLECTOR_PUBLIC_API)
#    if defined(_MSC_VER)
#        define ROCPROFILER_COLLECTOR_PUBLIC_API ROCPROFILER_COLLECTOR_ATTRIBUTE(dllexport)
#    else
#        define ROCPROFILER_COLLECTOR_PUBLIC_API                                                   \
            ROCPROFILER_COLLECTOR_ATTRIBUTE(visibility("default"))
#    endif
#endif

#if !defined(ROCPROFILER_COLLECTOR_IMPORT_DECORATOR)
#    if defined(_MSC_VER)
#        define ROCPROFILER_COLLECTOR_IMPORT_DECORATOR ROCPROFILER_COLLECTOR_ATTRIBUTE(dllimport)
#    else
#        define ROCPROFILER_COLLECTOR_IMPORT_DECORATOR
#    endif
#endif

#if !defined(ROCPROFILER_COLLECTOR_API)
#    if defined(ROCPROFILER_COLLECTOR_EXPORTS)
#        define ROCPROFILER_COLLECTOR_API ROCPROFILER_COLLECTOR_PUBLIC_API
#    else
#        define ROCPROFILER_COLLECTOR_API ROCPROFILER_COLLECTOR_IMPORT_DECORATOR
#    endif
#endif

#if !defined(ROCPROFILER_COLLECTOR_NONNULL)
#    if defined(__has_attribute)
#        if __has_attribute(nonnull)
#            define ROCPROFILER_COLLECTOR_NONNULL(...) __attribute__((nonnull(__VA_ARGS__)))
#        else
#            define ROCPROFILER_COLLECTOR_NONNULL(...)
#        endif
#    elif defined(__GNUC__)
#        define ROCPROFILER_COLLECTOR_NONNULL(...) __attribute__((nonnull(__VA_ARGS__)))
#    else
#        define ROCPROFILER_COLLECTOR_NONNULL(...)
#    endif
#endif

#ifdef __cplusplus
#    define ROCPROFILER_COLLECTOR_EXTERN_C_INIT extern "C" {
#    define ROCPROFILER_COLLECTOR_EXTERN_C_FINI }
#else
#    define ROCPROFILER_COLLECTOR_EXTERN_C_INIT
#    define ROCPROFILER_COLLECTOR_EXTERN_C_FINI
#endif
