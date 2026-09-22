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
 * Every function is annotated with one of the version macros defined in this
 * section.  Each macro specifies a corresponding symbol version string.  After
 * dynamically loading the shared library with @p dlopen, the address of each
 * function can be obtained using @p dlsym with the name of the function and
 * its corresponding symbol version string.  An error will be reported by @p
 * dlvsym if the installed library does not support the version for the
 * function specified in this version of the interface.
 *
 * @{
 */

/**
 * @brief The function was introduced in version 0.0 of the interface and has the
 * symbol version string of ``"ROCPROFILER_SDK_ROCPD_0.0"``.
 */
#define ROCPROFILER_SDK_ROCPD_VERSION_0_0

/** @} */

#if !defined(ROCPD_ATTRIBUTE)
#    if defined(_MSC_VER)
#        define ROCPD_ATTRIBUTE(...) __declspec(__VA_ARGS__)
#    else
#        define ROCPD_ATTRIBUTE(...) __attribute__((__VA_ARGS__))
#    endif
#endif

#if !defined(ROCPD_PUBLIC_API)
#    if defined(_MSC_VER)
#        define ROCPD_PUBLIC_API ROCPD_ATTRIBUTE(dllexport)
#    else
#        define ROCPD_PUBLIC_API ROCPD_ATTRIBUTE(visibility("default"))
#    endif
#endif

#if !defined(ROCPD_HIDDEN_API)
#    if defined(_MSC_VER)
#        define ROCPD_HIDDEN_API
#    else
#        define ROCPD_HIDDEN_API ROCPD_ATTRIBUTE(visibility("hidden"))
#    endif
#endif

#if !defined(ROCPD_EXPORT_DECORATOR)
#    define ROCPD_EXPORT_DECORATOR ROCPD_PUBLIC_API
#endif

#if !defined(ROCPD_IMPORT_DECORATOR)
#    if defined(_MSC_VER)
#        define ROCPD_IMPORT_DECORATOR ROCPD_ATTRIBUTE(dllimport)
#    else
#        define ROCPD_IMPORT_DECORATOR
#    endif
#endif

#define ROCPD_EXPORT ROCPD_EXPORT_DECORATOR
#define ROCPD_IMPORT ROCPD_IMPORT_DECORATOR

#if !defined(ROCPD_API)
#    if defined(rocpd_EXPORTS)
#        define ROCPD_API ROCPD_EXPORT
#    else
#        define ROCPD_API ROCPD_IMPORT
#    endif
#endif

#if defined(__has_attribute)
#    if __has_attribute(nonnull)
#        define ROCPD_NONNULL(...) __attribute__((nonnull(__VA_ARGS__)))
#    else
#        define ROCPD_NONNULL(...)
#    endif
#else
#    if defined(__GNUC__)
#        define ROCPD_NONNULL(...) __attribute__((nonnull(__VA_ARGS__)))
#    else
#        define ROCPD_NONNULL(...)
#    endif
#endif

#ifdef __cplusplus
#    define ROCPD_EXTERN_C_INIT extern "C" {
#    define ROCPD_EXTERN_C_FINI }
#else
#    define ROCPD_EXTERN_C_INIT
#    define ROCPD_EXTERN_C_FINI
#endif

#if !defined(ROCPD_EXPERIMENTAL_WARNINGS)
#    define ROCPD_EXPERIMENTAL_WARNINGS 0
#endif

#define ROCPD_EXPERIMENTAL_MESSAGE                                                                 \
    ROCPD_DEPRECATED_MESSAGE("Note: this feature has been marked as experimental. Define "         \
                             "ROCPD_EXPERIMENTAL_WARNINGS=0 to silence this message.")

#if defined(ROCPD_EXPERIMENTAL_WARNINGS) && ROCPD_EXPERIMENTAL_WARNINGS > 0
#    define ROCPD_EXPERIMENTAL ROCPD_EXPERIMENTAL_MESSAGE
#else
#    define ROCPD_EXPERIMENTAL
#endif
