/* Copyright (c) 2011 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef MACROS_HPP_
#define MACROS_HPP_

#ifndef OPENCL_EXPORTS
#define OPENCL_EXPORTS 1
#endif  // OPENCL_EXPORTS

#if defined(NDEBUG)
#define RELEASE 1
#else  // !NDEBUG
#define ASSERT 1
#define DEBUG 1
#endif  // !NDEBUG

#if defined(_WIN64) && !defined(_LP64)
#define _LP64 1
#endif

#if defined(_DEBUG) && !defined(DEBUG)
#define DEBUG 1
#endif  // _DEBUG && !DEBUG

#if defined(DEBUG) && defined(RELEASE)
#error "Build Error: cannot have both -DDEBUG and -DRELEASE"
#endif /*DEBUG && RELEASE*/

#if !defined(DEBUG) && !defined(RELEASE)
#error "Build Error: must have either -DDEBUG or -DRELEASE"
#endif /*DEBUG && RELEASE*/

#ifdef DEBUG
#define DEBUG_ONLY(x) x
#define RELEASE_ONLY(x)
#define IS_DEBUG true
#else  // !DEBUG
#define DEBUG_ONLY(x)
#define RELEASE_ONLY(x) x
#define IS_DEBUG false
#endif /*!DEBUG*/
#define DEBUG_SWITCH(d, r) DEBUG_ONLY(d) RELEASE_ONLY(r)
#define RELEASE_SWITCH(r, d) RELEASE_ONLY(r) DEBUG_ONLY(d)

//! \brief Make a c-string of __macro__
#define STR(__macro__) #__macro__
//! \brief Make a c-string of the expansion of __macro__
#define XSTR(__macro__) STR(__macro__)
//! \brief Concatenate 2 symbols
#define CONCAT(a, b) a##b
#define XCONCAT(a, b) CONCAT(a, b)


//! \cond ignore
#ifdef _LP64
#define LP64_ONLY(x) x
#define NOT_LP64(x)
#else  // !_LP64
#define LP64_ONLY(x)
#define NOT_LP64(x) x
#endif /*!_LP64*/
#define LP64_SWITCH(lp32, lp64) NOT_LP64(lp32) LP64_ONLY(lp64)

#ifdef __linux__
#define IS_LINUX true
#define LINUX_ONLY(x) x
#define NOT_LINUX(x)
#else  // !__linux__
#define LINUX_ONLY(x)
#define NOT_LINUX(x) x
#endif /*!__linux__*/

#ifdef __APPLE__
#define IS_MACOS true
#define MACOS_ONLY(x) x
#define NOT_MACOS(x)
#else  // !__APPLE__
#define MACOS_ONLY(x)
#define NOT_MACOS(x) x
#endif /*!__APPLE__*/

#ifdef _WIN32
#define IS_WINDOWS true
#define WINDOWS_ONLY(x) x
#define NOT_WINDOWS(x)
#else  // !_WIN32
#define WINDOWS_ONLY(x)
#define NOT_WINDOWS(x) x
#endif /*!_WIN32*/

#ifdef _WIN64
#define WIN64_ONLY(x) x
#define NOT_WIN64(x)
#else  // !_WIN64
#define WIN64_ONLY(x)
#define NOT_WIN64(x) x
#endif /*!_WIN64*/

#define IS_MAINLINE true

#ifndef IS_LINUX
#define IS_LINUX false
#endif
#ifndef IS_MACOS
#define IS_MACOS false
#endif
#ifndef IS_WINDOWS
#define IS_WINDOWS false
#endif

#if defined(__AVX512F__)
#define NATIVE_ALIGNMENT_SIZE 64
#elif defined(__AVX__)
#define NATIVE_ALIGNMENT_SIZE 32
#else
#define NATIVE_ALIGNMENT_SIZE 16
#endif

#define IF_LEFT_true(x) x
#define IF_LEFT_false(x)
#define IF_RIGHT_true(x)
#define IF_RIGHT_false(x) x

#define IF_LEFT(cond, x) IF_LEFT_##cond(x)
#define IF_RIGHT(cond, x) IF_RIGHT_##cond(x)
#define IF(cond, x, y) IF_LEFT(cond, x) IF_RIGHT(cond, y)

#define LINUX_SWITCH(x, other) LINUX_ONLY(x) NOT_LINUX(other)
#define MACOS_SWITCH(x, other) MACOS_ONLY(x) NOT_MACOS(other)
#define WINDOWS_SWITCH(x, other) WINDOWS_ONLY(x) NOT_WINDOWS(other)

#ifdef OPTIMIZED
#define OPTIMIZED_ONLY(x) x
#define NOT_OPTIMIZED(x)
#define IS_OPTIMIZED true
#else
#define OPTIMIZED_ONLY(x)
#define NOT_OPTIMIZED(x) x
#define IS_OPTIMIZED false
#endif

#if defined(__GNUC__)
#define __ALIGNED__(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#define __ALIGNED__(x) __declspec(align(x))
#elif defined(RC_INVOKED)
#define __ALIGNED__(x)
#else
#error
#endif /*_MSC_VER*/

#if defined(__GNUC__)
#define likely(cond) __builtin_expect(!!(cond), 1)
#define unlikely(cond) __builtin_expect(!!(cond), 0)
#else  // !__GNUC__
#define likely(cond) (cond)
#define unlikely(cond) (cond)
#endif  // !__GNUC__

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#define ALWAYSINLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#define ALWAYSINLINE __forceinline
#else  // !_MSC_VER
#define NOINLINE
#define ALWAYSINLINE
#endif  // !_MSC_VER

//! \endcond

#endif  // MACROS_HPP_
