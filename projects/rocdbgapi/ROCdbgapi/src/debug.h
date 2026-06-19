/* Copyright (c) 2019-2024 Advanced Micro Devices, Inc.

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

#ifndef AMD_DBGAPI_DEBUG_H
#define AMD_DBGAPI_DEBUG_H 1

#include "amd-dbgapi.h"

#include <stdexcept>
#include <string>
#include <utility>

#if defined(NDEBUG)

#define dbgapi_assert(expr) ((void)0)
#define dbgapi_assert_not_reached(message) __builtin_trap ()

#else /* !defined (NDEBUG) */

#define dbgapi_assert(expr)                                                   \
  ((void)((expr) ? 0 : (dbgapi_assert_fail (#expr, __FILE__, __LINE__), 0)))

#define dbgapi_assert_not_reached(message)                                    \
  amd::dbgapi::fatal_error ("%s:%d: Should not reach here: " message,         \
                            __FILE__, __LINE__)
#endif /* !defined(NDEBUG) */

#define dbgapi_assert_fail(assertion, file, line)                             \
  amd::dbgapi::fatal_error ("%s:%d: Assertion `%s' failed.", file, line,      \
                            assertion)

namespace amd::dbgapi
{

extern void warning (const char *format, ...)
#if defined(__GNUC__)
  __attribute__ ((format (printf, 1, 2)))
#endif /* defined (__GNUC__) */
  ;

extern void fatal_error [[noreturn]] (const char *format, ...)
#if defined(__GNUC__)
__attribute__ ((format (printf, 1, 2)))
#endif /* defined (__GNUC__) */
;

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_DEBUG_H */
