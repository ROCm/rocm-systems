/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RCCL_FORMAT_H_
#define RCCL_FORMAT_H_

// Single entry point for RCCL string formatting: rccl::format.
//
// C++20 builds use std::format. C++17 builds, and C++20 builds whose standard
// library predates <format> (libstdc++ before GCC 13), fall back to fmtlib.
// Both back ends accept the same brace syntax and return std::string, so call
// sites are written once and need no conditionals.
//
// cmake/RcclFormat.cmake performs the equivalent check to decide whether fmt
// remains a build dependency; keep the two in agreement.

#include <string>

#define RCCL_FORMAT_HAS_STD 0

#if __cplusplus >= 202002L && defined(__has_include)
#if __has_include(<format>)
#include <format>
// Defined by <format> only once the implementation is actually complete.
#ifdef __cpp_lib_format
#undef RCCL_FORMAT_HAS_STD
#define RCCL_FORMAT_HAS_STD 1
#endif
#endif
#endif

#if RCCL_FORMAT_HAS_STD

namespace rccl {
using std::format;
} // namespace rccl

#else

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY 1
#endif
#include <fmt/format.h>

namespace rccl {
using fmt::format;
} // namespace rccl

#endif

#endif // RCCL_FORMAT_H_
