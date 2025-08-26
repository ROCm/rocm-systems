/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#ifndef VERSIONS_HPP_
#define VERSIONS_HPP_

#include "utils/macros.hpp"

#ifndef AMD_PLATFORM_NAME
#define AMD_PLATFORM_NAME "AMD Accelerated Parallel Processing"
#endif  // AMD_PLATFORM_NAME

#ifndef AMD_PLATFORM_BUILD_NUMBER
#define AMD_PLATFORM_BUILD_NUMBER 3581
#endif  // AMD_PLATFORM_BUILD_NUMBER

#ifndef AMD_PLATFORM_REVISION_NUMBER
#define AMD_PLATFORM_REVISION_NUMBER 0
#endif  // AMD_PLATFORM_REVISION_NUMBER

#ifndef AMD_PLATFORM_RELEASE_INFO
#define AMD_PLATFORM_RELEASE_INFO
#endif  // AMD_PLATFORM_RELEASE_INFO

#define AMD_BUILD_STRING                                                                           \
  XSTR(AMD_PLATFORM_BUILD_NUMBER)                                                                  \
  "." XSTR(AMD_PLATFORM_REVISION_NUMBER)

#ifndef AMD_PLATFORM_INFO
#define AMD_PLATFORM_INFO                                                                          \
  "AMD-APP" AMD_PLATFORM_RELEASE_INFO DEBUG_ONLY(                                                  \
      "." IF(IS_OPTIMIZED, "opt", "dbg")) " (" AMD_BUILD_STRING ")"
#endif  // ATI_PLATFORM_INFO

#endif  // VERSIONS_HPP_
