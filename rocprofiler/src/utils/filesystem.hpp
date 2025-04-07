// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#if defined(__cpp_lib_filesystem)
#define ROCPROFILER_HAS_CPP_LIB_FILESYSTEM 1
#else
#if defined __has_include
#if __has_include(<filesystem>)
#define ROCPROFILER_HAS_CPP_LIB_FILESYSTEM 1
#endif
#endif
#endif

// include the correct filesystem header
#if defined(ROCPROFILER_HAS_CPP_LIB_FILESYSTEM) && ROCPROFILER_HAS_CPP_LIB_FILESYSTEM > 0
#include <filesystem>
#else
#include <experimental/filesystem>
#endif

// create a namespace alias
namespace rocprofiler {
namespace common {
#if defined(ROCPROFILER_HAS_CPP_LIB_FILESYSTEM) && ROCPROFILER_HAS_CPP_LIB_FILESYSTEM > 0
namespace filesystem = ::std::filesystem;  // NOLINT
#else
namespace filesystem = ::std::experimental::filesystem;  // NOLINT
#endif
}  // namespace common
}  // namespace rocprofiler