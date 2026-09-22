// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#if defined __has_include
#    if __has_include(<version>)
#        include <version>
#    endif
#endif

#if defined(__cpp_lib_filesystem)
#    define ROCPROFILER_REGISTER_HAS_CPP_LIB_FILESYSTEM 1
#else
#    if defined __has_include
#        if __has_include(<filesystem>)
#            include <filesystem>
#        endif
#    endif
#endif

#if defined(ROCPROFILER_REGISTER_HAS_CPP_LIB_FILESYSTEM) &&                              \
    ROCPROFILER_REGISTER_HAS_CPP_LIB_FILESYSTEM > 0
#    include <filesystem>
namespace fs = ::std::filesystem;  // NOLINT
#else
#    include <experimental/filesystem>
namespace fs = ::std::experimental::filesystem;  // NOLINT
#endif
