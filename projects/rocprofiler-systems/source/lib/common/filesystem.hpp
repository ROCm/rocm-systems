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

#if !defined(ROCPROFSYS_HAS_GHC_LIB_FILESYSTEM)
#    if defined __has_include
#        if __has_include(<ghc/filesystem.hpp>)
#            define ROCPROFSYS_HAS_GHC_LIB_FILESYSTEM 1
#        else
#            define ROCPROFSYS_HAS_GHC_LIB_FILESYSTEM 0
#        endif
#    else
#        define ROCPROFSYS_HAS_GHC_LIB_FILESYSTEM 0
#    endif
#endif

#if ROCPROFSYS_HAS_GHC_LIB_FILESYSTEM == 0
#    if defined __has_include
#        if __has_include(<version>)
#            include <version>
#        endif
#    endif

#    if defined(__cpp_lib_filesystem)
#        define ROCPROFSYS_HAS_CPP_LIB_FILESYSTEM 1
#    else
#        if defined __has_include
#            if __has_include(<filesystem>)
#                define ROCPROFSYS_HAS_CPP_LIB_FILESYSTEM 1
#            endif
#        endif
#    endif
#endif

#if defined(ROCPROFSYS_HAS_GHC_LIB_FILESYSTEM) && ROCPROFSYS_HAS_GHC_LIB_FILESYSTEM > 0
#    include <ghc/filesystem.hpp>
#elif defined(ROCPROFSYS_HAS_CPP_LIB_FILESYSTEM) && ROCPROFSYS_HAS_CPP_LIB_FILESYSTEM > 0
#    include <filesystem>
#else
#    include <experimental/filesystem>
#endif

#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>

namespace rocprofsys
{
namespace common
{
#if defined(ROCPROFSYS_HAS_GHC_LIB_FILESYSTEM) && ROCPROFSYS_HAS_GHC_LIB_FILESYSTEM > 0
namespace fs = ::ghc::filesystem;  // NOLINT(misc-unused-alias-decls)
#elif defined(ROCPROFSYS_HAS_CPP_LIB_FILESYSTEM) && ROCPROFSYS_HAS_CPP_LIB_FILESYSTEM > 0
namespace fs = std::filesystem;  // NOLINT(misc-unused-alias-decls)
#else
namespace fs = std::experimental::filesystem;  // NOLINT(misc-unused-alias-decls)
#endif

// Utility functions to replace tim::filepath functionality
namespace filepath
{
// Get directory name (equivalent to dirname)
inline std::string
dirname(std::string_view path)
{
    return fs::path(path).parent_path().string();
}

// Get base filename (equivalent to basename)
inline const char*
basename(std::string_view path)
{
    static thread_local std::string _result;
    _result = fs::path(path).filename().string();
    return _result.c_str();
}

// Check if path exists
inline bool
exists(const std::string& path)
{
    std::error_code ec;
    return fs::exists(path, ec);
}

// Check if directory exists
inline bool
direxists(const std::string& path)
{
    std::error_code ec;
    return fs::is_directory(path, ec);
}

// Create directory (and parent directories if needed)
inline bool
makedir(const std::string& path)
{
    std::error_code ec;
    return fs::create_directories(path, ec);
}

// Get canonical path (equivalent to realpath)
inline std::string
realpath(const std::string& path, std::nullptr_t = nullptr, bool weak = false)
{
    std::error_code ec;
    auto            result = weak ? fs::weakly_canonical(path, ec) : fs::canonical(path, ec);
    if(ec) return path;  // Return original path on error
    return result.string();
}

// Read symbolic link
inline std::string
readlink(const std::string& path)
{
    std::error_code ec;
    auto            result = fs::read_symlink(path, ec);
    if(ec) return path;  // Return original path on error
    return result.string();
}

// Open file stream with directory creation
template <typename StreamT>
inline bool
open(StreamT& stream, const std::string& path)
{
    // Create parent directory if needed
    auto parent = fs::path(path).parent_path();
    if(!parent.empty())
    {
        std::error_code ec;
        fs::create_directories(parent, ec);
    }

    // Open the stream
    stream.open(path);
    return stream.is_open();
}

}  // namespace filepath
}  // namespace common
}  // namespace rocprofsys
