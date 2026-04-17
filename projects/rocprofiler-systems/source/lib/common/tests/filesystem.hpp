// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM)
#    if defined __has_include
#        if __has_include(<ghc/filesystem.hpp>)
#            define ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM 1
#        else
#            define ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM 0
#        endif
#    else
#        define ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM 0
#    endif
#endif

#if ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM == 0
#    if defined __has_include
#        if __has_include(<version>)
#            include <version>
#        endif
#    endif

#    if defined(__cpp_lib_filesystem)
#        define ROCPROFSYS_TESTS_HAS_CPP_LIB_FILESYSTEM 1
#    else
#        if defined __has_include
#            if __has_include(<filesystem>)
#                define ROCPROFSYS_TESTS_HAS_CPP_LIB_FILESYSTEM 1
#            endif
#        endif
#    endif
#endif

#if defined(ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM) &&                                  \
    ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM > 0
#    include <ghc/filesystem.hpp>
#elif defined(ROCPROFSYS_TESTS_HAS_CPP_LIB_FILESYSTEM) &&                                \
    ROCPROFSYS_TESTS_HAS_CPP_LIB_FILESYSTEM > 0
#    include <filesystem>
#else
#    include <experimental/filesystem>
#endif

namespace test_common
{
#if defined(ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM) &&                                  \
    ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM > 0
namespace fs = ::ghc::filesystem;  // NOLINT(misc-unused-alias-decls)
#elif defined(ROCPROFSYS_TESTS_HAS_CPP_LIB_FILESYSTEM) &&                                \
    ROCPROFSYS_TESTS_HAS_CPP_LIB_FILESYSTEM > 0
namespace fs = ::std::filesystem;  // NOLINT(misc-unused-alias-decls)
#else
namespace fs = ::std::experimental::filesystem;  // NOLINT(misc-unused-alias-decls)
#endif
}  // namespace test_common
