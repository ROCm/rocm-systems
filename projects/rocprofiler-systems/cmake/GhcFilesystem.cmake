# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

include_guard(GLOBAL)

message(STATUS "Setting up ghc::filesystem for tests")

include(FetchContent)

rocprofiler_systems_checkout_git_submodule(
    RELATIVE_PATH external/filesystem
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    TEST_FILE include/ghc/filesystem.hpp
    REPO_URL https://github.com/gulrak/filesystem.git
    REPO_BRANCH "v1.5.14"
)

# Configure ghc/filesystem options before adding it
set(GHC_FILESYSTEM_BUILD_TESTING OFF CACHE BOOL "Disable ghc filesystem tests" FORCE)
set(GHC_FILESYSTEM_BUILD_EXAMPLES OFF CACHE BOOL "Disable ghc filesystem examples" FORCE)
set(GHC_FILESYSTEM_WITH_INSTALL OFF CACHE BOOL "Disable ghc filesystem install" FORCE)

# Declare ghc/filesystem from the submodule
FetchContent_Declare(ghc_filesystem SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/filesystem)

# Make ghc/filesystem available
FetchContent_MakeAvailable(ghc_filesystem)

# Create interface library that wraps ghc::filesystem target
add_library(rocprofiler-systems-ghc-filesystem INTERFACE)

target_link_libraries(rocprofiler-systems-ghc-filesystem INTERFACE ghc_filesystem)

target_compile_definitions(
    rocprofiler-systems-ghc-filesystem
    INTERFACE ROCPROFSYS_TESTS_HAS_GHC_LIB_FILESYSTEM=1
)

message(STATUS "ghc::filesystem configured successfully using FetchContent")
