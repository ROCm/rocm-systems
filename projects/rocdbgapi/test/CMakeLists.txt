## Copyright (c) 2026 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

# Test suite entry point.  Sub-directories are added based on the
# AMD_DBGAPI_BUILD_UNIT_TESTS / AMD_DBGAPI_BUILD_FEATURE_TESTS options
# set in the parent CMakeLists.txt.
#
# CTest label conventions used across the tiers:
#
#   unit             - OS-agnostic, no kernel driver, no GPU
#   feature          - any feature-tier test
#   feature-driver   - requires a kernel driver (KFD or KMD)
#   feature-kfd      - Linux-only, requires /dev/kfd
#   feature-kmd      - Windows-only, requires dxg/libdxg
#   feature-gpu      - requires a real GPU
#   feature-gpu-<arch> - pinned to a specific GPU architecture
#   os-linux, os-windows
#                    - OS-specific tests
#
# Helper functions add_dbgapi_unit_test() / add_dbgapi_feature_test()
# (defined in cmake/AddDbgapiTest.cmake) apply these labels automatically.

# Make GoogleTest + GMock available to all test sub-directories.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/DbgapiGoogleTest.cmake)

# Bring in the add_dbgapi_*_test() helper functions.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/AddDbgapiTest.cmake)

if(AMD_DBGAPI_BUILD_UNIT_TESTS)
  add_subdirectory(unit)
endif()

if(AMD_DBGAPI_BUILD_FEATURE_TESTS)
  add_subdirectory(feature)
endif()
