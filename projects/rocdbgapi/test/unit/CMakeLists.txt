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

# Unit tests for rocdbgapi.
#
# Tests added here must NOT depend on /dev/kfd, dxg/libdxg, or a real
# GPU.  Anything that touches the kernel driver belongs under
# ../feature/.

# Toolchain smoke test.  Confirms that GoogleTest is wired up on the
# host platform.  Uses the add_dbgapi_unit_test() helper which applies
# the standard labels ("dbgapi", "unit", OS label), C++17 settings,
# GTest/GMock linkage, and CTest working directory.
add_dbgapi_unit_test(dbgapi_smoke_test SOURCES smoke_test.cpp)
