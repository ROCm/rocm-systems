
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# executable RPATH
set(ROCPROFSYS_EXE_INSTALL_RPATH
    "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}"
)

# executables
add_subdirectory(rocprof-sys-avail)
add_subdirectory(rocprof-sys-causal)
add_subdirectory(rocprof-sys-sample)
add_subdirectory(rocprof-sys-instrument)
add_subdirectory(rocprof-sys-run)

# tests
if(ROCPROFSYS_BUILD_TESTING OR "$ENV{ROCPROFSYS_CI}" MATCHES "[1-9]+|ON|on|y|yes")
    add_subdirectory(tests)
    add_subdirectory(rocprof-sys-unit-tests)
endif()
