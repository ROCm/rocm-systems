# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

project(amdcuid_example)

set(CMAKE_CXX_STANDARD 14)

set(AMDCUID_EXAMPLE_EXECUTABLE amdcuid_example)
set(SRC_LIST main.cc)

add_executable(${AMDCUID_EXAMPLE_EXECUTABLE} ${SRC_LIST})

target_include_directories(
    ${AMDCUID_EXAMPLE_EXECUTABLE}
    PRIVATE ${CMAKE_SOURCE_DIR}/lib
)

# Link against the CUID library (public API only)
# The example now uses only the public API defined in amd_cuid.h
target_link_libraries(${AMDCUID_EXAMPLE_EXECUTABLE} amdcuid_static pthread)

# Install example source into documentation
install(
    FILES ${SRC_LIST}
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT ${EXAMPLE_COMPONENT}
)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  Finished Cmake amdcuid_example                   ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
