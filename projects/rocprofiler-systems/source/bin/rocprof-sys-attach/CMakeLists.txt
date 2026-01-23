# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

set(TARGET rocprofiler-systems-attach)

find_package(rocprofiler-sdk-rocattach QUIET) #QUIET to avoid error if not found

if(rocprofiler-sdk-rocattach_FOUND)
    add_executable(${TARGET} ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-attach.cpp)

    # Note: Hardcoeded paths are just a workaround to get the rocattach library.
    # The proper way would be to use find_package(rocprofiler-sdk-rocattach REQUIRED)
    # and use the rocprofiler-sdk-rocattach::rocprofiler-sdk-rocattach target.
    # In that way we can resolve paths for library and headers only with linking to the rocprofiler-sdk-rocattach library.

    # find_package(rocprofiler-sdk-rocattach REQUIRED)
    # message(STATUS "rocprofiler-sdk-rocattach_FOUND: ${rocprofiler-sdk-rocattach_FOUND}, rocprofiler-sdk-rocattach_INCLUDE_DIRS: ${rocprofiler-sdk-rocattach_INCLUDE_DIRS}, rocprofiler-sdk-rocattach_LIBRARIES: ${rocprofiler-sdk-rocattach_LIBRARIES}")

    target_include_directories(${TARGET} PRIVATE /opt/rocm/include/)

    target_link_directories(${TARGET} PRIVATE /opt/rocm/lib/)
    target_link_libraries(${TARGET} PRIVATE rocprofiler-sdk-rocattach)

    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
            INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
            OUTPUT_NAME ${BINARY_NAME_PREFIX}-attach
    )

    rocprofiler_systems_strip_target(${TARGET})

    install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
else()
    message(
        INFO
        "rocprofiler-sdk-rocattach not found, skipping rocprof-sys-attach target"
    )
endif()
