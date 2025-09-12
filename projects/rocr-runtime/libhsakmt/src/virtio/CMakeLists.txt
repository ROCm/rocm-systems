
# Copyright 2025 Advanced Micro Devices, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required ( VERSION 3.7 )

set (CMAKE_VERBOSE_MAKEFILE ON)

set ( HSAKMT_VIRTIO "hsakmt_virtio" )
set ( HSAKMT_VIRTIO_TARGET "${HSAKMT_VIRTIO}" )

project ( ${HSAKMT_VIRTIO_TARGET} VERSION 1.0)

## Compiler flags
set ( HSAKMT_VIRTIO_C_FLAGS -fPIC -W -Wall -Wextra -Wno-unused-parameter -Wformat-security -Wswitch-default -Wundef -Wshadow -Wpointer-arith -Wbad-function-cast -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wredundant-decls -Wunreachable-code -std=gnu99 -fvisibility=hidden )
if ( CMAKE_COMPILER_IS_GNUCC )
    set ( HSAKMT_VIRTIO_C_FLAGS "${HSAKMT_VIRTIO_C_FLAGS}" -Wlogical-op )
endif ()

set ( HSAKMT_VIRTIO_LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/libhsakmt_virtio.ver" )

set ( HSAKMT_VIRTIO_LINK_FLAGS "-Wl,--enable-new-dtags -Wl,--version-script=${HSAKMT_VIRTIO_LINKER_SCRIPT} -Wl,-z,nodelete")

if ( "${CMAKE_BUILD_TYPE}" STREQUAL Release )
    set ( HSAKMT_VIRTIO_C_FLAGS "${HSAKMT_VIRTIO_C_FLAGS}" -O2 )
else ()
    set ( HSAKMT_VIRTIO_C_FLAGS "${HSAKMT_VIRTIO_C_FLAGS}" -g )
endif ()

set ( HSAKMT_VIRTIO_SRC "virtio_gpu.c"
                        "hsakmt_virtio_vm.c"
                        "hsakmt_virtio_device.c"
                        "hsakmt_virtio_memory.c"
                        "hsakmt_virtio_amdgpu.c"
                        "hsakmt_virtio_events.c"
                        "hsakmt_virtio_queues.c"
                        "hsakmt_virtio_topology.c"
                        "hsakmt_virtio_openclose.c"
                        "../rbtree.c" )

add_library ( ${HSAKMT_VIRTIO_TARGET} STATIC ${HSAKMT_VIRTIO_SRC} )

target_sources ( ${HSAKMT_VIRTIO_TARGET} PRIVATE ${HSAKMT_VIRTIO_SRC} )

target_compile_options ( ${HSAKMT_VIRTIO_TARGET} PRIVATE ${HSAKMT_VIRTIO_C_FLAGS} )

target_include_directories ( ${HSAKMT_VIRTIO_TARGET}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/virtio
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/linux )

set_property(TARGET ${HSAKMT_VIRTIO_TARGET} PROPERTY LINK_FLAGS ${HSAKMT_VIRTIO_LINK_FLAGS})

find_package ( PkgConfig )

## If environment variable DRM_DIR is set, the script
## will pick up the corresponding libraries from that path.
list ( PREPEND CMAKE_PREFIX_PATH "${DRM_DIR}" )

pkg_check_modules ( DRM REQUIRED IMPORTED_TARGET libdrm )
pkg_check_modules ( DRM_AMDGPU REQUIRED IMPORTED_TARGET libdrm_amdgpu )
target_include_directories ( ${HSAKMT_VIRTIO_TARGET} PRIVATE ${DRM_AMDGPU_INCLUDE_DIRS} )
target_include_directories ( ${HSAKMT_VIRTIO_TARGET} PRIVATE ${DRM_INCLUDE_DIRS} )

target_link_libraries ( ${HSAKMT_VIRTIO_TARGET}
    PRIVATE ${DRM_LDFLAGS} ${DRM_AMDGPU_LDFLAGS} pthread rt c ${CMAKE_DL_LIBS} )
