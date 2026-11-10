# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Fetch and build libvfio-user, exposed as libvfio-user::libvfio-user.
#
# Modelled on emulation/rocjitsu/cmake/rj_libvfio_user.cmake.  Upstream builds
# with meson and installs no pkg-config file, so rather than requiring meson
# and ninja we compile its handful of sources directly.  The source list mirrors
# lib/meson.build; keep the two in step when bumping the pinned tag.  json-c is
# the library's only dependency and is a CMake project, so it is fetched the
# same way.

include(FetchContent)
include(CheckCSourceCompiles)

# vfio-user is a pure userspace protocol: no VFIO kernel module is loaded and
# nothing here talks to the kernel.  The dependency is on the uapi headers,
# which supply the VFIO structs and constants the protocol reuses.
check_c_source_compiles(
    "#include <linux/vfio.h>
     int main(void) {
         return VFIO_DEVICE_FEATURE_MIG_DEVICE_STATE +
                VFIO_DEVICE_FEATURE_DMA_LOGGING_START;
     }"
    SDXI_HAVE_VFIO_MIGRATION_HEADERS
)
if(NOT SDXI_HAVE_VFIO_MIGRATION_HEADERS)
    message(
        FATAL_ERROR
        "rocm-sdxi requires Linux uapi headers with the VFIO migration and DMA logging definitions (Linux 6.1 or newer). Install matching linux-libc-dev / kernel-headers."
    )
endif()

# EXCLUDE_FROM_ALL keeps json-c's own install rules out of our install graph.
FetchContent_Declare(
    json-c
    GIT_REPOSITORY https://github.com/json-c/json-c.git
    GIT_TAG json-c-0.18-20240915
    EXCLUDE_FROM_ALL
)
set(BUILD_APPS OFF CACHE BOOL "" FORCE)
set(DISABLE_WERROR ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    libvfio-user
    GIT_REPOSITORY https://github.com/nutanix/libvfio-user.git
    GIT_TAG v0.8
)

set(_sdxi_saved_build_shared_libs ${BUILD_SHARED_LIBS})
set(_sdxi_saved_build_testing ${BUILD_TESTING})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json-c libvfio-user)
set(BUILD_SHARED_LIBS ${_sdxi_saved_build_shared_libs} CACHE BOOL "" FORCE)
set(BUILD_TESTING ${_sdxi_saved_build_testing} CACHE BOOL "" FORCE)
unset(_sdxi_saved_build_shared_libs)
unset(_sdxi_saved_build_testing)

add_library(
    libvfio-user
    STATIC
    ${libvfio-user_SOURCE_DIR}/lib/btree.c
    ${libvfio-user_SOURCE_DIR}/lib/dma.c
    ${libvfio-user_SOURCE_DIR}/lib/fd_cache.c
    ${libvfio-user_SOURCE_DIR}/lib/irq.c
    ${libvfio-user_SOURCE_DIR}/lib/libvfio-user.c
    ${libvfio-user_SOURCE_DIR}/lib/migration.c
    ${libvfio-user_SOURCE_DIR}/lib/pci.c
    ${libvfio-user_SOURCE_DIR}/lib/pci_caps.c
    ${libvfio-user_SOURCE_DIR}/lib/tran.c
    ${libvfio-user_SOURCE_DIR}/lib/tran_sock.c
)
# Sources include headers as <vfio-user/libvfio-user.h>; create a shim
# directory with a vfio-user/ subdirectory symlink to satisfy that spelling.
set(_vfio_shim_dir "${CMAKE_CURRENT_BINARY_DIR}/vfio-user-include")
file(MAKE_DIRECTORY "${_vfio_shim_dir}")
if(NOT EXISTS "${_vfio_shim_dir}/vfio-user")
    file(CREATE_LINK "${libvfio-user_SOURCE_DIR}/include" "${_vfio_shim_dir}/vfio-user" SYMBOLIC)
endif()
unset(_vfio_shim_dir)

target_include_directories(libvfio-user SYSTEM PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}/vfio-user-include"
)
target_include_directories(libvfio-user PRIVATE
    ${libvfio-user_SOURCE_DIR}/lib
)
target_compile_definitions(libvfio-user PRIVATE _GNU_SOURCE)
target_compile_options(libvfio-user PRIVATE -w)
target_link_libraries(libvfio-user PRIVATE json-c)
set_target_properties(libvfio-user PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(libvfio-user::libvfio-user ALIAS libvfio-user)
