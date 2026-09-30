# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Fetch and build libvfio-user, exposed as libvfio-user::libvfio-user.
#
# Upstream builds with meson and installs no pkg-config file, so rather than
# requiring meson and ninja we compile its handful of sources directly. The
# source list mirrors lib/meson.build; keep the two in step when bumping the
# pinned tag. json-c is the library's only dependency and is a CMake project, so
# it is fetched the same way.
#
# Everything lands under FETCHCONTENT_BASE_DIR (third_party/ by default),
# matching how googletest and flatbuffers are handled.

include(FetchContent)
include(CheckCSourceCompiles)

# The project as a whole supports CMake 3.22, but keeping json-c out of our
# install graph relies on EXCLUDE_FROM_ALL in FetchContent_Declare, which landed
# in 3.28. Rather than let an older CMake silently produce a package that ships
# an internal dependency, this optional feature states its own floor.
if(CMAKE_VERSION VERSION_LESS 3.28)
    message(
        FATAL_ERROR
        "ROCJITSU_ENABLE_VFIO=ON requires CMake 3.28 or newer (found ${CMAKE_VERSION}); the rest of rocjitsu still builds with 3.22."
    )
endif()

# vfio-user is a pure userspace protocol: no VFIO kernel module is loaded and
# nothing here talks to the kernel. The dependency is on the uapi *headers*,
# which supply the VFIO structs and constants the protocol reuses. Check for the
# newest of those rather than testing a kernel version, since distributions
# backport freely and the header is what actually has to compile.
check_c_source_compiles(
    "#include <linux/vfio.h>
     int main(void) {
         return VFIO_DEVICE_FEATURE_MIG_DEVICE_STATE +
                VFIO_DEVICE_FEATURE_DMA_LOGGING_START;
     }"
    RJ_HAVE_VFIO_MIGRATION_HEADERS
)
if(NOT RJ_HAVE_VFIO_MIGRATION_HEADERS)
    message(
        FATAL_ERROR
        "ROCJITSU_ENABLE_VFIO=ON requires Linux uapi headers with the VFIO migration and DMA logging definitions (Linux 6.1 or newer). Install matching linux-libc-dev / kernel-headers."
    )
endif()

# EXCLUDE_FROM_ALL keeps json-c's own install rules out of our install graph: it
# is linked statically into the transport, so shipping its library, headers,
# CMake exports and pkg-config file with rocjitsu would export an internal
# dependency as though it were part of our interface.
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

# json-c reads two variables that belong to the enclosing project:
# BUILD_SHARED_LIBS (a shared build would have to be shipped alongside anything
# linking it) and BUILD_TESTING (which gates rocjitsu's own test suite). Both are
# overridden for the subproject only and restored immediately afterwards.
set(_rj_saved_build_shared_libs ${BUILD_SHARED_LIBS})
set(_rj_saved_build_testing ${BUILD_TESTING})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
# libvfio-user ships no CMakeLists.txt, so MakeAvailable only fetches it; the
# targets below are ours.
FetchContent_MakeAvailable(json-c libvfio-user)
set(BUILD_SHARED_LIBS ${_rj_saved_build_shared_libs} CACHE BOOL "" FORCE)
set(BUILD_TESTING ${_rj_saved_build_testing} CACHE BOOL "" FORCE)
unset(_rj_saved_build_shared_libs)
unset(_rj_saved_build_testing)

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
# SYSTEM: these headers use C11 and C99 constructs that our -Werror C++ build
# rejects, and they are not ours to fix.
target_include_directories(
    libvfio-user
    SYSTEM
    PUBLIC ${libvfio-user_SOURCE_DIR}/include
)
target_include_directories(libvfio-user PRIVATE ${libvfio-user_SOURCE_DIR}/lib)
target_compile_definitions(libvfio-user PRIVATE _GNU_SOURCE)
# Third-party C built with our warning settings would fail the -Werror build,
# and its warnings are not ours to fix.
target_compile_options(libvfio-user PRIVATE -w)
target_link_libraries(libvfio-user PRIVATE json-c)
set_target_properties(libvfio-user PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(libvfio-user::libvfio-user ALIAS libvfio-user)
