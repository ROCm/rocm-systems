# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

set(ROCM_SDXI_VERSION_MAJOR 0)
set(ROCM_SDXI_VERSION_MINOR 1)
set(ROCM_SDXI_VERSION_PATCH 0)
set(ROCM_SDXI_VERSION
  "${ROCM_SDXI_VERSION_MAJOR}.${ROCM_SDXI_VERSION_MINOR}.${ROCM_SDXI_VERSION_PATCH}"
)

project(rocm-sdxi
    VERSION ${ROCM_SDXI_VERSION}
    DESCRIPTION "vfio-user emulator for the AMD SDXI data-mover"
    HOMEPAGE_URL "https://github.com/ROCm/rocm-systems"
    LANGUAGES C
)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

get_property(SDXI_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(PROJECT_IS_TOP_LEVEL AND NOT SDXI_IS_MULTI_CONFIG)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel" "None"
    )
endif()

include(SdxiGitInfo)
include(SdxiCompilerOptions)

# ---------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------
find_package(PkgConfig REQUIRED)
find_package(ZLIB REQUIRED)

pkg_check_modules(VFIO_USER vfio-user)
if(NOT VFIO_USER_FOUND)
    find_library(VFIO_USER_FALLBACK_LIB NAMES vfio-user
        DOC "Path to libvfio-user library")
    find_path(VFIO_USER_FALLBACK_INC NAMES vfio-user/libvfio-user.h
        DOC "Path to libvfio-user headers")
    if(NOT VFIO_USER_FALLBACK_LIB OR NOT VFIO_USER_FALLBACK_INC)
        message(FATAL_ERROR
          "libvfio-user not found via pkg-config, /usr/local, or /usr.\n"
          "Build it from https://github.com/nutanix/libvfio-user\n"
          "then set CMAKE_PREFIX_PATH or PKG_CONFIG_PATH."
        )
    endif()
    set(VFIO_USER_INCLUDE_DIRS "${VFIO_USER_FALLBACK_INC}")
    set(VFIO_USER_LINK_LIBRARIES "${VFIO_USER_FALLBACK_LIB}")
    message(STATUS "Found libvfio-user (fallback): ${VFIO_USER_FALLBACK_LIB}")
else()
    set(VFIO_USER_LINK_LIBRARIES "${VFIO_USER_LIBRARIES}")
    message(STATUS "Found libvfio-user via pkg-config: ${VFIO_USER_VERSION}")
endif()

# ---------------------------------------------------------------
# Sources
# ---------------------------------------------------------------
set(SDXI_SOURCES
    src/rocm_sdxi_server.c
    src/sdxi_log.c
    src/sdxi_engine.c
)

# ---------------------------------------------------------------
# Executable
# ---------------------------------------------------------------
add_executable(rocm-sdxi ${SDXI_SOURCES})

target_include_directories(rocm-sdxi PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_include_directories(rocm-sdxi SYSTEM PRIVATE
    ${VFIO_USER_INCLUDE_DIRS}
)

target_compile_definitions(rocm-sdxi PRIVATE
    GIT_SHA_FULL="${SDXI_GIT_SHA_FULL}"
    GIT_SHA_SHORT="${SDXI_GIT_SHA_SHORT}"
)

target_link_directories(rocm-sdxi PRIVATE
    ${VFIO_USER_LIBRARY_DIRS}
)

target_link_libraries(rocm-sdxi PRIVATE
    ${VFIO_USER_LINK_LIBRARIES}
    ZLIB::ZLIB
)

sdxi_set_compiler_flags(rocm-sdxi)

# ---------------------------------------------------------------
# Install
# ---------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS rocm-sdxi RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
if(PROJECT_IS_TOP_LEVEL)
    message(STATUS "")
    message(STATUS "=== rocm-sdxi Configuration ===")
    message(STATUS "  Version     : ${PROJECT_VERSION}")
    message(STATUS "  Git SHA     : ${SDXI_GIT_SHA_SHORT}")
    message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C compiler  : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
    if(VFIO_USER_FOUND)
        message(STATUS "  libvfio-user: ${VFIO_USER_VERSION} (pkg-config)")
    else()
        message(STATUS "  libvfio-user: ${VFIO_USER_FALLBACK_LIB} (fallback)")
    endif()
    message(STATUS "")
endif()
