# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# See LICENSE.txt for license information

cmake_minimum_required(VERSION 3.21)

# Usable two ways: on its own, or via add_subdirectory() from a parent that has
# already declared a project with the HIP language enabled. No RCCL library
# target builds these sources -- the unit tests only consume the headers -- so
# the standalone path is the one that runs today, and it is the only one that
# defaults the build type.
if(NOT PROJECT_NAME)
  project(rccl_ep LANGUAGES CXX HIP)
  set(RCCL_EP_STANDALONE TRUE)
endif()

# These kernels are bandwidth-bound and unoptimised code does not merely run
# slower, it changes the conclusion: an -O0 build measures roughly half the
# dispatch bandwidth and a twentieth of the reduce. Default to Release so a
# plain configure cannot silently produce numbers worth nothing.
if(RCCL_EP_STANDALONE AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(RCCL_EP_GPU_TARGETS "gfx950" CACHE STRING
    "Semicolon-separated GPU architectures to compile for, e.g. gfx942;gfx950")
set(RCCL_EP_RCCL_ROOT "" CACHE PATH
    "Root of an EP-capable RCCL install, holding include/ and lib/")

# The window path needs symmetric memory, which needs cuMem, which RCCL only
# exposes on new enough ROCm; and the library must be built WITHOUT
# --rocshmem-gin, whose device symbols python3 cannot resolve at load time.
# Neither is discoverable from CMake, so RCCL is located explicitly rather than
# picked up from the system.
if(NOT RCCL_EP_RCCL_ROOT)
  message(FATAL_ERROR
    "RCCL_EP_RCCL_ROOT is not set. Point it at an RCCL built without "
    "--rocshmem-gin, e.g. -DRCCL_EP_RCCL_ROOT=/workspace/rccl-nogin")
endif()

find_library(RCCL_EP_RCCL_LIBRARY
  NAMES rccl
  PATHS "${RCCL_EP_RCCL_ROOT}/lib"
  NO_DEFAULT_PATH
  REQUIRED)
find_path(RCCL_EP_RCCL_INCLUDE_DIR
  NAMES nccl_device.h
  PATHS "${RCCL_EP_RCCL_ROOT}/include"
  NO_DEFAULT_PATH
  REQUIRED)

find_package(hip REQUIRED)

add_library(rccl_ep SHARED python/rccl_ep_capi.hip)

# No -fgpu-rdc. Without rocSHMEM there is no cross-TU device code, and a
# non-rdc object is self-contained, which is what lets dlopen load it into a
# Python process at all.
set_target_properties(rccl_ep PROPERTIES
  HIP_ARCHITECTURES "${RCCL_EP_GPU_TARGETS}"
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

# Sources include "device/..." and "include/..." relative to the module root.
target_include_directories(rccl_ep PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${RCCL_EP_RCCL_INCLUDE_DIR}")

target_link_libraries(rccl_ep PRIVATE "${RCCL_EP_RCCL_LIBRARY}" hip::device)

target_compile_options(rccl_ep PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-value>)

# Land the artifact beside the Python package: rccl_ep/__init__.py falls back to
# a co-located librccl_ep.so, so an installed build needs no RCCL_EP_LIB.
set_target_properties(rccl_ep PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/rccl_ep")

install(TARGETS rccl_ep LIBRARY DESTINATION "python/rccl_ep")

if(RCCL_EP_STANDALONE)
  message(STATUS "rccl_ep: build type    ${CMAKE_BUILD_TYPE}")
  message(STATUS "rccl_ep: targets       ${RCCL_EP_GPU_TARGETS}")
  message(STATUS "rccl_ep: RCCL library  ${RCCL_EP_RCCL_LIBRARY}")
endif()
