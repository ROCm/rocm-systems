# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# NCCL M2N CMake build. This file supports both a standalone checkout and
# add_subdirectory(contrib/nccl_m2n) from an NCCL build.

cmake_minimum_required(VERSION 3.20)

set(NCCL_M2N_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(NCCL_M2N_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(NCCL_M2N_INCLUDE_DIR "${NCCL_M2N_BINARY_DIR}/include")
set(NCCL_M2N_LIBRARY_DIR "${NCCL_M2N_BINARY_DIR}/lib")
set(NCCL_M2N_RUNTIME_DIR "${NCCL_M2N_BINARY_DIR}/bin")
set(NCCL_M2N_VERSION "0.2.0")
set(NCCL_M2N_STANDALONE OFF)

# CMake initializes its CUDA compiler and architecture flags in project(). Set
# the standalone default first so compiler detection never uses CMake's native
# architecture as an accidental build default. Parent builds retain the
# parent's architecture selection.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(NCCL_M2N_STANDALONE ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "80;90;100" CACHE STRING
        "CUDA architectures (sm_100 requires CUDA 12.4 or newer)")
  endif()
  project(nccl_m2n VERSION ${NCCL_M2N_VERSION} LANGUAGES CXX CUDA)
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
      "NCCL M2N currently supports Linux only (detected ${CMAKE_SYSTEM_NAME}).")
endif()

find_package(CUDAToolkit REQUIRED)
include(GNUInstallDirs)
if(NOT TARGET CUDA::cudart OR NOT TARGET CUDA::cuda_driver)
  message(FATAL_ERROR
      "The selected CUDA toolkit does not provide CUDA::cudart and "
      "CUDA::cuda_driver. Set CUDAToolkit_ROOT to a complete CUDA toolkit.")
endif()

# NCCL Device API headers include CCCL headers. Accept both toolkit layouts:
# CUDA 12 installs CCCL below include/cccl, while newer packages may expose
# cuda/std directly below the main include directory.
set(_NCCL_M2N_CCCL_INCLUDE_DIR "")
foreach(_cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
  if(EXISTS "${_cuda_include}/cuda/std/cstdint")
    set(_NCCL_M2N_CCCL_INCLUDE_DIR "${_cuda_include}")
    break()
  elseif(EXISTS "${_cuda_include}/cccl/cuda/std/cstdint")
    set(_NCCL_M2N_CCCL_INCLUDE_DIR "${_cuda_include}/cccl")
    break()
  endif()
endforeach()
if(NOT _NCCL_M2N_CCCL_INCLUDE_DIR)
  message(FATAL_ERROR
      "CCCL headers were not found in the selected CUDA toolkit. Expected "
      "cuda/std/cstdint under one of: ${CUDAToolkit_INCLUDE_DIRS}. Install "
      "the CUDA CCCL development package or select a complete toolkit with "
      "-DCUDAToolkit_ROOT=<path>.")
endif()

option(NCCL_M2N_BUILD_BENCH "Build nccl_m2n benchmarks (needs MPI)" OFF)
option(NCCL_M2N_BUILD_TESTS "Build nccl_m2n tests (needs GTest)" OFF)

# NCCL dependency. Parent mode consumes the existing target and discovers its
# generated include tree from the target's build interface. Standalone mode
# requires an NCCL_HOME with the Device API headers and shared library
# present.
set(_NCCL_M2N_NCCL_INCLUDES "")
if(TARGET nccl)
  set(_NCCL_M2N_NCCL_TARGET nccl)
  get_target_property(_nccl_interface_includes nccl INTERFACE_INCLUDE_DIRECTORIES)
  if(_nccl_interface_includes AND
     NOT _nccl_interface_includes MATCHES "-NOTFOUND$")
    foreach(_include IN LISTS _nccl_interface_includes)
      if(_include MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
        list(APPEND _NCCL_M2N_NCCL_INCLUDES "${CMAKE_MATCH_1}")
      elseif(IS_ABSOLUTE "${_include}")
        list(APPEND _NCCL_M2N_NCCL_INCLUDES "${_include}")
      endif()
    endforeach()
  endif()
  foreach(_include IN ITEMS
      "${CMAKE_BINARY_DIR}/include"
      "${CMAKE_SOURCE_DIR}/src/include")
    if(IS_DIRECTORY "${_include}")
      list(APPEND _NCCL_M2N_NCCL_INCLUDES "${_include}")
    endif()
  endforeach()

  set(_NCCL_M2N_NCCL_BUILD_INCLUDE "")
  foreach(_include IN LISTS _NCCL_M2N_NCCL_INCLUDES)
    if(EXISTS "${_include}/nccl_device.h")
      set(_NCCL_M2N_NCCL_BUILD_INCLUDE "${_include}")
      break()
    endif()
  endforeach()
  if(NOT _NCCL_M2N_NCCL_BUILD_INCLUDE)
    message(FATAL_ERROR
        "The parent nccl target does not expose a build include directory "
        "containing nccl_device.h. Add contrib/nccl_m2n after NCCL's src "
        "subdirectory creates the nccl target and generated headers.")
  endif()
  if(NOT IS_DIRECTORY "${_NCCL_M2N_NCCL_BUILD_INCLUDE}/nccl_device")
    message(FATAL_ERROR
        "The parent NCCL build include tree is incomplete: "
        "${_NCCL_M2N_NCCL_BUILD_INCLUDE}/nccl_device is missing. Configure "
        "NCCL so its Device API headers are generated before adding M2N.")
  endif()
  list(APPEND _NCCL_M2N_NCCL_INCLUDES
      "${_NCCL_M2N_NCCL_BUILD_INCLUDE}/nccl_device")
  if(TARGET nccl_header)
    set(_NCCL_M2N_NCCL_HEADER_TARGET nccl_header)
  endif()
else()
  set(NCCL_HOME "$ENV{NCCL_HOME}" CACHE PATH
      "Path to an NCCL install with Device API headers")
  if(NOT NCCL_HOME)
    message(FATAL_ERROR
        "NCCL_HOME is required for a standalone build. Pass "
        "-DNCCL_HOME=/path/to/nccl.")
  endif()
  if(NOT IS_DIRECTORY "${NCCL_HOME}")
    message(FATAL_ERROR "NCCL_HOME=${NCCL_HOME} is not a directory.")
  endif()
  foreach(_required_header IN ITEMS nccl.h nccl_device.h)
    if(NOT EXISTS "${NCCL_HOME}/include/${_required_header}")
      message(FATAL_ERROR
          "NCCL_HOME=${NCCL_HOME} is missing include/${_required_header}.")
    endif()
  endforeach()
  if(NOT IS_DIRECTORY "${NCCL_HOME}/include/nccl_device")
    message(FATAL_ERROR
        "NCCL_HOME=${NCCL_HOME} is missing include/nccl_device/.")
  endif()
  find_library(_NCCL_M2N_NCCL_LIBRARY
      NAMES nccl
      HINTS "${NCCL_HOME}/lib" "${NCCL_HOME}/lib64"
      NO_DEFAULT_PATH)
  if(NOT _NCCL_M2N_NCCL_LIBRARY)
    # NCCL_HOME may have only a versioned SONAME; find_library() needs
    # the unversioned symlink. Fall back to the versioned file.
    file(GLOB _NCCL_M2N_NCCL_LIBRARY_CANDIDATES
        "${NCCL_HOME}/lib/libnccl.so.*"
        "${NCCL_HOME}/lib64/libnccl.so.*")
    if(_NCCL_M2N_NCCL_LIBRARY_CANDIDATES)
      list(GET _NCCL_M2N_NCCL_LIBRARY_CANDIDATES 0 _NCCL_M2N_NCCL_LIBRARY)
    endif()
  endif()
  if(NOT _NCCL_M2N_NCCL_LIBRARY)
    message(FATAL_ERROR
        "NCCL_HOME=${NCCL_HOME} does not contain libnccl.so or a versioned "
        "libnccl.so.<N> under lib/ or lib64/.")
  endif()

  if(NOT TARGET NCCL::nccl)
    add_library(NCCL::nccl SHARED IMPORTED GLOBAL)
    set_target_properties(NCCL::nccl PROPERTIES
        IMPORTED_LOCATION "${_NCCL_M2N_NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_HOME}/include")
  endif()
  set(_NCCL_M2N_NCCL_TARGET NCCL::nccl)
  set(_NCCL_M2N_NCCL_INCLUDES
      "${NCCL_HOME}/include"
      "${NCCL_HOME}/include/nccl_device")
endif()
list(APPEND _NCCL_M2N_NCCL_INCLUDES "${_NCCL_M2N_CCCL_INCLUDE_DIR}")
list(REMOVE_DUPLICATES _NCCL_M2N_NCCL_INCLUDES)

# Library source manifest. The private CI check mechanically compares these
# lists with Makefile.common's LIB_CC_SRCS and LIB_CU_SRCS.
set(NCCL_M2N_HOST_SRC
    src/m2n_group.cc
    src/reshard_call_setup.cc
    src/reshard_cache.cc
    src/m2n_config.cc
    src/m2n_init.cc
    src/reshard_loadbalance.cc
    src/reshard_mesh.cc
    src/reshard_prepare.cc
    src/reshard_split_prepare.cc
    src/pack_staging.cc
    src/staging_buffer.cc
    src/staging_profile.cc
    src/staging_prepare.cc
)

set(NCCL_M2N_DEVICE_SRC
    src/reshard_split_comm.cu
    src/reshard_split_user_window.cu
    src/reshard_user_window.cu
    src/reshard_staging.cu
    src/staging_kernel.cu
)

set(NCCL_M2N_SRC ${NCCL_M2N_HOST_SRC} ${NCCL_M2N_DEVICE_SRC})
set(NCCL_M2N_SRC_FULL_PATHS "")
foreach(_source IN LISTS NCCL_M2N_SRC)
  list(APPEND NCCL_M2N_SRC_FULL_PATHS "${NCCL_M2N_SOURCE_DIR}/${_source}")
endforeach()

add_library(nccl_m2n_shared SHARED ${NCCL_M2N_SRC})
add_library(nccl_m2n_static STATIC ${NCCL_M2N_SRC})
add_library(NCCL::m2n ALIAS nccl_m2n_shared)
add_library(NCCL::m2n_static ALIAS nccl_m2n_static)
add_custom_target(nccl_m2n_lib DEPENDS nccl_m2n_shared nccl_m2n_static)
set(NCCL_M2N_TARGETS nccl_m2n_shared nccl_m2n_static)

set_target_properties(${NCCL_M2N_TARGETS} PROPERTIES
    OUTPUT_NAME "nccl_m2n"
    ARCHIVE_OUTPUT_DIRECTORY "${NCCL_M2N_LIBRARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${NCCL_M2N_LIBRARY_DIR}"
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
set_target_properties(nccl_m2n_shared PROPERTIES
    EXPORT_NAME m2n
    BUILD_RPATH "${NCCL_M2N_LIBRARY_DIR}"
    INSTALL_RPATH "$ORIGIN")
set_target_properties(nccl_m2n_static PROPERTIES EXPORT_NAME m2n_static)

foreach(_target IN LISTS NCCL_M2N_TARGETS)
  target_include_directories(${_target}
      PUBLIC
        $<BUILD_INTERFACE:${NCCL_M2N_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      PRIVATE
        "${NCCL_M2N_SOURCE_DIR}/src")
  target_include_directories(${_target} SYSTEM PRIVATE
      "${NCCL_M2N_SOURCE_DIR}/third_party"
      ${_NCCL_M2N_NCCL_INCLUDES}
      ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(${_target} PRIVATE NCCL_OS_LINUX)
  target_compile_options(${_target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
  if(_NCCL_M2N_NCCL_HEADER_TARGET)
    add_dependencies(${_target} ${_NCCL_M2N_NCCL_HEADER_TARGET})
  endif()
endforeach()

# The public header exposes NCCL and CUDA types. Those include/link
# requirements are therefore transitive for both variants. cuda_driver is an
# implementation dependency of the shared library and a transitive dependency
# of the static archive.
target_link_libraries(nccl_m2n_shared
    PUBLIC
      "$<BUILD_INTERFACE:${_NCCL_M2N_NCCL_TARGET}>"
      "$<INSTALL_INTERFACE:NCCL::nccl>"
      CUDA::cudart
    PRIVATE CUDA::cuda_driver)
target_link_libraries(nccl_m2n_static
    PUBLIC
      "$<BUILD_INTERFACE:${_NCCL_M2N_NCCL_TARGET}>"
      "$<INSTALL_INTERFACE:NCCL::nccl>"
      CUDA::cudart
      CUDA::cuda_driver)

set(NCCL_M2N_PUBLIC_HEADERS src/nccl_m2n.h)
file(MAKE_DIRECTORY "${NCCL_M2N_INCLUDE_DIR}")
foreach(_header IN LISTS NCCL_M2N_PUBLIC_HEADERS)
  get_filename_component(_header_name "${_header}" NAME)
  configure_file(
      "${NCCL_M2N_SOURCE_DIR}/${_header}"
      "${NCCL_M2N_INCLUDE_DIR}/${_header_name}"
      COPYONLY)
endforeach()

# Build-tree and installed find_package(NCCLM2N CONFIG) support. Parent builds
# consume the in-tree targets directly; exporting them would also try to export
# the parent-owned nccl target.
if(NCCL_M2N_STANDALONE)
  include(CMakePackageConfigHelpers)
  set(NCCL_M2N_PACKAGE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/NCCLM2N")
  set(NCCL_M2N_BUILD_PACKAGE_DIR "${NCCL_M2N_BINARY_DIR}/cmake")
  file(MAKE_DIRECTORY "${NCCL_M2N_BUILD_PACKAGE_DIR}/install")

  install(TARGETS nccl_m2n_shared nccl_m2n_static
      EXPORT NCCLM2NTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES ${NCCL_M2N_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT NCCLM2NTargets
      FILE NCCLM2NTargets.cmake
      NAMESPACE NCCL::
      DESTINATION "${NCCL_M2N_PACKAGE_INSTALL_DIR}")
  export(EXPORT NCCLM2NTargets
      FILE "${NCCL_M2N_BUILD_PACKAGE_DIR}/NCCLM2NTargets.cmake"
      NAMESPACE NCCL::)

  set(NCCL_M2N_CONFIG_NCCL_HINT "${NCCL_HOME}")
  configure_package_config_file(
      "${NCCL_M2N_SOURCE_DIR}/cmake/NCCLM2NConfig.cmake.in"
      "${NCCL_M2N_BUILD_PACKAGE_DIR}/NCCLM2NConfig.cmake"
      INSTALL_DESTINATION ".")
  set(NCCL_M2N_CONFIG_NCCL_HINT "")
  configure_package_config_file(
      "${NCCL_M2N_SOURCE_DIR}/cmake/NCCLM2NConfig.cmake.in"
      "${NCCL_M2N_BUILD_PACKAGE_DIR}/install/NCCLM2NConfig.cmake"
      INSTALL_DESTINATION "${NCCL_M2N_PACKAGE_INSTALL_DIR}")
  write_basic_package_version_file(
      "${NCCL_M2N_BUILD_PACKAGE_DIR}/NCCLM2NConfigVersion.cmake"
      VERSION "${NCCL_M2N_VERSION}"
      COMPATIBILITY SameMajorVersion)
  install(FILES
      "${NCCL_M2N_BUILD_PACKAGE_DIR}/install/NCCLM2NConfig.cmake"
      "${NCCL_M2N_BUILD_PACKAGE_DIR}/NCCLM2NConfigVersion.cmake"
      DESTINATION "${NCCL_M2N_PACKAGE_INSTALL_DIR}")
else()
  install(TARGETS nccl_m2n_shared nccl_m2n_static
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES ${NCCL_M2N_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(NCCL_M2N_BUILD_BENCH)
  add_subdirectory(benchmarks)
endif()
if(NCCL_M2N_BUILD_TESTS)
  if(NCCL_M2N_STANDALONE)
    include(CTest)
  elseif(NOT BUILD_TESTING)
    message(WARNING
        "NCCL_M2N_BUILD_TESTS=ON but parent testing is disabled. Call "
        "include(CTest) in the parent top-level for root ctest discovery.")
  endif()
  add_subdirectory(tests)
endif()
