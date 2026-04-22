# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Allow ROCM_INSTALL_PATH to be supplied via the environment (e.g. by a
# developer setup script) in addition to -DROCM_INSTALL_PATH on the CMake
# command line.
if(NOT ROCM_INSTALL_PATH AND DEFINED ENV{ROCM_INSTALL_PATH})
  set(ROCM_INSTALL_PATH "$ENV{ROCM_INSTALL_PATH}")
endif()

if (AMD_COMPUTE_WIN)
  find_path(AMD_HSA_INCLUDE_DIR hsa.h
    PATHS
      ${ROCCLR_SRC_DIR}/../../rocr-runtime/runtime/hsa-runtime/inc
      ${ROCCLR_SRC_DIR}/../../rocr-runtime/runtime/hsa-runtime
      ${ROCM_INSTALL_PATH}
      /opt/rocm
      ${CMAKE_CURRENT_BINARY_DIR}
    PATHS
      ${CMAKE_CURRENT_BINARY_DIR}/..
      ${CMAKE_CURRENT_BINARY_DIR}/../..
      ${CMAKE_CURRENT_BINARY_DIR}/../../rocr
    PATH_SUFFIXES
      include
      include/hsa
      inc
    NO_DEFAULT_PATH)
  message("Roc CLR: " ${ROCCLR_SRC_DIR} "; HSA headers:" ${AMD_HSA_INCLUDE_DIR})
  target_include_directories(rocclr PUBLIC ${AMD_HSA_INCLUDE_DIR})
  target_include_directories(rocclr PUBLIC ${AMD_HSA_INCLUDE_DIR}/..)
  # Build hsa-runtime64 as a subdirectory
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
  set(ROCM_PATCH_VERSION 99999 CACHE STRING "")
  add_subdirectory(${ROCCLR_SRC_DIR}/../../rocr-runtime hsa-runtime64)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
  # Create alias target to match find_package() convention
  # The actual static library target is hsa-runtime64_static, with hsa-runtime64 being an INTERFACE wrapper
  if(TARGET hsa-runtime64_static AND NOT TARGET hsa-runtime64::hsa-runtime64_static)
    add_library(hsa-runtime64::hsa-runtime64_static ALIAS hsa-runtime64_static)
  endif()
  # Link the static library (use the INTERFACE wrapper which applies --whole-archive correctly)
  target_link_libraries(rocclr PUBLIC hsa-runtime64)
  if (NOT ROCCLR_ENABLE_PAL)
    find_package(AMD_HSA_LOADER)
    target_link_libraries(rocclr PUBLIC oclelf)
  endif()
  target_compile_definitions(rocclr PUBLIC ROCR_STATIC_OPEN)
else()
  if(UNIX)
    find_package(hsa-runtime64 1.11 REQUIRED CONFIG
      PATHS
        ${ROCM_INSTALL_PATH}
        /opt/rocm/
      PATH_SUFFIXES
        cmake/hsa-runtime64
        lib/cmake/hsa-runtime64
        lib64/cmake/hsa-runtime64)

    # Ensure HSA headers from the package we just resolved are searched
    # BEFORE any other -isystem path contributed by transitively-linked
    # dependencies (e.g. amd_comgr / rocprofiler-register at /opt/rocm),
    # otherwise /opt/rocm/include/hsa/hsa.h will shadow the one shipped
    # alongside hsa-runtime64.
    get_target_property(_hsa_interface_includes
      hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)
    if(_hsa_interface_includes)
      target_include_directories(rocclr SYSTEM BEFORE PUBLIC
        ${_hsa_interface_includes})
    endif()
    unset(_hsa_interface_includes)
  else()
    find_package(hsa-runtime64 1.11 REQUIRED CONFIG
      PATHS
        ${ROCM_INSTALL_PATH}
        /opt/rocm/
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_INSTALL_PREFIX}
        ${CMAKE_INSTALL_PREFIX}/..
      PATH_SUFFIXES
        rocr/lib/cmake/hsa-runtime64
        rocr/runtime/hsa-runtime
        cmake/hsa-runtime64
        lib/cmake/hsa-runtime64
        lib64/cmake/hsa-runtime64)

    # note: Temporarily for PAL backend build
    find_path(AMD_HSA_INCLUDE_DIR hsa.h
      HINTS
        ${ROCM_INSTALL_PATH}
        /opt/rocm
        ${CMAKE_CURRENT_BINARY_DIR}
      PATHS
        ${CMAKE_CURRENT_BINARY_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}/../..
        ${CMAKE_CURRENT_BINARY_DIR}/../../rocr
        ${ROCCLR_SRC_DIR}/../../rocr-runtime/runtime/hsa-runtime
      PATH_SUFFIXES
        include
        include/hsa
        inc)
    message("Roc CLR: " ${ROCCLR_SRC_DIR} "; HSA headers:" ${AMD_HSA_INCLUDE_DIR})
    target_include_directories(rocclr PUBLIC ${AMD_HSA_INCLUDE_DIR})
    target_include_directories(rocclr PUBLIC ${AMD_HSA_INCLUDE_DIR}/..)
    # Static linking on Windows with ROCR
    set (STATIC_ROCR ON)
  endif()

  if (ROCR_DLL_LOAD)
    target_compile_definitions(rocclr PUBLIC ROCR_DYN_DLL)
  else()
    if (STATIC_ROCR)
      target_link_libraries(rocclr PUBLIC hsa-runtime64::hsa-runtime64_static)
      if (WIN32)  # D3DKMTEnumAdapters3 requires OneCoreUAP.Lib
        target_link_libraries (rocclr PRIVATE OneCoreUAP.Lib)
      endif()
    else()
      target_link_libraries(rocclr PUBLIC hsa-runtime64::hsa-runtime64)
    endif()
  endif()
endif()
find_package(OpenGL REQUIRED)

target_sources(rocclr PRIVATE
  ${ROCCLR_SRC_DIR}/device/rocm/rocappprofile.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocrctx.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocblit.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocblitcl.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/roccounters.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocdevice.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rockernel.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocmemory.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocprintf.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocprogram.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocsettings.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocsignal.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocvirtual.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocurilocator.cpp)

if(UNIX)
  target_sources(rocclr PRIVATE
    ${ROCCLR_SRC_DIR}/device/rocm/rocglinterop.cpp)
else()
  target_sources(rocclr PRIVATE
    ${ROCCLR_SRC_DIR}/device/rocm/rocglinterop_windows.cpp)
endif()

target_compile_definitions(rocclr PUBLIC WITH_HSA_DEVICE)
