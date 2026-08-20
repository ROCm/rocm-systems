# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if (AMD_COMPUTE_WIN)
  find_path(AMD_HSA_INCLUDE_DIR hsa.h
    PATHS
      ${ROCCLR_SRC_DIR}/../../rocr-runtime/runtime/hsa-runtime/inc
      ${ROCCLR_SRC_DIR}/../../rocr-runtime/runtime/hsa-runtime
      ${CMAKE_CURRENT_BINARY_DIR}/../../rocr/inc
      ${CMAKE_CURRENT_BINARY_DIR}/../../rocr
      ${CMAKE_CURRENT_BINARY_DIR}/../..
      ${CMAKE_CURRENT_BINARY_DIR}/..
      ${CMAKE_CURRENT_BINARY_DIR}
      ${ROCM_PATH}
      ${ROCM_INSTALL_PATH}
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
        ${ROCM_PATH}
        ${ROCM_INSTALL_PATH}
      PATH_SUFFIXES
        cmake/hsa-runtime64
        lib/cmake/hsa-runtime64
        lib64/cmake/hsa-runtime64)
  else()
    find_package(hsa-runtime64 1.11 REQUIRED CONFIG
      PATHS
        ${ROCM_PATH}
        ${ROCM_INSTALL_PATH}
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
        ${ROCM_PATH}
        ${ROCM_INSTALL_PATH}
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

# Regenerate the embedded GPU graph-scheduler source header whenever the .hip
# source changes, so the runtime-compiled scheduler (compiled via comgr in
# Device::compileGraphSchedulerCodeObject, like the blit kernels) stays in sync.
set(GRAPH_SCHED_SRC ${ROCCLR_SRC_DIR}/device/rocm/graph_scheduler_kernel.hip)
set(GRAPH_SCHED_HDR ${ROCCLR_SRC_DIR}/device/rocm/graph_scheduler_source.h)
set(GRAPH_SCHED_GEN ${ROCCLR_SRC_DIR}/device/rocm/gen_scheduler_source.sh)
add_custom_command(
  OUTPUT ${GRAPH_SCHED_HDR}
  COMMAND bash ${GRAPH_SCHED_GEN} ${GRAPH_SCHED_SRC} ${GRAPH_SCHED_HDR}
  DEPENDS ${GRAPH_SCHED_SRC} ${GRAPH_SCHED_GEN}
  COMMENT "Embedding GPU graph scheduler source (graph_scheduler_source.h)"
  VERBATIM)
add_custom_target(gen_graph_scheduler_source DEPENDS ${GRAPH_SCHED_HDR})
add_dependencies(rocclr gen_graph_scheduler_source)
set_source_files_properties(${GRAPH_SCHED_HDR} PROPERTIES GENERATED TRUE)

if(UNIX)
  target_sources(rocclr PRIVATE
    ${ROCCLR_SRC_DIR}/device/rocm/rocglinterop.cpp)
else()
  target_sources(rocclr PRIVATE
    ${ROCCLR_SRC_DIR}/device/rocm/rocglinterop_windows.cpp)
endif()

target_compile_definitions(rocclr PUBLIC WITH_HSA_DEVICE)
