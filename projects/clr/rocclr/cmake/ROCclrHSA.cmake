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
  ${ROCCLR_SRC_DIR}/device/rocm/rgp/rocgpuopen.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rockernel.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocmemory.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocprintf.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocprogram.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocsettings.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocsignal.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocvirtual.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rgp/rocurilocator.cpp)

if(UNIX)
  target_sources(rocclr PRIVATE
    ${ROCCLR_SRC_DIR}/device/rocm/rocglinterop.cpp)
else()
  target_sources(rocclr PRIVATE
    ${ROCCLR_SRC_DIR}/device/rocm/rocglinterop_windows.cpp)
endif()

target_compile_definitions(rocclr PUBLIC WITH_HSA_DEVICE)

# On MSVC multi-config generators CMAKE_BUILD_TYPE is empty, so the /MTd vs /MT
# logic in CLR's top-level CMakeLists.txt never fires correctly. Set the runtime
# library explicitly per-configuration so the Release build does not pull in /MTd
# (which auto-defines _DEBUG, conflicting with NDEBUG → RELEASE in macros.hpp).
if(MSVC)
  set_property(TARGET rocclr PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(ROCCLR_ENABLE_GPUOPEN)
  target_compile_definitions(rocclr PUBLIC ROC_GPUOPEN)

  # ── RDF headers — needed by roctracesession.cpp in all builds ────────────────
  # RDF_DIR points to the standalone RDF checkout.  In the PAL build the amdrdf
  # *library* comes from PAL's link graph, but the compiler still needs amdrdf.h
  # on the include path because roctracesession.cpp is compiled in both paths.
  set(RDF_DIR "C:/github-emu/rdf"
      CACHE PATH "Root of the RDF source checkout")

  if(NOT EXISTS "${RDF_DIR}/rdf/inc/amdrdf.h")
    message(FATAL_ERROR
      "ROCclr: ROCCLR_ENABLE_GPUOPEN is ON but RDF source was not found at "
      "'${RDF_DIR}'. Set -DRDF_DIR=<path> to the rdf checkout.")
  endif()

  # Expose amdrdf.h to roctracesession.cpp regardless of PAL vs. non-PAL.
  target_include_directories(rocclr PRIVATE "${RDF_DIR}/rdf/inc")

  if(ROCCLR_ENABLE_PAL)
    # PAL carries its own DevDriver and RDF builds.  Linking them again from
    # source would produce duplicate targets and ODR violations.  The PAL
    # CMake integration exposes devdriver / amdrdf transitively through PAL's
    # own link graph, so rocclr picks them up automatically.
    message(STATUS "ROCclr GPUOpen: PAL build detected — skipping standalone "
                   "DevDriver / RDF source builds (provided by PAL).")
  else()
    # ── DevDriver source build ────────────────────────────────────────────────
    # Build DevDriver from source at DD_SOURCE_DIR (default: C:/github-emu/devdriver).
    # The source tree uses add_subdirectory() as its integration mechanism; there
    # are no exported CMake config files.
    set(DD_SOURCE_DIR "C:/github-emu/devdriver"
        CACHE PATH "Root of the DevDriver source checkout")

    if(NOT EXISTS "${DD_SOURCE_DIR}/shared/legacy/inc/devDriverServer.h")
      message(FATAL_ERROR
        "ROCclr: ROCCLR_ENABLE_GPUOPEN is ON but DevDriver source was not found at "
        "'${DD_SOURCE_DIR}'. Set -DDD_SOURCE_DIR=<path> to the devdriver checkout.")
    endif()

    # ── RDF source build ──────────────────────────────────────────────────────
    if(NOT TARGET amdrdf)
      set(RDF_STATIC              ON  CACHE BOOL "" FORCE)
      set(RDF_ENABLE_CXX_BINDINGS ON  CACHE BOOL "" FORCE)
      add_subdirectory("${RDF_DIR}" rdf_build EXCLUDE_FROM_ALL)
    endif()

    if(NOT TARGET devdriver)
      # Disable everything we don't need so the build stays fast.
      set(DD_BP_BUILD_MODULES         OFF CACHE BOOL "" FORCE)
      set(DD_BP_INSTALL               OFF CACHE BOOL "" FORCE)
      set(DD_BUILD_TESTS              OFF CACHE BOOL "" FORCE)
      set(DD_BP_ENABLE_TOOL_LIBRARIES OFF CACHE BOOL "" FORCE)
      set(DD_BP_ENABLE_DD_MODULE_APIS OFF CACHE BOOL "" FORCE)
      set(DD_BP_ENABLE_DD_TEST_UTIL   OFF CACHE BOOL "" FORCE)
      set(DD_BP_ENABLE_DD_SETTINGS    OFF CACHE BOOL "" FORCE)
      # DD_BUILD_RDF=ON so ddRpcServer and related APIs are built.
      set(DD_BUILD_RDF                ON  CACHE BOOL "" FORCE)

      add_subdirectory("${DD_SOURCE_DIR}" devdriver_build EXCLUDE_FROM_ALL)
    endif()

    # devdriver (STATIC) already carries ddCore, dd_common, stb_sprintf, metrohash,
    # mpack, and dd_interface as transitive link dependencies via its own
    # target_link_libraries(), so we only need to add devdriver + ddSocket here.
    # SetupAPI is needed on Windows for the KMD message-bus transport.
    if(NOT TARGET devdriver_roc)
      add_library(devdriver_roc INTERFACE)

      target_include_directories(devdriver_roc INTERFACE
        "${DD_SOURCE_DIR}/shared/legacy/inc"
        "${DD_SOURCE_DIR}/third_party/dd_crc32"
        "${DD_SOURCE_DIR}/apis/ddRpc/ddRpcServer/inc"   # ddRpcServer.h / ddRpcServerApi.h
        "${RDF_DIR}/rdf/inc")                            # amdrdf.h

      target_compile_definitions(devdriver_roc INTERFACE
        GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION=42)

      target_link_libraries(devdriver_roc INTERFACE
        devdriver    # RGPServer, DriverControlServer, DevDriverServer (+ transitive deps)
        ddSocket     # Named-pipe / KMD message-bus transport
        ddRpcServer  # UberTrace RPC server
        ddRpcShared  # Shared RPC types
        amdrdf       # RDF chunk-file writer for RocTraceSession
        SetupAPI.Lib) # Windows KMD bus enumeration
    endif()

    target_link_libraries(rocclr PUBLIC devdriver_roc)
  endif()

  # UberTrace source files — compiled in both PAL and non-PAL builds.
  # In the PAL path, devdriver/amdrdf/ddRpcServer are provided by PAL's link graph.
  # In the non-PAL path they come from the devdriver_roc interface target above.
  target_sources(rocclr PRIVATE
    ${ROCCLR_SRC_DIR}/device/rocm/rgp/roctracesession.cpp
    ${ROCCLR_SRC_DIR}/device/rocm/rgp/rocubertracesvc.cpp
    ${ROCCLR_SRC_DIR}/device/rocm/rgp/g_service/UberTraceService.cpp
    ${ROCCLR_SRC_DIR}/device/rocm/rgp/g_service/g_DriverUtilsService.cpp)

  # g_service generated header needs to be found as <UberTraceService.h>
  target_include_directories(rocclr PRIVATE
    ${ROCCLR_SRC_DIR}/device/rocm/rgp/g_service)
endif()
