cmake_minimum_required(VERSION 3.16.0)

if(NOT DEFINED VERSION_MAJOR)
    set(VERSION_MAJOR 0)
endif()

if(NOT DEFINED VERSION_MINOR)
    set(VERSION_MINOR 2)
endif()

if(NOT DEFINED VERSION_PATCH)
    set(VERSION_PATCH 7)
endif()

project(
    rocprof-trace-decoder
    VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
    LANGUAGES CXX)
if(NOT DEFINED PROJECT_NAME)
    set(PROJECT_NAME "rocprof-trace-decoder")
endif()

if(NOT DEFINED TTD_PROJECT_SOURCE_DIR)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(TTD_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

if(NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH ${TTD_PROJECT_SOURCE_DIR} ${TTD_PROJECT_SOURCE_DIR}/cmake)
endif()

include(GNUInstallDirs) # install directories
set(CMAKE_INSTALL_LIBDIR "lib") # rocm doesn't use lib64

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Default install prefix to /opt/rocm for ROCm projects (standalone build)
if(NOT DEFINED TTD_ALL_VERSIONS AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "/opt/rocm"
        CACHE PATH "Default ROCm install prefix" FORCE)
endif()

include(CMakePackageConfigHelpers)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Collect source files
file(
    GLOB
    ATT_DECODER_V3_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/trie.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gfx9/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gfx10/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gfx11/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gfx12/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mi400/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stitch/stitch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stitch/pctranslator.cpp)

# Public headers to install
set(TTD_PUBLIC_HEADERS
    ${TTD_PROJECT_SOURCE_DIR}/include/rocprof_trace_decoder/rocprof_trace_decoder.h
    ${TTD_PROJECT_SOURCE_DIR}/include/rocprof_trace_decoder/trace_decoder_types.h
    ${TTD_PROJECT_SOURCE_DIR}/include/rocprof_trace_decoder/trace_decoder_instrument.h)

set(TTD_PUBLIC_CXX_HEADERS
    ${TTD_PROJECT_SOURCE_DIR}/include/rocprof_trace_decoder/cxx/code_printing.hpp
    ${TTD_PROJECT_SOURCE_DIR}/include/rocprof_trace_decoder/cxx/common.hpp
    ${TTD_PROJECT_SOURCE_DIR}/include/rocprof_trace_decoder/cxx/disassembly.hpp
    ${TTD_PROJECT_SOURCE_DIR}/include/rocprof_trace_decoder/cxx/funcmap.hpp
    ${TTD_PROJECT_SOURCE_DIR}/include/rocprof_trace_decoder/cxx/segment.hpp)

# --- Disassembly backend selection -----------------------------------------
#
# Mutually exclusive backends, picked at configure time:
#
#   DISABLE_COMGR=OFF / USE_LLVM_DISASM=OFF  (default)
#       Use amd_comgr. Default on ROCm because comgr hides its embedded
#       LLVM behind hidden-visibility symbols, avoiding global-state
#       collisions (e.g. duplicate LLVM CommandLine registrations) when
#       the rest of the process also drags LLVM in transitively.
#
#   USE_LLVM_DISASM=ON
#       Use LLVM-C's disassembler. Works on Linux/Windows/macOS without ROCm.
#       LLVM_DIR may be passed to find_package(LLVM) for a custom build —
#       useful when a new ASIC ships disasm support before the system LLVM
#       picks it up. Note: links LLVM PUBLIC, so consumers inherit it.
#
#   DISABLE_COMGR=ON and USE_LLVM_DISASM=OFF
#       Build without disassembly. va2fo + symbol enumeration still work
#       (inline ELF parsing) — callers that don't disassemble are fine.
# ---------------------------------------------------------------------------
if(NOT DEFINED DISABLE_COMGR)
    option(DISABLE_COMGR "Disable dependency on amd_comgr disassembly" OFF)
endif()
if(NOT DEFINED USE_LLVM_DISASM)
    option(USE_LLVM_DISASM "Use LLVM-C disassembler instead of amd_comgr" OFF)
endif()

set(TTD_DISASM_LINK_LIBS)
set(TTD_DISASM_INCLUDE_DIRS)
set(TTD_DISASM_COMPILE_DEFS)

if(USE_LLVM_DISASM)
    # CONFIG mode is the supported path: LLVM ships LLVMConfig.cmake with
    # every dev install (apt llvm-XX-dev, brew llvm, official Windows
    # installer, custom builds). Honor LLVM_DIR for non-system installs.
    #
    # Default to the ROCm-bundled LLVM (always built with AMDGPU and matched
    # to the rest of the stack) instead of letting CMake pick a system LLVM
    # such as /usr/lib/llvm-XX, which may lag on AMDGPU support. Users can
    # still override by passing -DLLVM_DIR=... on the command line or
    # exporting it in the environment.
    if(NOT DEFINED LLVM_DIR AND NOT DEFINED ENV{LLVM_DIR})
        foreach(_rocm_llvm_dir "/opt/rocm/lib/llvm/lib/cmake/llvm" "/opt/rocm/llvm/lib/cmake/llvm")
            if(EXISTS "${_rocm_llvm_dir}/LLVMConfig.cmake")
                set(LLVM_DIR "${_rocm_llvm_dir}")
                message(STATUS "Defaulting LLVM_DIR to ROCm-bundled LLVM: ${LLVM_DIR}")
                break()
            endif()
        endforeach()
    endif()
    find_package(LLVM CONFIG QUIET)
    if(LLVM_FOUND)
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (dir: ${LLVM_DIR})")
        # Verify AMDGPU is in the targets list — bailing here is far better
        # than a confused link error or a runtime "unknown target" message.
        list(FIND LLVM_TARGETS_TO_BUILD "AMDGPU" _amdgpu_idx)
        if(_amdgpu_idx LESS 0)
            message(
                FATAL_ERROR
                    "LLVM at ${LLVM_DIR} was not built with the AMDGPU target "
                    "(targets: ${LLVM_TARGETS_TO_BUILD}). Pass LLVM_DIR pointing "
                    "at an LLVM build that includes AMDGPU, or set " "USE_LLVM_DISASM=OFF.")
        endif()
        # llvm_map_components_to_libnames gives us the smallest set; falling
        # back to the monolithic libLLVM if components aren't separable.
        if(LLVM_LINK_LLVM_DYLIB)
            set(TTD_DISASM_LINK_LIBS LLVM)
        else()
            llvm_map_components_to_libnames(_llvm_libs MCDisassembler AMDGPUDisassembler AMDGPUDesc AMDGPUInfo Support)
            set(TTD_DISASM_LINK_LIBS ${_llvm_libs})
        endif()
        set(TTD_DISASM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
        list(APPEND TTD_DISASM_COMPILE_DEFS ROCPROF_TRACE_DECODER_USE_LLVM)
        # Force comgr off when LLVM is in use; the two backends are mutually
        # exclusive (header guards enforce this anyway).
        set(DISABLE_COMGR ON)
    else()
        message(STATUS "LLVM not found via find_package(LLVM CONFIG); falling back to amd_comgr if available")
        set(USE_LLVM_DISASM OFF)
    endif()
endif()

if(NOT USE_LLVM_DISASM AND NOT DISABLE_COMGR)
    find_library(
        COMGR_LIB
        NAMES amd_comgr
        HINTS /opt/rocm/lib)
    if(COMGR_LIB)
        message(STATUS "Found amd_comgr: ${COMGR_LIB}")
        set(TTD_DISASM_LINK_LIBS ${COMGR_LIB})
        find_path(
            ROCM_INCLUDE_DIR
            NAMES amd_comgr/amd_comgr.h
            HINTS /opt/rocm/include)
        if(NOT ROCM_INCLUDE_DIR)
            set(ROCM_INCLUDE_DIR "/opt/rocm/include")
        endif()
        set(TTD_DISASM_INCLUDE_DIRS ${ROCM_INCLUDE_DIR})
        list(APPEND TTD_DISASM_COMPILE_DEFS ROCPROF_TRACE_DECODER_USE_COMGR)
    else()
        message(WARNING "amd_comgr not found and USE_LLVM_DISASM=OFF — disassembly disabled")
        set(DISABLE_COMGR ON)
    endif()
endif()

if(DISABLE_COMGR AND NOT USE_LLVM_DISASM)
    list(APPEND TTD_DISASM_COMPILE_DEFS ROCPROF_TRACE_DECODER_COMGR_DISABLED)
endif()

# --- Object library (compiled once, reused by shared and static) ---
add_library(rocprof-trace-decoder-objects OBJECT ${ATT_DECODER_V3_FILES})
target_include_directories(
    rocprof-trace-decoder-objects
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC $<BUILD_INTERFACE:${TTD_PROJECT_SOURCE_DIR}/include>)
if(TTD_DISASM_INCLUDE_DIRS)
    target_include_directories(rocprof-trace-decoder-objects PRIVATE ${TTD_DISASM_INCLUDE_DIRS})
endif()
if(TTD_DISASM_COMPILE_DEFS)
    target_compile_definitions(rocprof-trace-decoder-objects PRIVATE ${TTD_DISASM_COMPILE_DEFS})
endif()
set_target_properties(rocprof-trace-decoder-objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(SQTT_LOGGING "Enable SQTT logging" OFF)
if(SQTT_LOGGING)
    target_compile_definitions(rocprof-trace-decoder-objects PRIVATE SQTT_LOGGING)
endif()

option(ARCH_MODEL "Enable ARCH_MODEL stitcher path" OFF)
if(ARCH_MODEL)
    target_compile_definitions(rocprof-trace-decoder-objects PRIVATE ARCH_MODEL)
endif()

# --- Shared library (installed) ---
#
# LLVM is exposed PUBLICly because cxx/disassembly.hpp is header-only and
# inlines LLVM-C calls into the consuming TU. Consumers of either the shared
# or static library therefore need LLVM's include dirs, link libs, and the
# `ROCPROF_TRACE_DECODER_USE_LLVM` define on their own command line.
# Propagating via PUBLIC means downstream targets (e.g. RCV) just link the
# namespaced target and inherit the right LLVM transitively — no second
# `find_package(LLVM)` needed in the consumer.
add_library(rocprof-trace-decoder SHARED $<TARGET_OBJECTS:rocprof-trace-decoder-objects>)
target_include_directories(
    rocprof-trace-decoder PUBLIC $<BUILD_INTERFACE:${TTD_PROJECT_SOURCE_DIR}/include>
                                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocprof-trace-decoder>)
if(TTD_DISASM_INCLUDE_DIRS)
    target_include_directories(rocprof-trace-decoder PUBLIC ${TTD_DISASM_INCLUDE_DIRS})
endif()
if(TTD_DISASM_COMPILE_DEFS)
    target_compile_definitions(rocprof-trace-decoder PUBLIC ${TTD_DISASM_COMPILE_DEFS})
endif()
target_link_libraries(
    rocprof-trace-decoder
    PRIVATE pthread
    PUBLIC ${TTD_DISASM_LINK_LIBS})
set_target_properties(
    rocprof-trace-decoder
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME rocprof-trace-decoder
               INSTALL_RPATH "\$ORIGIN;\$ORIGIN/llvm/lib"
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# --- Static library (built but NOT installed) ---
#
# Same PUBLIC propagation rationale as the shared lib. The static target is
# additionally exported into the build-tree CMake package below so consumers
# (e.g. RCV) can link against it directly without an install step.
add_library(rocprof-trace-decoder-static STATIC $<TARGET_OBJECTS:rocprof-trace-decoder-objects>)
target_include_directories(rocprof-trace-decoder-static PUBLIC $<BUILD_INTERFACE:${TTD_PROJECT_SOURCE_DIR}/include>)
if(TTD_DISASM_INCLUDE_DIRS)
    target_include_directories(rocprof-trace-decoder-static PUBLIC ${TTD_DISASM_INCLUDE_DIRS})
endif()
if(TTD_DISASM_COMPILE_DEFS)
    target_compile_definitions(rocprof-trace-decoder-static PUBLIC ${TTD_DISASM_COMPILE_DEFS})
endif()
target_link_libraries(
    rocprof-trace-decoder-static
    PRIVATE pthread
    PUBLIC ${TTD_DISASM_LINK_LIBS})
set_target_properties(rocprof-trace-decoder-static PROPERTIES OUTPUT_NAME rocprof-trace-decoder
                                                              ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# --- Sanitized shared library (test-only, not installed) ---
if(DEFINED BUILD_TESTS AND BUILD_TESTS)
    add_library(rocprof-trace-decoder-sanitized SHARED ${ATT_DECODER_V3_FILES})
    target_include_directories(
        rocprof-trace-decoder-sanitized
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC $<BUILD_INTERFACE:${TTD_PROJECT_SOURCE_DIR}/include>)
    target_link_libraries(rocprof-trace-decoder-sanitized PRIVATE pthread ${TTD_DISASM_LINK_LIBS})
    if(TTD_DISASM_INCLUDE_DIRS)
        target_include_directories(rocprof-trace-decoder-sanitized PRIVATE ${TTD_DISASM_INCLUDE_DIRS})
    endif()
    if(TTD_DISASM_COMPILE_DEFS)
        target_compile_definitions(rocprof-trace-decoder-sanitized PRIVATE ${TTD_DISASM_COMPILE_DEFS})
    endif()
    target_compile_options(rocprof-trace-decoder-sanitized PRIVATE -fsanitize=address,undefined -g -O1)
    target_link_options(rocprof-trace-decoder-sanitized PRIVATE -fsanitize=address,undefined)
    if(ARCH_MODEL)
        target_compile_definitions(rocprof-trace-decoder-sanitized PRIVATE ARCH_MODEL)
    endif()
    set_target_properties(rocprof-trace-decoder-sanitized PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

# --- Install shared library ---
install(
    TARGETS rocprof-trace-decoder
    EXPORT rocprof-trace-decoder-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)

# --- Build-tree export (no install required) ---
#
# Lets downstream projects consume the decoder directly from this build tree:
#
#   find_package(rocprof-trace-decoder REQUIRED CONFIG
#                PATHS /path/to/decoder/build/source)
#   target_link_libraries(myapp PRIVATE rocprof-trace-decoder::rocprof-trace-decoder-static)
#
# Both shared and static targets are exported so the consumer picks. The
# static target carries the PUBLIC LLVM dependency, so the consumer inherits
# the right LLVM transitively without re-running find_package(LLVM).
export(
    TARGETS rocprof-trace-decoder rocprof-trace-decoder-static
    NAMESPACE rocprof-trace-decoder::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/rocprof-trace-decoder-targets.cmake")

# --- Install public headers ---
install(
    FILES ${TTD_PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprof-trace-decoder/rocprof_trace_decoder
    COMPONENT devel)

install(
    FILES ${TTD_PUBLIC_CXX_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprof-trace-decoder/rocprof_trace_decoder/cxx
    COMPONENT devel)

# --- CMake package config for find_package(rocprof-trace-decoder) ---
set(TTD_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/rocprof-trace-decoder")

install(
    EXPORT rocprof-trace-decoder-targets
    FILE rocprof-trace-decoder-targets.cmake
    NAMESPACE rocprof-trace-decoder::
    DESTINATION ${TTD_CONFIG_INSTALL_DIR}
    COMPONENT devel)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rocprof-trace-decoder-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

# Snapshot the disasm-backend variables so the generated config file can
# re-find the same LLVM (or comgr) the library was built against. Booleans
# need to be normalized to 0/1 because @VAR@ substitution is textual and
# CMake will otherwise emit empty strings for unset variables.
if(USE_LLVM_DISASM)
    set(USE_LLVM_DISASM 1)
else()
    set(USE_LLVM_DISASM 0)
endif()
if(DISABLE_COMGR)
    set(DISABLE_COMGR 1)
else()
    set(DISABLE_COMGR 0)
endif()

configure_package_config_file(
    "${TTD_PROJECT_SOURCE_DIR}/cmake/rocprof-trace-decoder-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rocprof-trace-decoder-config.cmake" INSTALL_DESTINATION ${TTD_CONFIG_INSTALL_DIR})

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/rocprof-trace-decoder-config.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/rocprof-trace-decoder-config-version.cmake"
    DESTINATION ${TTD_CONFIG_INSTALL_DIR}
    COMPONENT devel)

# --- CPack variables (used when source/ is built standalone) ---
if(NOT DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_PACKAGE_NAME "rocprof-trace-decoder")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ROCprof Trace Decoder for Rocprofiler-SDK")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "ROCm Profiler Support <dl.ROCm-Profiler.support@amd.com>")

set(CPACK_RESOURCE_FILE_LICENSE "${TTD_PROJECT_SOURCE_DIR}/LICENSE")

if(NOT DEFINED TTD_ALL_VERSIONS)
    include(packaging)
endif()
