cmake_minimum_required(VERSION 3.16.0)

if(NOT DEFINED VERSION_MAJOR)
    set(VERSION_MAJOR 0)
endif()

if(NOT DEFINED VERSION_MINOR)
    set(VERSION_MINOR 1)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/stitch/stitch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stitch/pctranslator.cpp)

# Public headers to install
set(TTD_PUBLIC_HEADERS
    ${TTD_PROJECT_SOURCE_DIR}/include/trace_decoder_api.h ${TTD_PROJECT_SOURCE_DIR}/include/trace_decoder_types.h
    ${TTD_PROJECT_SOURCE_DIR}/include/trace_decoder_instrument.h)

# --- Object library (compiled once, reused by shared and static) ---
add_library(rocprof-trace-decoder-objects OBJECT ${ATT_DECODER_V3_FILES})
target_include_directories(
    rocprof-trace-decoder-objects
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC $<BUILD_INTERFACE:${TTD_PROJECT_SOURCE_DIR}/include>)
set_target_properties(rocprof-trace-decoder-objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
option(SQTT_LOGGING "Enable SQTT logging" OFF)
if(SQTT_LOGGING)
    target_compile_definitions(rocprof-trace-decoder-objects PRIVATE SQTT_LOGGING)
endif()

# --- Shared library (installed) ---
add_library(rocprof-trace-decoder SHARED $<TARGET_OBJECTS:rocprof-trace-decoder-objects>)
target_include_directories(
    rocprof-trace-decoder PUBLIC $<BUILD_INTERFACE:${TTD_PROJECT_SOURCE_DIR}/include>
                                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocprof-trace-decoder>)
target_link_libraries(rocprof-trace-decoder PRIVATE pthread)
set_target_properties(
    rocprof-trace-decoder
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
               EXPORT_NAME rocprof-trace-decoder
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# --- Static library (built but NOT installed) ---
add_library(rocprof-trace-decoder-static STATIC $<TARGET_OBJECTS:rocprof-trace-decoder-objects>)
target_include_directories(rocprof-trace-decoder-static PUBLIC $<BUILD_INTERFACE:${TTD_PROJECT_SOURCE_DIR}/include>)
target_link_libraries(rocprof-trace-decoder-static PRIVATE pthread)
set_target_properties(rocprof-trace-decoder-static PROPERTIES OUTPUT_NAME rocprof-trace-decoder
                                                              ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# --- Sanitized shared library (test-only, not installed) ---
if(DEFINED BUILD_TESTS AND BUILD_TESTS)
    add_library(rocprof-trace-decoder-sanitized SHARED ${ATT_DECODER_V3_FILES})
    target_include_directories(
        rocprof-trace-decoder-sanitized
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC $<BUILD_INTERFACE:${TTD_PROJECT_SOURCE_DIR}/include>)
    target_link_libraries(rocprof-trace-decoder-sanitized PRIVATE pthread)
    target_compile_options(rocprof-trace-decoder-sanitized PRIVATE -fsanitize=address,undefined -g -O1)
    target_link_options(rocprof-trace-decoder-sanitized PRIVATE -fsanitize=address,undefined)
    set_target_properties(rocprof-trace-decoder-sanitized PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

# --- Install shared library ---
install(
    TARGETS rocprof-trace-decoder
    EXPORT rocprof-trace-decoder-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)

# --- Install public headers ---
install(
    FILES ${TTD_PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprof-trace-decoder
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
