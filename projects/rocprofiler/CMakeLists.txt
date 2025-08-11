# ##############################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ##############################################################################

cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

set(CMAKE_C_FLAGS_COVERAGE_INIT
    "-Og -g3 -fno-omit-frame-pointer -fprofile-abs-path -fprofile-arcs -ftest-coverage --coverage"
    CACHE STRING "C flags for code coverage builds")
set(CMAKE_CXX_FLAGS_COVERAGE_INIT
    "-Og -g3 -fno-omit-frame-pointer -fprofile-abs-path -fprofile-arcs -ftest-coverage --coverage"
    CACHE STRING "C++ flags for code coverage builds")

set(CMAKE_C_FLAGS_COVERAGE
    "${CMAKE_C_FLAGS_COVERAGE_INIT}"
    CACHE STRING "C flags for code coverage builds")
set(CMAKE_CXX_FLAGS_COVERAGE
    "${CMAKE_CXX_FLAGS_COVERAGE_INIT}"
    CACHE STRING "C++ flags for code coverage builds")

# Build is not supported on Windows plaform
if(WIN32)
    message(FATAL_ERROR "Windows build is not supported.")
endif()

# Set module name and project name.
set(ROCPROFILER_NAME "rocprofiler")
set(ROCPROFILER_TARGET "${ROCPROFILER_NAME}64")
set(ROCPROFILER_LIBRARY "lib${ROCPROFILER_TARGET}")
project(rocprofiler VERSION 2.0.0)

if(${ROCM_PATCH_VERSION})
   set(PROJECT_VERSION_PATCH ${ROCM_PATCH_VERSION})
   set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
endif()

include(GNUInstallDirs)

# set default ROCM_PATH
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH
        "/opt/rocm"
        CACHE STRING "Default ROCM installation directory")
endif()
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_BUILD_RPATH
    "${PROJECT_BINARY_DIR}:${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_BUILD_RPATH OFF)

# Adding default path cmake modules
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

# Set build environment
include(rocprofiler_options)
include(rocprofiler_utils)
include(rocprofiler_env)
include(rocprofiler_formatting)
include(rocprofiler_linting)

# Setup the package version.
rocprofiler_get_version("1.0.0")
message("-- LIB-VERSION: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

set(BUILD_VERSION_MAJOR ${VERSION_MAJOR})
set(BUILD_VERSION_MINOR ${VERSION_MINOR})
set(BUILD_VERSION_PATCH ${VERSION_PATCH})
if(DEFINED VERSION_BUILD AND NOT ${VERSION_BUILD} STREQUAL "")
    message("VERSION BUILD DEFINED ${VERSION_BUILD}")
    set(BUILD_VERSION_PATCH "${BUILD_VERSION_PATCH}-${VERSION_BUILD}")
endif()
set(BUILD_VERSION_STRING
    "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}")

set(LIB_VERSION_MAJOR ${VERSION_MAJOR})
set(LIB_VERSION_MINOR ${VERSION_MINOR})
if(${ROCM_PATCH_VERSION})
    set(LIB_VERSION_PATCH ${ROCM_PATCH_VERSION})
else()
    set(LIB_VERSION_PATCH ${VERSION_PATCH})
endif()
set(LIB_VERSION_STRING "${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}.${LIB_VERSION_PATCH}")
message("-- LIB-VERSION STRING: ${LIB_VERSION_STRING}")

# Set target and root/lib/test directory
set(TARGET_NAME "${ROCPROFILER_TARGET}")
set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LIB_DIR "${ROOT_DIR}/src")
set(TEST_DIR "${ROOT_DIR}/test")

# To set additional RUNPATH in libraries installed in /opt/rocm-ver/lib/roctracer
set(ROCM_APPEND_PRIVLIB_RPATH "\$ORIGIN/..")

find_package(
    amd_comgr REQUIRED CONFIG
    HINTS ${CMAKE_INSTALL_PREFIX}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib/cmake/amd_comgr)
message(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")
link_libraries(amd_comgr)

find_package(Threads REQUIRED)
find_package(
    hsa-runtime64 REQUIRED CONFIG
    HINTS ${CMAKE_INSTALL_PREFIX}
    PATHS ${ROCM_PATH})
find_package(
    HIP REQUIRED CONFIG
    HINTS ${CMAKE_INSTALL_PREFIX}
    PATHS ${ROCM_PATH})

find_package(
    Clang CONFIG
    PATHS "${ROCM_PATH}"
    PATH_SUFFIXES "llvm/lib/cmake/clang")

if(NOT Clang_FOUND)
    message(
        FATAL_ERROR
            "\
The \"rocm-llvm-dev\" package is not installed. \
Please install it for your OS distribution. e.g: for Ubuntu: \"sudo apt-get install rocm-llvm-dev\".\
")
endif()

find_library(NUMA NAME numa REQUIRED)
link_libraries(${NUMA})

get_property(
    HSA_RUNTIME_INCLUDE_DIRECTORIES
    TARGET hsa-runtime64::hsa-runtime64
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(
    HSA_H hsa.h
    PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES hsa
    NO_DEFAULT_PATH REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH ${HSA_H} DIRECTORY)
include_directories(${HSA_RUNTIME_INC_PATH})

if(NOT DEFINED LIBRARY_TYPE)
    set(LIBRARY_TYPE SHARED)
endif()

# Enable tracing API
if(NOT USE_PROF_API)
    set(USE_PROF_API 1)
endif()

# Enable use of getROCmInstallPath() API
if(USE_GET_ROCM_PATH_API)
    set(USE_GET_ROCM_PATH_API 1)
    message(STATUS "Enabled Use of getROCmInstallPath() API" )
endif()

# Protocol header lookup
set(PROF_API_HEADER_NAME prof_protocol.h)
if(USE_PROF_API EQUAL 1)
    find_path(
        PROF_API_HEADER_DIR ${PROF_API_HEADER_NAME}
        HINTS ${PROF_API_HEADER_PATH}
        PATHS /opt/rocm/include
        PATH_SUFFIXES roctracer/ext)
    if(NOT PROF_API_HEADER_DIR)
        message(
            FATAL_ERROR
                "Profiling API header not found. Tracer integration disabled. Use -DPROF_API_HEADER_PATH=<path to ${PROF_API_HEADER_NAME} header>"
            )
    else()
        include_directories(${PROF_API_HEADER_DIR})
        message(STATUS "Profiling API: ${PROF_API_HEADER_DIR}/${PROF_API_HEADER_NAME}")
    endif()
endif()

enable_testing()

# Temporarily for CI to work
set(ROCPROFILER_BUILD_TESTS ON)
set(ROCPROFILER_BUILD_CI ON)

if(ROCPROFILER_BUILD_CI)
    include(CTest)
endif()

# Build libraries
add_subdirectory(src)

# Build Plugins
add_subdirectory(plugin)

if(ROCPROFILER_BUILD_SAMPLES)
    # Build samples
    add_subdirectory(samples)
endif()

if(ROCPROFILER_BUILD_TESTS)
    # Build tests
    add_subdirectory(test)
    add_subdirectory(tests-v2)
endif()

# Installation and packaging
set(DEST_NAME ${ROCPROFILER_NAME})
if(DEFINED CMAKE_INSTALL_PREFIX)
    get_filename_component(prefix_name ${CMAKE_INSTALL_PREFIX} NAME)
    get_filename_component(prefix_dir ${CMAKE_INSTALL_PREFIX} DIRECTORY)
    if(prefix_name STREQUAL ${DEST_NAME})
        set(CMAKE_INSTALL_PREFIX ${prefix_dir})
    endif()
endif()
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    get_filename_component(prefix_name ${CPACK_PACKAGING_INSTALL_PREFIX} NAME)
    get_filename_component(prefix_dir ${CPACK_PACKAGING_INSTALL_PREFIX} DIRECTORY)
    if(prefix_name STREQUAL ${DEST_NAME})
        set(CPACK_PACKAGING_INSTALL_PREFIX ${prefix_dir})
    endif()
else()
    set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

set(ORIGINAL_SCRIPT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/tblextr.py)
set(OUTPUT_SCRIPT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin/tblextr.py)
configure_file(${ORIGINAL_SCRIPT_PATH} ${OUTPUT_SCRIPT_PATH} @ONLY)

message("CMake-install-prefix: ${CMAKE_INSTALL_PREFIX}")
message("CPack-install-prefix: ${CPACK_PACKAGING_INSTALL_PREFIX}")
message("-----------Dest-name: ${DEST_NAME}")

# Install headers
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/core/activity.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${ROCPROFILER_NAME}
    COMPONENT dev)

# rpl_run.sh
install(
    PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/bin/rpl_run.sh
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    RENAME rocprof
    COMPONENT runtime)

configure_file(bin/rocprofv2 ${PROJECT_BINARY_DIR} COPYONLY)
install(
    PROGRAMS ${PROJECT_SOURCE_DIR}/bin/rocprofv2
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT runtime)

install(
    PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/bin/txt2xml.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/merge_traces.sh
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${ROCPROFILER_NAME}
    COMPONENT runtime)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/bin/txt2params.py
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/txt2params.py
          ${CMAKE_CURRENT_BINARY_DIR}/bin/tblextr.py
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/dform.py
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/mem_manager.py
          ${CMAKE_CURRENT_SOURCE_DIR}/bin/sqlitedb.py
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${ROCPROFILER_NAME}
    COMPONENT runtime)

# gfx_metrics.xml metrics.xml
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/test/tool/metrics.xml
          ${CMAKE_CURRENT_SOURCE_DIR}/test/tool/gfx_metrics.xml
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/${ROCPROFILER_NAME}
    COMPONENT runtime)

if(${LIBRARY_TYPE} STREQUAL SHARED)
    # Packaging directives
    set(CPACK_GENERATOR "DEB" "RPM" "TGZ" CACHE STRING "CPACK GENERATOR DEB;RPM")
    set(ENABLE_LDCONFIG
        ON
        CACHE BOOL "Set library links and caches using ldconfig.")
    set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
    set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(CPACK_PACKAGE_VERSION
        "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
        )
    set(CPACK_PACKAGE_CONTACT "ROCm Profiler Support <dl.ROCm-Profiler.support@amd.com>")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
        "ROCPROFILER library for AMD HSA runtime API extension support")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

    if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
        set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
        message("Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}")
    endif()

    # Debian package specific variable for ASAN
    set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${ROCPROFILER_NAME}-asan")
    set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "hsa-rocr-asan, rocm-core-asan")

    # Install license file
    install(
        FILES ${CPACK_RESOURCE_FILE_LICENSE}
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        COMPONENT runtime)
    install(
        FILES ${CPACK_RESOURCE_FILE_LICENSE}
        DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan
        COMPONENT asan)

    # Debian package specific variables
    if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
        set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    else()
        set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
    endif()

    message("Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}")
    set(CPACK_DEB_COMPONENT_INSTALL ON)
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
    set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
    set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS
        "rocminfo, hsa-rocr, rocm-core, libelf-dev, libnuma-dev, libxml2-dev"
        )
    set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PROJECT_NAME}-dev")
    set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "${PROJECT_NAME}, hsa-rocr-dev, rocm-core")
    set(CPACK_DEBIAN_TESTS_PACKAGE_NAME "${PROJECT_NAME}-tests")
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "${PROJECT_NAME}-dev, hsa-rocr-dev, rocm-core")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_NAME "${PROJECT_NAME}-samples")
    set(CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS
        "${PROJECT_NAME}-dev, hsa-rocr-dev, rocm-core")
    set(CPACK_DEBIAN_DOCS_PACKAGE_NAME "${PROJECT_NAME}-docs")
    set(CPACK_DEBIAN_DOCS_PACKAGE_DEPENDS "${PROJECT_NAME}-dev, hsa-rocr-dev, rocm-core")
    set(CPACK_DEBIAN_PLUGINS_PACKAGE_NAME "${PROJECT_NAME}-plugins")
    set(CPACK_DEBIAN_PLUGINS_PACKAGE_DEPENDS "${PROJECT_NAME}, hsa-rocr-dev, rocm-core")

    set(CPACK_DEBIAN_CHANGELOG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CHANGELOG.md")

    # RPM package specific variables
    if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
        set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
    else()
        set(CPACK_RPM_PACKAGE_RELEASE "local")
    endif()

    message("Using CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE}")

    set(CPACK_RPM_PACKAGE_LICENSE "MIT")

    # 'dist' breaks manual builds on debian systems due to empty Provides
    execute_process(
        COMMAND rpm --eval %{?dist}
        RESULT_VARIABLE PROC_RESULT
        OUTPUT_VARIABLE EVAL_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")

    if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
        string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
    endif()

    set(CPACK_RPM_COMPONENT_INSTALL ON)
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
    set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
    set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES
        "rocminfo, hsa-rocr, rocm-core, libxml2-devel")
    set(CPACK_RPM_DEV_PACKAGE_NAME "${PROJECT_NAME}-devel")
    set(CPACK_RPM_DEV_PACKAGE_REQUIRES "${PROJECT_NAME}, hsa-rocr-devel, rocm-core")
    set(CPACK_RPM_DEV_PACKAGE_PROVIDES "${PROJECT_NAME}-dev")
    set(CPACK_RPM_DEV_PACKAGE_OBSOLETES "${PROJECT_NAME}-dev")
    set(CPACK_RPM_TESTS_PACKAGE_NAME "${PROJECT_NAME}-tests")
    set(CPACK_RPM_TESTS_PACKAGE_REQUIRES "${PROJECT_NAME}-devel, hsa-rocr-devel, rocm-core")
    set(CPACK_RPM_DOCS_PACKAGE_NAME "${PROJECT_NAME}-docs")
    set(CPACK_RPM_DOCS_PACKAGE_REQUIRES "${PROJECT_NAME}-devel, hsa-rocr-devel, rocm-core")
    set(CPACK_RPM_PLUGINS_PACKAGE_NAME "${PROJECT_NAME}-plugins")
    set(CPACK_RPM_PLUGINS_PACKAGE_REQUIRES "${PROJECT_NAME}, hsa-rocr-devel, rocm-core")
    set(CPACK_RPM_PACKAGE_AUTOREQ 0)
    set(CPACK_RPM_SAMPLES_PACKAGE_NAME "${PROJECT_NAME}-samples")
    set(CPACK_RPM_SAMPLES_PACKAGE_REQUIRES
        "${PROJECT_NAME}-devel, hsa-rocr-devel, rocm-core, hip-runtime-amd")
    message("CPACK_RPM_PACKAGE_RELEASE: ${CPACK_RPM_PACKAGE_RELEASE}")

    # Disable build id for rocprofiler as its creating transaction error
    set(CPACK_RPM_SPEC_MORE_DEFINE
        "%define _build_id_links none
                                    %global __strip ${CPACK_STRIP_EXECUTABLE}
                                    %global __objdump ${CPACK_OBJDUMP_EXECUTABLE}
                                    %global __objcopy ${CPACK_OBJCOPY_EXECUTABLE}
                                    %global __readelf ${CPACK_READELF_EXECUTABLE}")

    # RPM package specific variable for ASAN
    set(CPACK_RPM_ASAN_PACKAGE_NAME "${ROCPROFILER_NAME}-asan")
    set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "hsa-rocr-asan, rocm-core-asan")

    # set ( CPACK_RPM_CHANGELOG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CHANGELOG.md" )

    # Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
    if(NOT ROCM_DEP_ROCMCORE)
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_RUNTIME_PACKAGE_REQUIRES
                             ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DEV_PACKAGE_REQUIRES
                             ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
        string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_RPM_ASAN_PACKAGE_REQUIRES
                             ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_TESTS_PACKAGE_REQUIRES
                             ${CPACK_RPM_TESTS_PACKAGE_REQUIRES})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_SAMPLES_PACKAGE_REQUIRES
                             ${CPACK_RPM_SAMPLES_PACKAGE_REQUIRES})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DOCS_PACKAGE_REQUIRES
                             ${CPACK_RPM_DOCS_PACKAGE_REQUIRES})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PLUGINS_PACKAGE_REQUIRES
                             ${CPACK_RPM_PLUGINS_PACKAGE_REQUIRES})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS
                             ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS
                             ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
        string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS
                             ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS
                             ${CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS
                             ${CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DOCS_PACKAGE_DEPENDS
                             ${CPACK_DEBIAN_DOCS_PACKAGE_DEPENDS})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PLUGINS_PACKAGE_DEPENDS
                             ${CPACK_DEBIAN_PLUGINS_PACKAGE_DEPENDS})
    endif()

    # set components
    if(ENABLE_ASAN_PACKAGING)
        # ASAN Package requires only asan component with libraries and license file
        set(CPACK_COMPONENTS_ALL asan)
    else()
        set(CPACK_COMPONENTS_ALL runtime dev tests docs plugins samples)
    endif()

    include(CPack)

    cpack_add_component(
        runtime
        DISPLAY_NAME "Runtime"
        DESCRIPTION "Dynamic libraries for the ROCProfiler")

    cpack_add_component(
        dev
        DISPLAY_NAME "Development"
        DESCRIPTION "Development needed header files for ROCProfiler"
        DEPENDS runtime)

    cpack_add_component(
        plugins
        DISPLAY_NAME "ROCProfile Plugins"
        DESCRIPTION "Plugins for handling ROCProfiler data output"
        DEPENDS runtime)

    cpack_add_component(
        tests
        DISPLAY_NAME "Tests"
        DESCRIPTION "Tests for the ROCProfiler"
        DEPENDS dev)

    cpack_add_component(
        samples
        DISPLAY_NAME "Samples"
        DESCRIPTION "Samples for the ROCProfiler"
        DEPENDS dev)

    cpack_add_component(
        docs
        DISPLAY_NAME "Documentation"
        DESCRIPTION "Documentation for the ROCProfiler API"
        DEPENDS dev)

    cpack_add_component(
        asan
        DISPLAY_NAME "ASAN"
        DESCRIPTION "ASAN libraries for the ROCPROFILER"
        DEPENDS asan)
endif()

find_package(Doxygen)

if(DOXYGEN_FOUND)
    # # Set input and output files for API Document
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile_API.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_API)

    # # Request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html
               ${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/doc/latex pdf
        MAIN_DEPENDENCY ${DOXYGEN_OUT}
        ${DOXYGEN_IN}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/rocprofiler/v2/rocprofiler_plugin.h
                ${CMAKE_CURRENT_SOURCE_DIR}/include/rocprofiler/v2/rocprofiler.h
        COMMENT "Generating API documentation")

    add_custom_target(doc DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html
                                  ${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf)

    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf"
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        RENAME "${PROJECT_NAME}v2_api_spec.pdf"
        OPTIONAL
        COMPONENT docs)

    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc/html/"
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/html
        OPTIONAL
        COMPONENT docs)

    # # Set input and output files for Tools Document
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile_Tool.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/tooldoc/Doxyfile_Tool)

    # # Request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tooldoc/html/index.html
               ${CMAKE_CURRENT_BINARY_DIR}/tooldoc/latex/refman.pdf
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/tooldoc/latex pdf
        MAIN_DEPENDENCY ${DOXYGEN_OUT}
        ${DOXYGEN_IN}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/doc/rocprofv2_tool.md
        COMMENT "Generating Tools documentation")

    add_custom_target(
        doc_tool DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tooldoc/html/index.html
                         ${CMAKE_CURRENT_BINARY_DIR}/tooldoc/latex/refman.pdf)

    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/tooldoc/latex/refman.pdf"
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        RENAME "${PROJECT_NAME}v2_tool.pdf"
        OPTIONAL
        COMPONENT docs)

    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tooldoc/html/"
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/html
        OPTIONAL
        COMPONENT docs)

    # # Set input and output files for changelog document
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile_ChangeLog.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doc/changelog/Doxyfile_ChangeLog)

    # # Request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doc/changelog/latex/refman.pdf
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/doc/changelog/latex pdf
        MAIN_DEPENDENCY ${DOXYGEN_OUT}
        ${DOXYGEN_IN}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/CHANGELOG.md
        COMMENT "Generating changelog documentation")

    add_custom_target(doc_changelog
                      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doc/changelog/latex/refman.pdf)

    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/doc/changelog/latex/refman.pdf"
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        RENAME "${PROJECT_NAME}_ChangeLog.pdf"
        OPTIONAL
        COMPONENT docs)

    add_dependencies(doc doc_changelog)
endif()
