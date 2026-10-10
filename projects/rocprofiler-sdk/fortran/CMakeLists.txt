# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.19)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(roctx-fortran LANGUAGES NONE)
endif()

option(BUILD_FORTRAN_BINDINGS "Build the rocTX Fortran bindings" "${UNIX}")
option(BUILD_FORTRAN_CLIENTS "Build the rocTX Fortran binding tests" ON)

if(DEFINED ROCTX_BUILD_FORTRAN_BINDINGS)
    set(BUILD_FORTRAN_BINDINGS ${ROCTX_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED ROCTX_BUILD_FORTRAN_CLIENTS)
    set(BUILD_FORTRAN_CLIENTS ${ROCTX_BUILD_FORTRAN_CLIENTS})
endif()

set(FORTRAN_ARRAY_INTERFACES
    "assumed-shape"
    CACHE STRING
          "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS none assumed-shape
                                                     assumed-rank)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ROCTX_HAVE_FORTRAN_BINDINGS
        FALSE
        PARENT_SCOPE)
endif()

if(NOT BUILD_FORTRAN_BINDINGS)
    return()
endif()

if(NOT DEFINED ROCM_HAVE_FORTRAN)
    if(NOT DEFINED ROCM_SYSTEMS_ROOT)
        get_filename_component(ROCM_SYSTEMS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
                               ABSOLUTE)
    endif()
    include("${ROCM_SYSTEMS_ROOT}/shared/cmake/ROCmFortran.cmake" OPTIONAL)
endif()

if(DEFINED ROCM_HAVE_FORTRAN)
    set(_have_fortran ${ROCM_HAVE_FORTRAN})
else()
    if(NOT CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
        set(_fc_hints)
        foreach(_cc "${CMAKE_CXX_COMPILER}" "${CMAKE_C_COMPILER}" "${CMAKE_HIP_COMPILER}")
            if(_cc)
                get_filename_component(_cc_dir "${_cc}" DIRECTORY)
                list(APPEND _fc_hints "${_cc_dir}")
            endif()
        endforeach()
        foreach(_rocm_root "${ROCM_PATH}" "$ENV{ROCM_PATH}" "/opt/rocm")
            if(_rocm_root)
                list(APPEND _fc_hints "${_rocm_root}/lib/llvm/bin" "${_rocm_root}/bin")
            endif()
        endforeach()
        find_program(
            _amdflang
            NAMES amdflang
            HINTS ${_fc_hints}
            NO_DEFAULT_PATH)
        if(_amdflang)
            execute_process(
                COMMAND "${_amdflang}" --version
                RESULT_VARIABLE _amdflang_rc
                OUTPUT_QUIET ERROR_QUIET)
            if(_amdflang_rc EQUAL 0)
                set(CMAKE_Fortran_COMPILER
                    "${_amdflang}"
                    CACHE FILEPATH "Fortran compiler used for the ROCm Fortran bindings"
                          FORCE)
            endif()
        endif()
        unset(_amdflang CACHE)
    endif()

    include(CheckLanguage)
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
        set(_have_fortran TRUE)
    else()
        set(_have_fortran FALSE)
    endif()
endif()

if(NOT _have_fortran)
    message(STATUS "rocTX: no Fortran compiler found, skipping the Fortran bindings")
    return()
endif()

enable_language(Fortran)
include(GNUInstallDirs)

add_library(roctx_fortran STATIC roctx.F90)
add_library(roc::roctx_fortran ALIAS roctx_fortran)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(_fc_name "gfortran")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
    set(_fc_name "amdflang")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
    set(_fc_name "ftn")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
    set(_fc_name "ifx")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    set(_fc_name "ifort")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    set(_fc_name "nvfortran")
else()
    get_filename_component(_fc_name "${CMAKE_Fortran_COMPILER}" NAME_WE)
    string(TOLOWER "${_fc_name}" _fc_name)
endif()

set(ROCTX_FORTRAN_COMPILER_DIR
    "${_fc_name}"
    CACHE STRING "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(roctx_fortran PROPERTIES Fortran_MODULE_DIRECTORY "${_moddir}"
                                               POSITION_INDEPENDENT_CODE ON)

target_include_directories(
    roctx_fortran
    PUBLIC
        "$<BUILD_INTERFACE:${_moddir}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCTX_FORTRAN_COMPILER_DIR}>"
    )

if(NOT TARGET rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
   AND NOT TARGET rocprofiler-sdk-roctx-shared-library)
    if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
        enable_language(C)
    endif()
    find_package(rocprofiler-sdk-roctx QUIET)
endif()

if(TARGET rocprofiler-sdk-roctx-shared-library)
    target_link_libraries(
        roctx_fortran
        INTERFACE $<BUILD_INTERFACE:rocprofiler-sdk-roctx-shared-library>
                  $<INSTALL_INTERFACE:rocprofiler-sdk-roctx::rocprofiler-sdk-roctx>)
    set(ROCTX_FORTRAN_NEEDS_C_PACKAGE TRUE)
elseif(TARGET rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
    target_link_libraries(roctx_fortran
                          INTERFACE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
    set(ROCTX_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
    set(ROCTX_FORTRAN_NEEDS_C_PACKAGE FALSE)
    message(
        STATUS
            "rocTX: rocprofiler-sdk-roctx was not found, so the Fortran bindings carry "
            "no link interface; a consumer will have to link it "
            "(-lrocprofiler-sdk-roctx) itself.")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ROCTX_HAVE_FORTRAN_BINDINGS
        TRUE
        PARENT_SCOPE)
    if(NOT (ROCPROFILER_BUILD_TESTS OR ROCTX_BUILD_FORTRAN_CLIENTS))
        set(BUILD_FORTRAN_CLIENTS OFF)
    endif()
endif()

if(BUILD_FORTRAN_CLIENTS)
    enable_testing()
    add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/roctx-fortran")

install(
    TARGETS roctx_fortran
    EXPORT roctx-fortran-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${ROCTX_FORTRAN_COMPILER_DIR}"
            COMPONENT roctx)
install(
    DIRECTORY "${_moddir}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCTX_FORTRAN_COMPILER_DIR}"
    COMPONENT roctx
    FILES_MATCHING
    PATTERN "*.mod")
install(
    EXPORT roctx-fortran-targets
    FILE roctx-fortran-targets.cmake
    NAMESPACE roc::
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT roctx)

set(ROCTX_FORTRAN_LIBDIR "${CMAKE_INSTALL_LIBDIR}/fortran")
set(ROCTX_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/roctx-fortran-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config.cmake"
    INSTALL_DESTINATION "${_fortran_pkgdir}"
    PATH_VARS ROCTX_FORTRAN_LIBDIR ROCTX_FORTRAN_INCLUDEDIR)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config.cmake"
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT roctx)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" _roctx_version LIMIT_COUNT 1)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _roctx_version "${_roctx_version}")
endif()

if(_roctx_version)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config-version.cmake"
        VERSION "${_roctx_version}"
        COMPATIBILITY SameMajorVersion)
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config-version.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT roctx)
endif()

install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/roctx.F90"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/roctx/fortran"
    COMPONENT roctx)
