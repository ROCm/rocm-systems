# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Fortran bindings for ROCTx: the `roctx` module, roc::roctx_fortran and the
# roctx-fortran package. roctx.F90 is generated from the ROCTx headers by
# rocm-fortran-generator; do not edit it by hand, regenerate it. A standalone
# project, built against an installed ROCTx (find_package(rocprofiler-sdk-roctx)).
# ROCTx has no array overloads, so there is no FORTRAN_ARRAY_INTERFACES here.

cmake_minimum_required(VERSION 3.19)

project(roctx-fortran LANGUAGES NONE)

# "${UNIX}": ROCm ships no Fortran compiler on Windows, where the probe would
# pick up a stray gfortran that enable_language(Fortran) then rejects.
option(BUILD_FORTRAN_BINDINGS "Build the ROCTx Fortran bindings" "${UNIX}")
option(BUILD_FORTRAN_CLIENTS "Build the ROCTx Fortran binding tests" ON)

# dev, as the HIP Fortran bindings: the two install side by side.
set(ROCTX_FORTRAN_INSTALL_COMPONENT
    "dev"
    CACHE STRING
    "Install component of the ROCTx Fortran bindings"
)

if(NOT BUILD_FORTRAN_BINDINGS)
    return()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmFortranCompiler.cmake")
if(NOT CMAKE_Fortran_COMPILER)
    message(
        STATUS
        "ROCTx: no Fortran compiler found, skipping the Fortran bindings"
    )
    return()
endif()

enable_language(Fortran)

# Before the target: its INSTALL_INTERFACE include directory uses CMAKE_INSTALL_*.
include(GNUInstallDirs)

add_library(roctx_fortran STATIC roctx.F90)
add_library(roc::roctx_fortran ALIAS roctx_fortran)

# Each module compiles to one object, and a linker extracts whole objects from
# an archive, so a program that calls one wrapper links all of them. One section
# per procedure lets a program linked with -Wl,--gc-sections keep only the
# wrappers it calls, and still link when another wrapper names a C function the
# installed library does not export. Only these compilers split the sections
# (the Intel ones accept the flags but do not), and not every version takes the
# flags: the amdflang of ROCm 7.13 rejects them.
include(CheckFortranCompilerFlag)
check_fortran_compiler_flag(
    -ffunction-sections
    ROCTX_FORTRAN_HAVE_FUNCTION_SECTIONS
)
check_fortran_compiler_flag(-fdata-sections ROCTX_FORTRAN_HAVE_DATA_SECTIONS)
if(ROCTX_FORTRAN_HAVE_FUNCTION_SECTIONS AND ROCTX_FORTRAN_HAVE_DATA_SECTIONS)
    target_compile_options(
        roctx_fortran
        PRIVATE
            "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,LLVMFlang,Flang,NVHPC>:-ffunction-sections;-fdata-sections>"
    )
endif()

# A .mod is compiler-specific, so artifacts install per compiler, named as the
# user types it. Keep in step with roctx-fortran-config.cmake.in, which
# recomputes this name.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(_fc_name "gfortran")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang") # Flang, LLVMFlang
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

# A .mod is tied to the compiler version too: sites with several set this.
set(ROCTX_FORTRAN_COMPILER_DIR
    "${_fc_name}"
    CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under"
)

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(
    roctx_fortran
    PROPERTIES
        Fortran_MODULE_DIRECTORY "${_moddir}"
        Fortran_PREPROCESS ON
        Fortran_FORMAT FREE
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
    roctx_fortran
    PUBLIC
        "$<BUILD_INTERFACE:${_moddir}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCTX_FORTRAN_COMPILER_DIR}>"
)

# Link interface only. rocprofiler-sdk-roctx-config.cmake needs C or CXX
# enabled; QUIET because the bindings build without ROCTx.
enable_language(C)
find_package(rocprofiler-sdk-roctx QUIET)

# Never left empty: it is @-substituted into a bare if() in the installed config.
if(TARGET rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
    target_link_libraries(
        roctx_fortran
        INTERFACE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
    )
    set(ROCTX_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
    message(
        STATUS
        "ROCTx: rocprofiler-sdk-roctx was not found, so the Fortran bindings "
        "carry no link interface; a consumer will have to link it "
        "(-lrocprofiler-sdk-roctx) itself."
    )
    set(ROCTX_FORTRAN_NEEDS_C_PACKAGE FALSE)
endif()

# The uninstrumented Fortran executables cannot link an ASan
# librocprofiler-sdk-roctx. ROCPROFILER_MEMCHECK is rocprofiler-sdk's switch,
# THEROCK_SANITIZER (ASAN, HOST_ASAN) TheRock's.
if(
    BUILD_FORTRAN_CLIENTS
    AND (
        ROCPROFILER_MEMCHECK STREQUAL "AddressSanitizer"
        OR THEROCK_SANITIZER MATCHES "ASAN"
    )
)
    message(
        STATUS
        "ROCTx: address sanitizer is enabled, skipping the Fortran binding tests."
    )
    set(BUILD_FORTRAN_CLIENTS OFF)
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
    ARCHIVE
        DESTINATION
            "${CMAKE_INSTALL_LIBDIR}/fortran/${ROCTX_FORTRAN_COMPILER_DIR}"
        COMPONENT ${ROCTX_FORTRAN_INSTALL_COMPONENT}
)
install(
    DIRECTORY "${_moddir}/"
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCTX_FORTRAN_COMPILER_DIR}"
    COMPONENT ${ROCTX_FORTRAN_INSTALL_COMPONENT}
    FILES_MATCHING
    PATTERN "*.mod"
)

# Under share/, not include/fortran/, whose entries the config lists as compilers.
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/roctx.F90"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/roctx/fortran"
    COMPONENT ${ROCTX_FORTRAN_INSTALL_COMPONENT}
)
install(
    EXPORT roctx-fortran-targets
    FILE roctx-fortran-targets.cmake
    NAMESPACE roc::
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT ${ROCTX_FORTRAN_INSTALL_COMPONENT}
)

set(ROCTX_FORTRAN_LIBDIR "${CMAKE_INSTALL_LIBDIR}/fortran")
set(ROCTX_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/roctx-fortran-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config.cmake"
    INSTALL_DESTINATION "${_fortran_pkgdir}"
    # Not ROCTX_FORTRAN_COMPILER_DIR: it is a directory name, not a path.
    PATH_VARS ROCTX_FORTRAN_LIBDIR ROCTX_FORTRAN_INCLUDEDIR
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config.cmake"
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT ${ROCTX_FORTRAN_INSTALL_COMPONENT}
)

# From rocprofiler-sdk's source VERSION file, which versions
# rocprofiler-sdk-roctx. Not rocprofiler-sdk-roctx_VERSION: standalone, that is
# the INSTALLED ROCTx's version.
set(_roctx_version_file
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/rocprofiler-sdk/VERSION"
)
set(_fortran_version "")
if(EXISTS "${_roctx_version_file}")
    file(STRINGS "${_roctx_version_file}" _roctx_v LIMIT_COUNT 1)
    string(
        REGEX MATCH
        "^[0-9]+\\.[0-9]+\\.[0-9]+"
        _fortran_version
        "${_roctx_v}"
    )
    unset(_roctx_v)
endif()

if(_fortran_version)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config-version.cmake"
        VERSION "${_fortran_version}"
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config-version.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT ${ROCTX_FORTRAN_INSTALL_COMPONENT}
    )
else()
    message(
        STATUS
        "ROCTx: could not determine the version from rocprofiler-sdk's VERSION "
        "file, so no roctx-fortran-config-version.cmake is installed and a "
        "versioned find_package(roctx-fortran) will not match."
    )
endif()
