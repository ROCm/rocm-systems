# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Fortran bindings for HIP: the `hip` module, hip::hip_fortran and the
# hip-fortran package. hip.F90 is generated from the HIP headers by
# rocm-fortran-generator; do not edit it by hand, regenerate it. A standalone
# project, built against an installed HIP (find_package(hip)).

cmake_minimum_required(VERSION 3.19)

project(hip-fortran LANGUAGES NONE)

# "${UNIX}": ROCm ships no Fortran compiler on Windows, where the probe would
# pick up a stray gfortran that enable_language(Fortran) then rejects.
option(BUILD_FORTRAN_BINDINGS "Build the HIP Fortran bindings" "${UNIX}")
option(BUILD_FORTRAN_CLIENTS "Build the HIP Fortran binding tests" ON)

set(FORTRAN_ARRAY_INTERFACES
    "assumed-shape"
    CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank"
)
set_property(
    CACHE FORTRAN_ARRAY_INTERFACES
    PROPERTY STRINGS none assumed-shape assumed-rank
)

# dev: the component HIP's headers and CMake config install under.
set(HIP_FORTRAN_INSTALL_COMPONENT
    "dev"
    CACHE STRING
    "Install component of the HIP Fortran bindings"
)

if(NOT BUILD_FORTRAN_BINDINGS)
    return()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmFortranCompiler.cmake")
if(NOT CMAKE_Fortran_COMPILER)
    message(
        STATUS
        "HIP: no Fortran compiler found, skipping the Fortran bindings"
    )
    return()
endif()

enable_language(Fortran)

# Before the target: its INSTALL_INTERFACE include directory uses CMAKE_INSTALL_*.
include(GNUInstallDirs)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
    message(
        FATAL_ERROR
        "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
        "not '${FORTRAN_ARRAY_INTERFACES}'."
    )
endif()

add_library(hip_fortran STATIC hip.F90)
add_library(hip::hip_fortran ALIAS hip_fortran)

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
    HIP_FORTRAN_HAVE_FUNCTION_SECTIONS
)
check_fortran_compiler_flag(-fdata-sections HIP_FORTRAN_HAVE_DATA_SECTIONS)
if(HIP_FORTRAN_HAVE_FUNCTION_SECTIONS AND HIP_FORTRAN_HAVE_DATA_SECTIONS)
    target_compile_options(
        hip_fortran
        PRIVATE
            "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,LLVMFlang,Flang,NVHPC>:-ffunction-sections;-fdata-sections>"
    )
endif()

# A .mod is compiler-specific, so artifacts install per compiler, named as the
# user types it. Keep in step with hip-fortran-config.cmake.in, which recomputes
# this name.
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
set(HIP_FORTRAN_COMPILER_DIR
    "${_fc_name}"
    CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under"
)

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(
    hip_fortran
    PROPERTIES
        Fortran_MODULE_DIRECTORY "${_moddir}"
        Fortran_PREPROCESS ON
        Fortran_FORMAT FREE
        POSITION_INDEPENDENT_CODE ON
)

if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
    target_compile_definitions(hip_fortran PRIVATE USE_ASSUMED_SHAPE=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
    target_compile_definitions(hip_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

# No line-length flag: the generated source stays within 131 columns.

target_include_directories(
    hip_fortran
    PUBLIC
        "$<BUILD_INTERFACE:${_moddir}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}>"
)

# Link interface only. hip-config.cmake needs C or CXX enabled for FindThreads;
# QUIET because the bindings build without HIP.
enable_language(C)
find_package(hip QUIET)

# Never left empty: it is @-substituted into a bare if() in the installed config.
if(TARGET hip::host)
    target_link_libraries(hip_fortran INTERFACE hip::host)
    set(HIP_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
    message(
        STATUS
        "HIP: the runtime was not found, so the Fortran bindings carry no link "
        "interface; a consumer will have to link it (-lamdhip64) itself."
    )
    set(HIP_FORTRAN_NEEDS_C_PACKAGE FALSE)
endif()

# The uninstrumented Fortran executables cannot link an ASan libamdhip64.
# ENABLE_SANITIZER (ASAN, HOST_ASAN) and THEROCK_SANITIZER are the switches CLR
# reads.
if(
    BUILD_FORTRAN_CLIENTS
    AND (ENABLE_SANITIZER MATCHES "ASAN" OR THEROCK_SANITIZER MATCHES "ASAN")
)
    message(
        STATUS
        "HIP: address sanitizer is enabled, skipping the Fortran binding tests."
    )
    set(BUILD_FORTRAN_CLIENTS OFF)
endif()

if(BUILD_FORTRAN_CLIENTS)
    enable_testing()
    add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/hip-fortran")

install(
    TARGETS hip_fortran
    EXPORT hip-fortran-targets
    ARCHIVE
        DESTINATION
            "${CMAKE_INSTALL_LIBDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)
install(
    DIRECTORY "${_moddir}/"
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
    FILES_MATCHING
    PATTERN "*.mod"
)

# Under share/, not include/fortran/, whose entries the config lists as compilers.
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/hip.F90"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/hip/fortran"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)
install(
    EXPORT hip-fortran-targets
    FILE hip-fortran-targets.cmake
    NAMESPACE hip::
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)

set(HIP_FORTRAN_LIBDIR "${CMAKE_INSTALL_LIBDIR}/fortran")
set(HIP_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-fortran-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
    INSTALL_DESTINATION "${_fortran_pkgdir}"
    # Not HIP_FORTRAN_COMPILER_DIR: it is a directory name, not a path.
    PATH_VARS HIP_FORTRAN_LIBDIR HIP_FORTRAN_INCLUDEDIR
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)

# From HIP's source VERSION file, as hipamd does. Not hip_VERSION: standalone,
# that is the INSTALLED HIP's version.
set(_hip_version_file
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hip/VERSION"
)
set(_fortran_version "")
if(EXISTS "${_hip_version_file}")
    file(STRINGS "${_hip_version_file}" _hip_vlist REGEX "^[0-9]+")
    list(LENGTH _hip_vlist _hip_vn)
    if(_hip_vn GREATER_EQUAL 3)
        list(GET _hip_vlist 0 1 2 _hip_v)
        list(JOIN _hip_v "." _fortran_version)
    endif()
    unset(_hip_vlist)
endif()

if(_fortran_version)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config-version.cmake"
        VERSION "${_fortran_version}"
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config-version.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
    )
else()
    message(
        STATUS
        "HIP: could not determine the version from the VERSION file, so no "
        "hip-fortran-config-version.cmake is installed and a versioned "
        "find_package(hip-fortran) will not match."
    )
endif()
