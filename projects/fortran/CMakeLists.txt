# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Fortran bindings for the rocm-systems libraries: the `hip` and `roctx` modules.
# hip.F90 and roctx.F90 are generated from the HIP and ROCTx headers by
# rocm-fortran-generator; do not edit them by hand. A standalone project, built
# against an installed HIP (find_package(hip)) and ROCTx
# (find_package(rocprofiler-sdk-roctx)).

cmake_minimum_required(VERSION 3.19)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(hip-fortran LANGUAGES NONE)
endif()

# "${UNIX}": ROCm ships no Fortran compiler on Windows, where the probe would
# pick up a stray gfortran that enable_language(Fortran) then rejects.
option(
    BUILD_FORTRAN_BINDINGS
    "Build the HIP and ROCTx Fortran bindings"
    "${UNIX}"
)

option(BUILD_FORTRAN_CLIENTS "Build the HIP and ROCTx Fortran binding tests" ON)

# Both options are shared by every library in the monorepo; the HIP_ and ROCTX_
# names win for their library.
foreach(_lib IN ITEMS HIP ROCTX)
    set(_${_lib}_bindings ${BUILD_FORTRAN_BINDINGS})
    if(DEFINED ${_lib}_BUILD_FORTRAN_BINDINGS)
        set(_${_lib}_bindings ${${_lib}_BUILD_FORTRAN_BINDINGS})
    endif()
    set(_${_lib}_clients ${BUILD_FORTRAN_CLIENTS})
    if(DEFINED ${_lib}_BUILD_FORTRAN_CLIENTS)
        set(_${_lib}_clients ${${_lib}_BUILD_FORTRAN_CLIENTS})
    endif()
endforeach()

set(FORTRAN_ARRAY_INTERFACES
    "assumed-shape"
    CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank"
)
set_property(
    CACHE FORTRAN_ARRAY_INTERFACES
    PROPERTY STRINGS none assumed-shape assumed-rank
)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HIP_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
    set(ROCTX_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT _HIP_bindings AND NOT _ROCTX_bindings)
    return()
endif()

# A deliberate ROCM_HAVE_FORTRAN=OFF must not be overridden by a fresh probe.
if(NOT DEFINED ROCM_HAVE_FORTRAN)
    if(NOT DEFINED ROCM_SYSTEMS_ROOT)
        get_filename_component(
            ROCM_SYSTEMS_ROOT
            "${CMAKE_CURRENT_SOURCE_DIR}/../.."
            ABSOLUTE
        )
    endif()
    include("${ROCM_SYSTEMS_ROOT}/shared/cmake/ROCmFortran.cmake" OPTIONAL)
endif()

# Fallback for a build extracted from the monorepo, where the shared module is
# absent. Mirrors shared/cmake/ROCmFortran.cmake; keep the two in sync.
if(DEFINED ROCM_HAVE_FORTRAN)
    set(_have_fortran ${ROCM_HAVE_FORTRAN})
else()
    include(CheckLanguage)

    set(_rocm_warn_toolchain_var "${ROCM_WARN_TOOLCHAIN_VAR}")
    set(_rocm_error_toolchain_var "${ROCM_ERROR_TOOLCHAIN_VAR}")
    set(ROCM_WARN_TOOLCHAIN_VAR OFF)
    set(ROCM_ERROR_TOOLCHAIN_VAR OFF)

    if(NOT CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
        set(_rocm_fc_hints)
        foreach(
            _rocm_fc_cc
            "${CMAKE_CXX_COMPILER}"
            "${CMAKE_C_COMPILER}"
            "${CMAKE_HIP_COMPILER}"
        )
            if(_rocm_fc_cc)
                get_filename_component(_rocm_fc_dir "${_rocm_fc_cc}" DIRECTORY)
                list(APPEND _rocm_fc_hints "${_rocm_fc_dir}")
            endif()
        endforeach()
        foreach(_rocm_fc_root "${ROCM_PATH}" "$ENV{ROCM_PATH}" "/opt/rocm")
            if(_rocm_fc_root)
                list(
                    APPEND _rocm_fc_hints
                    "${_rocm_fc_root}/bin"
                    "${_rocm_fc_root}/lib/llvm/bin"
                )
            endif()
        endforeach()
        unset(_rocm_amdflang CACHE)
        find_program(
            _rocm_amdflang
            NAMES amdflang
            HINTS ${_rocm_fc_hints}
            NO_DEFAULT_PATH
        )
        if(_rocm_amdflang)
            execute_process(
                COMMAND "${_rocm_amdflang}" --version
                RESULT_VARIABLE _rocm_fc_rc
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(_rocm_fc_rc EQUAL 0)
                set(CMAKE_Fortran_COMPILER
                    "${_rocm_amdflang}"
                    CACHE FILEPATH
                    "Fortran compiler used for the ROCm Fortran bindings"
                    FORCE
                )
            endif()
            unset(_rocm_fc_rc)
        endif()
        unset(_rocm_amdflang CACHE)
        unset(_rocm_fc_hints)
        unset(_rocm_fc_cc)
        unset(_rocm_fc_dir)
        unset(_rocm_fc_root)
    endif()

    check_language(Fortran)
    set(ROCM_WARN_TOOLCHAIN_VAR "${_rocm_warn_toolchain_var}")
    set(ROCM_ERROR_TOOLCHAIN_VAR "${_rocm_error_toolchain_var}")
    unset(_rocm_warn_toolchain_var)
    unset(_rocm_error_toolchain_var)

    if(CMAKE_Fortran_COMPILER)
        set(_have_fortran TRUE)
    else()
        set(_have_fortran FALSE)
    endif()
endif()

if(NOT _have_fortran)
    message(
        STATUS
        "HIP, ROCTx: no Fortran compiler found, skipping the Fortran bindings"
    )
    return()
endif()

enable_language(Fortran)

include(GNUInstallDirs)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
    message(
        FATAL_ERROR
        "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
        "not '${FORTRAN_ARRAY_INTERFACES}'."
    )
endif()

# A .mod is compiler-specific, so artifacts install per compiler, named as the
# user types it. Must match the mapping in hip-fortran-config.cmake.in and
# roctx-fortran-config.cmake.in.
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

include(CMakePackageConfigHelpers)

# dev: the component HIP's headers and CMake config install under.
set(HIP_FORTRAN_INSTALL_COMPONENT
    "dev"
    CACHE STRING
    "Install component of the HIP and ROCTx Fortran bindings"
)

if(_HIP_bindings)
    include("${CMAKE_CURRENT_SOURCE_DIR}/hip.cmake")
endif()
if(_ROCTX_bindings)
    include("${CMAKE_CURRENT_SOURCE_DIR}/roctx.cmake")
endif()

if(
    (TARGET hip_fortran AND _HIP_clients)
    OR (TARGET roctx_fortran AND _ROCTX_clients)
)
    enable_testing()
    add_subdirectory(test)
endif()
