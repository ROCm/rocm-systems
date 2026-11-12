# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Runs check_language(Fortran), preferring ROCm's amdflang over the first
# Fortran compiler on PATH (often gfortran). An explicit CMAKE_Fortran_COMPILER
# or FC wins. CMAKE_Fortran_COMPILER is left false when none is found. The same
# file sits in each binding under bindings/fortran/; keep the copies in step.

include(CheckLanguage)

# rocm-cmake's ROCMChecks watches CMAKE_Fortran_COMPILER and fails under
# ROCM_ERROR_TOOLCHAIN_VAR=ON; silence it across the writes.
set(_rocm_fc_warn_toolchain_var "${ROCM_WARN_TOOLCHAIN_VAR}")
set(_rocm_fc_error_toolchain_var "${ROCM_ERROR_TOOLCHAIN_VAR}")
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
    # Uncached, so a changed ROCM_PATH is honoured on reconfigure.
    unset(_rocm_amdflang CACHE)
    find_program(
        _rocm_amdflang
        NAMES amdflang
        HINTS ${_rocm_fc_hints}
        NO_DEFAULT_PATH
    )
    # A ROCm tree can carry a dangling amdflang symlink.
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
    endif()
    unset(_rocm_amdflang CACHE)
endif()

check_language(Fortran)

set(ROCM_WARN_TOOLCHAIN_VAR "${_rocm_fc_warn_toolchain_var}")
set(ROCM_ERROR_TOOLCHAIN_VAR "${_rocm_fc_error_toolchain_var}")
unset(_rocm_fc_warn_toolchain_var)
unset(_rocm_fc_error_toolchain_var)
unset(_rocm_fc_hints)
unset(_rocm_fc_cc)
unset(_rocm_fc_dir)
unset(_rocm_fc_root)
unset(_rocm_fc_rc)
