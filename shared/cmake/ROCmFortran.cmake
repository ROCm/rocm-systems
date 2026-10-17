# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Probes once per configure for a Fortran compiler, preferring ROCm's amdflang,
# and publishes ROCM_HAVE_FORTRAN. Include it OPTIONAL: a project extracted from
# the monorepo must fall back to its own probe.

if(NOT DEFINED ROCM_HAVE_FORTRAN)

  # A global property, because sibling directories do not share variables.
  get_property(_rocm_fortran_probed GLOBAL PROPERTY ROCM_FORTRAN_PROBED)

  if(_rocm_fortran_probed)
    get_property(ROCM_HAVE_FORTRAN GLOBAL PROPERTY ROCM_FORTRAN_AVAILABLE)
  else()
    include(CheckLanguage)

    # rocm-cmake's ROCMChecks watches CMAKE_Fortran_COMPILER and fails the
    # configure under ROCM_ERROR_TOOLCHAIN_VAR=ON; silence it across the probe.
    set(_rocm_warn_toolchain_var "${ROCM_WARN_TOOLCHAIN_VAR}")
    set(_rocm_error_toolchain_var "${ROCM_ERROR_TOOLCHAIN_VAR}")
    set(ROCM_WARN_TOOLCHAIN_VAR OFF)
    set(ROCM_ERROR_TOOLCHAIN_VAR OFF)

    # Prefer amdflang over a system gfortran found first on PATH; an explicit
    # CMAKE_Fortran_COMPILER or FC wins. Mirrored in hip/fortran/CMakeLists.txt.
    if(NOT CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
      set(_rocm_fc_hints)
      foreach(_rocm_fc_cc "${CMAKE_CXX_COMPILER}" "${CMAKE_C_COMPILER}"
                          "${CMAKE_HIP_COMPILER}")
        if(_rocm_fc_cc)
          get_filename_component(_rocm_fc_dir "${_rocm_fc_cc}" DIRECTORY)
          list(APPEND _rocm_fc_hints "${_rocm_fc_dir}")
        endif()
      endforeach()
      foreach(_rocm_fc_root "${ROCM_PATH}" "$ENV{ROCM_PATH}" "/opt/rocm")
        if(_rocm_fc_root)
          list(APPEND _rocm_fc_hints "${_rocm_fc_root}/bin"
                                     "${_rocm_fc_root}/lib/llvm/bin")
        endif()
      endforeach()
      unset(_rocm_amdflang CACHE)
      find_program(_rocm_amdflang NAMES amdflang HINTS ${_rocm_fc_hints}
                   NO_DEFAULT_PATH)
      # A dangling amdflang symlink would otherwise surface only at enable_language.
      if(_rocm_amdflang)
        execute_process(COMMAND "${_rocm_amdflang}" --version
                        RESULT_VARIABLE _rocm_fc_rc
                        OUTPUT_QUIET ERROR_QUIET)
        if(_rocm_fc_rc EQUAL 0)
          set(CMAKE_Fortran_COMPILER "${_rocm_amdflang}" CACHE FILEPATH
              "Fortran compiler used for the ROCm Fortran bindings" FORCE)
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
      set(ROCM_HAVE_FORTRAN TRUE)
    else()
      set(ROCM_HAVE_FORTRAN FALSE)
      message(STATUS
        "ROCm: no Fortran compiler found - the Fortran bindings will be skipped")
    endif()
    set_property(GLOBAL PROPERTY ROCM_FORTRAN_PROBED TRUE)
    set_property(GLOBAL PROPERTY ROCM_FORTRAN_AVAILABLE "${ROCM_HAVE_FORTRAN}")
  endif()

  unset(_rocm_fortran_probed)

endif()
