# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Compiler and linker security hardening flags for the ROCm runtime components.
#
# This module is intentionally self-contained and duplicated per project: every
# project under rocm-systems/projects configures standalone and is mirrored into
# its own repository, so there is no shared cmake directory to include from.
#
# Usage:
#
#   list(APPEND CMAKE_MODULE_PATH "<dir containing this file>")
#   include(ROCmHardening)
#   rocm_hardening_probe()
#
# then either apply the flags to a target:
#
#   rocm_harden_target(mylib)
#
# or splice them into an existing flag variable, for the components that still
# accumulate flags in strings and lists of their own:
#
#   list(APPEND MY_CXX_FLAGS ${ROCM_HARDENING_CXX_FLAGS})
#   set(MY_LINK_FLAGS "${MY_LINK_FLAGS} ${ROCM_HARDENING_LINK_FLAGS_STR}")
#
# rocm_hardening_probe() populates these globally visible variables, each of
# which is empty when hardening is disabled or unsupported:
#
#   ROCM_HARDENING_C_FLAGS         supported C compile flags, as a list
#   ROCM_HARDENING_CXX_FLAGS       supported C++ compile flags, as a list
#   ROCM_HARDENING_LINK_FLAGS      supported link flags, as a list
#   ROCM_HARDENING_LINK_FLAGS_STR  the same link flags, space separated

include_guard(GLOBAL)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

option(ROCM_ENABLE_HARDENING
       "Enable compiler and linker security hardening flags" ON)

# CheckLinkerFlag only exists from CMake 3.18. On older CMake fall back to
# check_c_compiler_flag, whose try_compile also links, so -Wl, flags still reach
# the linker -- but without the -Werror probe, because the compile step of that
# same invocation reports them as unused arguments.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  include(CheckLinkerFlag)
endif()

function(_rocm_hardening_cache_var_name prefix flag out_var)
  string(REGEX REPLACE "[^A-Za-z0-9]+" "_" _name "${prefix}_${flag}")
  string(TOUPPER "${_name}" _name)
  set(${out_var} "ROCM_HARDENING_HAVE_${_name}" PARENT_SCOPE)
endfunction()

function(_rocm_hardening_languages out_var)
  get_property(_enabled GLOBAL PROPERTY ENABLED_LANGUAGES)
  set(_result "")
  foreach(_lang IN ITEMS C CXX)
    if("${_lang}" IN_LIST _enabled)
      list(APPEND _result "${_lang}")
    endif()
  endforeach()
  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

# A flag the compiler merely warns about must count as unsupported, so probe
# with -Werror whenever -Werror itself is accepted.
function(_rocm_hardening_check_compile_flag lang flag out_supported)
  _rocm_hardening_cache_var_name("${lang}" "${flag}" _cache_var)

  _rocm_hardening_cache_var_name("${lang}" "-Werror" _werror_var)
  if("${lang}" STREQUAL "C")
    check_c_compiler_flag("-Werror" ${_werror_var})
  else()
    check_cxx_compiler_flag("-Werror" ${_werror_var})
  endif()

  set(_probe "${flag}")
  if(${_werror_var})
    set(_probe "${flag} -Werror")
  endif()

  if("${lang}" STREQUAL "C")
    check_c_compiler_flag("${_probe}" ${_cache_var})
  else()
    check_cxx_compiler_flag("${_probe}" ${_cache_var})
  endif()

  set(${out_supported} "${${_cache_var}}" PARENT_SCOPE)
endfunction()

function(_rocm_hardening_check_link_flag flag out_supported)
  _rocm_hardening_cache_var_name("LINK" "${flag}" _cache_var)

  _rocm_hardening_languages(_languages)
  if(NOT _languages)
    set(${out_supported} FALSE PARENT_SCOPE)
    return()
  endif()
  list(GET _languages 0 _lang)

  if(COMMAND check_linker_flag)
    check_linker_flag(${_lang} "${flag}" ${_cache_var})
  elseif("${_lang}" STREQUAL "C")
    check_c_compiler_flag("${flag}" ${_cache_var})
  else()
    check_cxx_compiler_flag("${flag}" ${_cache_var})
  endif()

  set(${out_supported} "${${_cache_var}}" PARENT_SCOPE)
endfunction()

# The sanitizer switches in these components are spelled inconsistently: some
# are booleans, THEROCK_SANITIZER is a name such as ASAN or HOST_ASAN, and some
# arrive through the environment. if(<var>) is the wrong test for those, because
# CMake treats "NONE" as true and an unquoted string constant as false.
function(_rocm_hardening_is_enabled value out_var)
  string(TOUPPER "${value}" _value)
  if(_value STREQUAL "" OR _value STREQUAL "OFF" OR _value STREQUAL "NO"
     OR _value STREQUAL "FALSE" OR _value STREQUAL "N" OR _value STREQUAL "0"
     OR _value STREQUAL "NONE" OR _value STREQUAL "IGNORE"
     OR _value STREQUAL "NOTFOUND" OR _value MATCHES "-NOTFOUND$")
    set(${out_var} FALSE PARENT_SCOPE)
  else()
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

# _FORTIFY_SOURCE is only meaningful with optimization enabled: without it glibc
# emits "#warning _FORTIFY_SOURCE requires compiling with optimization", which
# becomes a hard failure in the components that build with -Werror. It also
# duplicates and conflicts with the sanitizer interceptors, so it stays off for
# sanitizer builds. Neither condition is detectable by a compiler probe, because
# the probe source pulls in no libc header.
function(_rocm_hardening_fortify_source_allowed out_var)
  foreach(_sanitizer IN ITEMS
          "${ADDRESS_SANITIZER}" "${UNDEFINED_SANITIZER}" "${THEROCK_SANITIZER}"
          "$ENV{ADDRESS_SANITIZER}" "$ENV{UNDEFINED_SANITIZER}"
          "$ENV{THEROCK_SANITIZER}")
    _rocm_hardening_is_enabled("${_sanitizer}" _enabled)
    if(_enabled)
      set(${out_var} FALSE PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(_flags "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS}")
  if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _config)
    set(_flags "${_flags} ${CMAKE_C_FLAGS_${_config}} ${CMAKE_CXX_FLAGS_${_config}}")
  endif()

  if(_flags MATCHES "-O[123s]")
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(rocm_hardening_probe)
  # Components may each probe so they also work when configured standalone.
  # A global property resets every configure run, unlike a cache entry, so the
  # flags are still recomputed when ROCM_ENABLE_HARDENING or the build type
  # changes.
  get_property(_already_probed GLOBAL PROPERTY ROCM_HARDENING_PROBED)
  if(_already_probed)
    return()
  endif()
  set_property(GLOBAL PROPERTY ROCM_HARDENING_PROBED TRUE)

  set(_c_flags "")
  set(_cxx_flags "")
  set(_link_flags "")
  set(_skip_reason "")

  if(NOT ROCM_ENABLE_HARDENING)
    set(_skip_reason "ROCM_ENABLE_HARDENING is OFF")
  elseif(MSVC
         OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
         OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # These are GCC/Clang spellings targeting ELF. MSVC and clang-cl enable the
    # equivalent mitigations (/GS, /NXCOMPAT, /DYNAMICBASE) by default.
    set(_skip_reason "not applicable to MSVC-style toolchains")
  elseif(NOT UNIX)
    set(_skip_reason "not applicable to non-UNIX targets")
  endif()

  if(NOT _skip_reason)
    _rocm_hardening_languages(_languages)
    _rocm_hardening_fortify_source_allowed(_fortify_allowed)

    foreach(_lang IN LISTS _languages)
      set(_lang_flags "")

      foreach(_flag IN ITEMS "-fstack-protector-strong" "-fstack-clash-protection")
        _rocm_hardening_check_compile_flag("${_lang}" "${_flag}" _supported)
        if(_supported)
          list(APPEND _lang_flags "${_flag}")
        endif()
      endforeach()

      # Level 3 is preferred, and tried first: Ubuntu 24.04's GCC already
      # predefines _FORTIFY_SOURCE 3 when optimizing, so pinning level 2 there
      # would be a downgrade. Toolchains that cannot do level 3 fail the probe
      # and fall back. -U comes first so an already predefined level does not
      # turn a redefinition warning into an error under -Werror.
      if(_fortify_allowed)
        foreach(_level IN ITEMS 3 2)
          _rocm_hardening_check_compile_flag(
            "${_lang}" "-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=${_level}" _supported)
          if(_supported)
            list(APPEND _lang_flags "-U_FORTIFY_SOURCE" "-D_FORTIFY_SOURCE=${_level}")
            break()
          endif()
        endforeach()
      endif()

      if("${_lang}" STREQUAL "C")
        set(_c_flags ${_lang_flags})
      else()
        set(_cxx_flags ${_lang_flags})
      endif()
    endforeach()

    foreach(_flag IN ITEMS "-Wl,-z,relro" "-Wl,-z,now" "-Wl,-z,noexecstack")
      _rocm_hardening_check_link_flag("${_flag}" _supported)
      if(_supported)
        list(APPEND _link_flags "${_flag}")
      endif()
    endforeach()
  endif()

  string(REPLACE ";" " " _link_flags_str "${_link_flags}")

  set(ROCM_HARDENING_C_FLAGS "${_c_flags}" CACHE INTERNAL
      "ROCm hardening: supported C compile flags")
  set(ROCM_HARDENING_CXX_FLAGS "${_cxx_flags}" CACHE INTERNAL
      "ROCm hardening: supported C++ compile flags")
  set(ROCM_HARDENING_LINK_FLAGS "${_link_flags}" CACHE INTERNAL
      "ROCm hardening: supported link flags")
  set(ROCM_HARDENING_LINK_FLAGS_STR "${_link_flags_str}" CACHE INTERNAL
      "ROCm hardening: supported link flags, space separated")

  if(_skip_reason)
    message(STATUS "ROCm hardening: disabled (${_skip_reason})")
  else()
    message(STATUS "ROCm hardening: C   ${_c_flags}")
    message(STATUS "ROCm hardening: CXX ${_cxx_flags}")
    message(STATUS "ROCm hardening: link ${_link_flags_str}")
    if(NOT _fortify_allowed)
      message(STATUS "ROCm hardening: _FORTIFY_SOURCE skipped "
                     "(needs optimization and no sanitizer)")
    endif()
  endif()
endfunction()

# Applies the probed flags to a single target. Compile flags are attached per
# language so they never leak into device compilation, which uses the HIP
# language or a separate clang invocation targeting amdgcn-amd-amdhsa.
function(rocm_harden_target target)
  if(NOT ROCM_ENABLE_HARDENING)
    return()
  endif()

  if(NOT TARGET ${target})
    message(FATAL_ERROR "rocm_harden_target: '${target}' is not a target")
  endif()

  get_target_property(_type ${target} TYPE)
  if("${_type}" STREQUAL "INTERFACE_LIBRARY")
    return()
  endif()

  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)

  if(ROCM_HARDENING_C_FLAGS)
    target_compile_options(${target} PRIVATE
                           "$<$<COMPILE_LANGUAGE:C>:${ROCM_HARDENING_C_FLAGS}>")
  endif()
  if(ROCM_HARDENING_CXX_FLAGS)
    target_compile_options(${target} PRIVATE
                           "$<$<COMPILE_LANGUAGE:CXX>:${ROCM_HARDENING_CXX_FLAGS}>")
  endif()

  # Static archives and object libraries are not linked, so link options would
  # be silently dropped; they reach the shared library that consumes them.
  if(ROCM_HARDENING_LINK_FLAGS
     AND NOT "${_type}" STREQUAL "STATIC_LIBRARY"
     AND NOT "${_type}" STREQUAL "OBJECT_LIBRARY")
    if(COMMAND target_link_options)
      target_link_options(${target} PRIVATE ${ROCM_HARDENING_LINK_FLAGS})
    else()
      set_property(TARGET ${target} APPEND_STRING PROPERTY
                   LINK_FLAGS " ${ROCM_HARDENING_LINK_FLAGS_STR}")
    endif()
  endif()
endfunction()
