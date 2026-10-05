# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Fortran bindings for HIP. hip.F90 is generated from the HIP headers by the
# rocm-fortran generator; do not edit it by hand. clr adds this directory
# through HIP_COMMON_DIR, as the headers live here and the runtime in clr.

cmake_minimum_required(VERSION 3.19)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(hip-fortran LANGUAGES NONE)
endif()

# "${UNIX}": ROCm ships no Fortran compiler on Windows, where the probe would
# pick up a stray gfortran that enable_language(Fortran) then rejects.
option(BUILD_FORTRAN_BINDINGS "Build the HIP Fortran bindings" "${UNIX}")

option(BUILD_FORTRAN_CLIENTS "Build the HIP Fortran clients and binding tests" ON)

# Both options are shared by every library in the monorepo; the HIP_ names win.
if(DEFINED HIP_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${HIP_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED HIP_BUILD_FORTRAN_CLIENTS)
  set(BUILD_FORTRAN_CLIENTS ${HIP_BUILD_FORTRAN_CLIENTS})
endif()

# For HIP, none and assumed-shape select the same module: hip.F90 has no
# USE_ASSUMED_SHAPE guard. none stays accepted because the cache entry is
# shared with the other ROCm Fortran bindings.
set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIP_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

# A deliberate ROCM_HAVE_FORTRAN=OFF must not be overridden by a fresh probe.
if(NOT DEFINED ROCM_HAVE_FORTRAN)
  if(NOT DEFINED ROCM_SYSTEMS_ROOT)
    get_filename_component(ROCM_SYSTEMS_ROOT
                           "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
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
    set(_have_fortran TRUE)
  else()
    set(_have_fortran FALSE)
  endif()
endif()

if(NOT _have_fortran)
  message(STATUS "HIP: no Fortran compiler found, skipping the Fortran bindings")
  return()
endif()

enable_language(Fortran)

include(GNUInstallDirs)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

add_library(hip_fortran STATIC hip.F90)
add_library(hip::hip_fortran ALIAS hip_fortran)

# A .mod is compiler-specific, so artifacts install per compiler, named as the
# user types it. Must match the mapping in hip-fortran-config.cmake.in.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(_fc_name "gfortran")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")     # Flang, LLVMFlang
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

# Not version-specific: a site installing two versions of one compiler sets it.
set(HIP_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(hip_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  Fortran_PREPROCESS ON
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(hip_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

# No line-length flag: the generated source stays within 131 columns.

target_include_directories(hip_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}>")

# In-tree the runtime is the amdhip64 target, and find_package(hip) would be
# circular (and fatal under TheRock's dependency provider). hip-config.cmake
# needs C or CXX enabled for FindThreads.
if(NOT TARGET hip::host AND NOT TARGET amdhip64)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hip QUIET)
endif()

# HIP_FORTRAN_NEEDS_C_PACKAGE tells the config package whether to find hip.
if(TARGET hip::host)
  target_link_libraries(hip_fortran INTERFACE hip::host)
  set(HIP_FORTRAN_NEEDS_C_PACKAGE TRUE)
elseif(TARGET amdhip64)
  # amdhip64 is in two export sets, which install(EXPORT) rejects in a link
  # interface, so the installed interface names hip::host instead.
  target_link_libraries(hip_fortran INTERFACE
    $<BUILD_INTERFACE:amdhip64>
    $<INSTALL_INTERFACE:hip::host>)
  set(HIP_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
  set(HIP_FORTRAN_NEEDS_C_PACKAGE FALSE)
  message(STATUS
    "HIP: the runtime was not found, so the Fortran bindings carry no link "
    "interface; a consumer will have to link it (-lamdhip64) itself.")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIP_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
endif()

# clr has no client switch, so in-tree the GPU tests also need BUILD_TESTING;
# not written back to the shared cache entry.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND BUILD_FORTRAN_CLIENTS
   AND NOT BUILD_TESTING)
  message(STATUS
    "HIP: testing is not enabled, skipping the Fortran binding tests. Pass "
    "-DBUILD_TESTING=ON to build them.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

if(BUILD_FORTRAN_CLIENTS)
  # In-tree, ctest at the build root also needs testing enabled at top level.
  enable_testing()
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/hip-fortran")

# dev is the component clr's CPack packages HIP's headers and config under.
set(HIP_FORTRAN_INSTALL_COMPONENT "dev" CACHE STRING
    "Install component of the HIP Fortran bindings")

install(TARGETS hip_fortran
        EXPORT hip-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}"
                COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT})
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
        FILES_MATCHING PATTERN "*.mod")
install(EXPORT hip-fortran-targets
        FILE hip-fortran-targets.cmake
        NAMESPACE hip::
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT})

set(HIP_FORTRAN_LIBDIR     "${CMAKE_INSTALL_LIBDIR}/fortran")
set(HIP_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/hip-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
  INSTALL_DESTINATION "${_fortran_pkgdir}"
  # Not HIP_FORTRAN_COMPILER_DIR: it is a bare name, not a path under the prefix.
  PATH_VARS HIP_FORTRAN_LIBDIR HIP_FORTRAN_INCLUDEDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT})

# From the source VERSION file, as hipamd does; hip_VERSION is the INSTALLED HIP.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION")
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" _hip_vlist REGEX "^[0-9]+")
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
    COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT})
else()
  message(STATUS
    "HIP: could not determine the version from the VERSION file, so no "
    "hip-fortran-config-version.cmake is installed and a versioned "
    "find_package(hip-fortran) will not match.")
endif()

# Under share/, not include/fortran/, whose entries the config lists as compilers.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hip.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/hip/fortran"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT})
