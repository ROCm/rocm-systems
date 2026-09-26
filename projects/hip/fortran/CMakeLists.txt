# Fortran bindings for HIP.
#
# hip.F90 is GENERATED from the HIP C headers by the rocm-fortran generator and
# committed here, next to the headers it binds, so that a header change and its
# binding travel in the same pull request. Do not edit it by hand: regenerate it
# instead.
#
# The bindings are a single self-contained Fortran module, `hip`, carrying the
# interfaces, the enum constants, the derived types, the opaque handle types and
# the hipCheck status helper. A consumer writes one `use hip`.
#
# This directory is self-contained: it builds with no dependency on the HIP C
# target, because a Fortran binding compiles against nothing (it only declares
# bind(C) interfaces). The HIP runtime is required to LINK a final executable,
# and is carried as the target's link interface for that reason.
#
# HIP is the one ROCm project whose headers live apart from its build: the
# headers are here in projects/hip, the runtime that implements them is built in
# projects/clr. The bindings belong with the headers, so they sit here, and
# clr's CMakeLists adds this directory through HIP_COMMON_DIR.

cmake_minimum_required(VERSION 3.19)

# Configured directly (not as a subdirectory of the clr build), this directory is
# its own project, so that the bindings can be built and packaged on their own.
# In-tree, clr's own project() is already in effect.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(hip-fortran LANGUAGES NONE)
endif()

# ON by default but GUARDED: a site with no Fortran compiler must still configure
# and build the C library. The monorepo root already probes once and publishes
# ROCM_HAVE_FORTRAN; a standalone HIP build falls back to its own probe.
option(BUILD_FORTRAN_BINDINGS "Build the HIP Fortran bindings" ON)

# The bindings' own test suite. OFF by default and guarded the same way: asking
# for the tests when the bindings were not built is a skip, not an error.
option(BUILD_FORTRAN_TESTS "Build the HIP Fortran binding tests" OFF)

# BUILD_FORTRAN_BINDINGS and BUILD_FORTRAN_TESTS are ONE cache entry shared by
# every library in the monorepo, which is what you want for the common case:
# -DBUILD_FORTRAN_BINDINGS=OFF from the top turns the whole Fortran track off in
# a single flag. It is not what you want when one library has to be singled out,
# so each library also honours a name of its own, which wins when it is set.
if(DEFINED HIP_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${HIP_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED HIP_BUILD_FORTRAN_TESTS)
  set(BUILD_FORTRAN_TESTS ${HIP_BUILD_FORTRAN_TESTS})
endif()

# Which array-argument overloads the module exposes. One tri-state replaces the
# two booleans hipFORT used (USE_FPOINTER_INTERFACES / ASSUMED_RANK), because the
# two upper tiers cannot legally coexist: an assumed-rank dummy is not
# distinguishable by rank from the per-rank specifics, so assumed-rank REPLACES
# the per-rank variants rather than adding to them.
#
#   none           the raw type(c_ptr) surface only (Fortran 2003)
#   assumed-shape  + the per-rank array overloads (Fortran 2008), the default,
#                  and what the bindings shipped in ROCm are built with
#   assumed-rank   + the single dimension(..) overload per routine (Fortran 2018)
set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

# Report to the parent scope whether the bindings were really built. Distinct
# from ROCM_HAVE_FORTRAN, which reports only that a compiler exists: the
# compiler can be present while this directory is switched off.
# Only when there IS a parent: standalone, this directory is the top-level scope
# and PARENT_SCOPE warns under -Wdev rather than being a silent no-op.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIP_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

# Probing for a Fortran compiler is not free, and every library in a monorepo
# build would otherwise repeat it, so the answer is computed ONCE and published as
# ROCM_HAVE_FORTRAN. That is the single name, in both monorepos: a library from
# one is regularly built inside the other, and a repo-specific name would have
# each side reading a variable the other never sets.
#
# It is published either by a monorepo root that probes on the way in, or by the
# shared module below, which is how rocm-systems does it: it has no root
# CMakeLists, so the probe travels as a module each interested project includes,
# the same way shared/ctest/TestCategories.cmake is consumed here.
#
# OPTIONAL, and guarded on the answer not already being in hand: an answer
# published upstream must beat a fresh probe, or a deliberate
# ROCM_HAVE_FORTRAN=OFF would be silently contradicted.
if(NOT DEFINED ROCM_HAVE_FORTRAN)
  if(NOT DEFINED ROCM_SYSTEMS_ROOT)
    get_filename_component(ROCM_SYSTEMS_ROOT
                           "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
  endif()
  include("${ROCM_SYSTEMS_ROOT}/shared/cmake/ROCmFortran.cmake" OPTIONAL)
endif()

# The local probe is the last resort: this directory is also built extracted from
# the monorepo, where neither a root nor the shared module exists, and something
# that can be configured on its own has to answer this on its own.
if(DEFINED ROCM_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_HAVE_FORTRAN})
else()
  include(CheckLanguage)
  check_language(Fortran)
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

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

add_library(hip_fortran STATIC hip.F90)
add_library(hip::hip_fortran ALIAS hip_fortran)

# A .mod is not portable across compilers, so the artifacts are installed under a
# per-compiler directory rather than a shared one. Two compilers can then be
# installed side by side from the same package.
#
# The directory is named after the compiler a user actually types (amdflang,
# gfortran), not after CMake's compiler ID, so that the -I/-L paths in the
# documentation are the ones a raw (non-CMake) link line needs. Deriving it from
# the ID and mapping back, rather than taking the basename of
# CMAKE_Fortran_COMPILER, is what makes that survive the cases where the two
# differ: an MPI wrapper (mpif90 around gfortran), a versioned binary
# (gfortran-13), an absolute path, or the Cray wrapper (ftn). All four would give
# the wrong directory under a plain basename, and the .mod would then be filed
# under a name no consumer looks up.
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
  # Unknown compiler: fall back to the basename, lowercased, which is the best
  # guess available and still keeps two compilers from colliding.
  get_filename_component(_fc_name "${CMAKE_Fortran_COMPILER}" NAME_WE)
  string(TOLOWER "${_fc_name}" _fc_name)
endif()

# Strictly a .mod is tied to the compiler VERSION too, not just the compiler, so
# this name is deliberately lossy: gfortran 13 and gfortran 14 install to the
# same directory and the second overwrites the first. That is the documented
# layout (include/fortran/gfortran), and it is the right default because the
# overwhelmingly common case is one Fortran compiler per prefix. A site that
# really does install several versions of the SAME compiler side by side sets
# HIP_FORTRAN_COMPILER_DIR itself, for example to gfortran-13.3.0.
set(HIP_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(hip_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  # hip.F90 is uppercase: the array overloads sit behind preprocessor guards, so
  # the source must go through the C preprocessor.
  Fortran_PREPROCESS ON
  # Free form is the .F90 default, but stating it lets CMake emit whichever flag
  # the compiler in use expects rather than relying on the extension.
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

# A static archive is pulled in whole by the object that references it, so a
# consumer acquires every undefined C symbol the object carries, including entry
# points their installed runtime may predate. Per-function sections plus
# --gc-sections lets the linker drop the wrappers nobody called, resolving at
# object granularity rather than archive-member granularity.
#
# Guarded on the compilers known to accept the spelling; others keep the current
# behaviour, which is correct whenever the installed runtime matches.
if(CMAKE_Fortran_COMPILER_ID MATCHES "^(GNU|Flang|LLVMFlang|Intel|IntelLLVM)$")
  target_compile_options(hip_fortran PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(hip_fortran INTERFACE
    "$<$<LINK_LANG_AND_ID:Fortran,GNU,Flang,LLVMFlang,Intel,IntelLLVM>:LINKER:--gc-sections>")
endif()

# Translate the tri-state into the two macros the generated source is guarded
# with. assumed-rank implies the F2008 tier: in the source the assumed-rank
# specific is nested inside USE_FPOINTER_INTERFACES, and selects itself over the
# per-rank ones.
if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
  target_compile_definitions(hip_fortran PRIVATE USE_FPOINTER_INTERFACES=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(hip_fortran PRIVATE
    USE_FPOINTER_INTERFACES=1 USE_ASSUMED_RANK_INTERFACES=1)
endif()

# No line-length flag: the generator wraps the emitted source at 112 columns,
# inside the 132 that free-form Fortran guarantees, so every compiler reads it
# without a vendor-specific option.

target_include_directories(hip_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:include/fortran/${HIP_FORTRAN_COMPILER_DIR}>")

# Link interface only: needed to link an executable, not to build this archive.
# In-tree hip::host comes from clr. Standalone it has to be looked for, and
# testing for the target without ever calling find_package would mean it is never
# found: the archive would then export no link interface at all, and every
# consumer would hit undefined references to hipMalloc and friends despite
# linking hip::hip_fortran exactly as documented.
#
# hip-config.cmake reaches FindThreads, which hard-errors unless C or CXX is
# enabled, so C is enabled first. QUIET, because a site that has no installed
# ROCm can still legitimately build the bindings, since they compile against nothing.
if(NOT TARGET hip::host)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hip QUIET)
endif()

if(TARGET hip::host)
  target_link_libraries(hip_fortran INTERFACE hip::host)
else()
  message(STATUS
    "HIP: the runtime was not found, so the Fortran bindings carry no link "
    "interface; a consumer will have to link it (-lamdhip64) itself.")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIP_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
endif()

if(BUILD_FORTRAN_TESTS)
  # enable_testing() here is enough for a STANDALONE build of this directory, but
  # NOT for an in-tree one: CMake only writes the chain of CTestTestfile.cmake
  # that ctest walks down from a build root when testing was enabled at the TOP
  # level. Enabled only here, the tests are built and registered, and `ctest` run
  # from the top of the parent build reports "Total Tests: 0", a silent
  # nothing, which is the worst way to find out. So say it.
  enable_testing()
  if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    get_property(_top_testing GLOBAL PROPERTY CTEST_TARGETS_ADDED)
    if(NOT _top_testing)
      message(STATUS
        "HIP: the Fortran binding tests are registered, but ctest discovers "
        "tests only from a build root where testing was enabled at the top "
        "level. If `ctest` there reports no tests, add include(CTest) or "
        "enable_testing() to the top-level CMakeLists, or run ctest from "
        "${CMAKE_CURRENT_BINARY_DIR}/test.")
    endif()
  endif()
  add_subdirectory(test)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS hip_fortran
        EXPORT hip-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}")
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}"
        FILES_MATCHING PATTERN "*.mod")
install(EXPORT hip-fortran-targets
        FILE hip-fortran-targets.cmake
        NAMESPACE hip::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hip")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/hip-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hip")
# No PATH_VARS: it only creates PACKAGE_<var>, and the template interpolates
# @HIP_FORTRAN_COMPILER_DIR@ directly. Passing it was a dead argument, and worse
# than dead if anyone later switched the template to the PACKAGE_ form, since
# PATH_VARS would then resolve this relative value as a path under the prefix.
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hip")

# Version read from the VERSION file in the source tree, the same one hipamd
# parses, and NOT from hip_VERSION: that is set by find_package(hip) above and
# reports whatever HIP is INSTALLED, which is not what these bindings were
# generated from. Components are read as a list rather than captured by regex,
# so a legitimate 0 in any position cannot be mistaken for a failed match.
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
  # Installed in BOTH package directories, and it has to be: CMake looks for
  # <pkg>-config-version.cmake NEXT TO the <pkg>-config.cmake it actually
  # resolved, and find_package(hip-fortran 7.17) resolves the FORWARDER.
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config-version.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hip")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config-version.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hip-fortran")
else()
  message(STATUS
    "HIP: could not determine the version from the VERSION file, so no "
    "hip-fortran-config-version.cmake is installed and a versioned "
    "find_package(hip-fortran) will not match.")
endif()

# The generated source ships too, so a user on a compiler ROCm ships no .mod for
# can rebuild the bindings without cloning a monorepo for one file. Under share/
# and not include/, because everything under include/fortran/ is partitioned by
# compiler and the config enumerates that directory to report which compilers
# are installed; a source file there would be listed as one.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hip.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/hip/fortran")

# The config lands in lib/cmake/hip, BESIDE the C package, because that is what
# lets hip-config.cmake pick it up with a bare
# include(hip-fortran-config.cmake OPTIONAL): a Fortran binding installed after
# the C library is then found at find_package(hip) time, and ignored when absent.
#
# But find_package(hip-fortran), the other documented spelling, searches
# lib/cmake/hip-fortran*, which lib/cmake/hip does not match, so on its own that
# placement makes the direct spelling fail with "could not find a package
# configuration file". A forwarder gives the search somewhere to land without
# duplicating the real config, which would also need its own copy of the targets
# file and would then be a second thing to keep in step.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forward/hip-fortran-config.cmake"
"# Generated forwarder. The real config lives beside the HIP C package, in
# lib/cmake/hip, so that hip-config.cmake can include it as a sibling. This file
# only exists so that find_package(hip-fortran) resolves to the same place.
include(\"\${CMAKE_CURRENT_LIST_DIR}/../hip/hip-fortran-config.cmake\")
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forward/hip-fortran-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hip-fortran")
