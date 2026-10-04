# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

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
#
# "${UNIX}" rather than ON, because Windows is not such a site: ROCm ships no
# Fortran compiler there at all. TheRock puts flang in LLVM_ENABLE_PROJECTS
# only in the non-WIN32 branch of compiler/pre_hook_amd-llvm.cmake, so the
# amdflang preference below finds nothing, the probe falls through to whatever
# gfortran happens to be on PATH, and enable_language(Fortran) then fails to
# identify it. clr add_subdirectory()s this directory under CLR_BUILD_HIP, so
# the guard matters from the first build that picks it up. A default and not a
# veto: -DBUILD_FORTRAN_BINDINGS=ON still works
# wherever a usable compiler exists.
option(BUILD_FORTRAN_BINDINGS "Build the HIP Fortran bindings" "${UNIX}")

# The bindings' own test suite, and the Fortran clients where a library has
# them: one switch rather than two neighbouring ones. ON by default, and
# guarded the same way as the bindings, so asking for them when the bindings
# were not built is a skip rather than an error.
option(BUILD_FORTRAN_CLIENTS "Build the HIP Fortran clients and binding tests" ON)

# BUILD_FORTRAN_BINDINGS and BUILD_FORTRAN_CLIENTS are ONE cache entry shared by
# every library in the monorepo, which is what you want for the common case:
# -DBUILD_FORTRAN_BINDINGS=OFF from the top turns the whole Fortran track off in
# a single flag. It is not what you want when one library has to be singled out,
# so each library also honours a name of its own, which wins when it is set.
if(DEFINED HIP_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${HIP_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED HIP_BUILD_FORTRAN_CLIENTS)
  set(BUILD_FORTRAN_CLIENTS ${HIP_BUILD_FORTRAN_CLIENTS})
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
#
# For HIP, none and assumed-shape select the SAME module. The typed memory
# families (hipMalloc, hipMemcpy and their variants) are the only routines here
# that take array arguments, and hipfort has always shipped their per-rank
# overloads unguarded, so the generator emits them as the plain #else of
# USE_ASSUMED_RANK with no macro to switch them off. The per-rank set is
# therefore what a build defining nothing at all gets, which is why the module
# tests USE_ASSUMED_RANK and nothing else. Verified by preprocessing hip.F90
# four ways with amdflang -cpp -E: no macro and USE_ASSUMED_SHAPE=1 hash
# identically, USE_ASSUMED_RANK=1 and both-defined hash identically to each
# other, and the two groups differ.
#
# none is still accepted rather than rejected, because this cache entry is
# shared with every other Fortran binding in ROCm, where the three tiers really
# are distinct; a monorepo-wide -DFORTRAN_ARRAY_INTERFACES=none must not turn
# into a fatal error here.
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

  # check_language() answers by writing the CACHE entry CMAKE_Fortran_COMPILER,
  # and rocm-cmake's ROCMChecks arms a variable_watch on exactly that name to
  # catch a project setting a toolchain variable behind the toolchain file's
  # back. A compiler probe is the one legitimate writer that watch cannot tell
  # apart from a violation, so it reports this one: a stray twelve-line banner
  # by default, and a hard configure failure under -DROCM_ERROR_TOOLCHAIN_VAR=ON
  # (or ROCMCHECKS_ERROR_TOOLCHAIN_VAR in the environment), which is a switch CI
  # turns on. develop has no probe here and so never tripped it, which makes
  # this directory the regression.
  #
  # So silence the checker across the probe and put it back afterwards, the same
  # save/set/restore rocRAND already uses in cmake/Dependencies.cmake. Plain
  # variables rather than cache ones: they shadow the cache for this scope only,
  # which is what the watch callback reads, and the user's cache entries are
  # left exactly as they were. Both names have to be cleared, because the
  # callback escalates to SEND_ERROR on ROCM_ERROR_TOOLCHAIN_VAR before it ever
  # looks at the warn flag.
  set(_rocm_warn_toolchain_var "${ROCM_WARN_TOOLCHAIN_VAR}")
  set(_rocm_error_toolchain_var "${ROCM_ERROR_TOOLCHAIN_VAR}")
  set(ROCM_WARN_TOOLCHAIN_VAR OFF)
  set(ROCM_ERROR_TOOLCHAIN_VAR OFF)

  # Prefer ROCm's own amdflang over whatever CMake would otherwise find first.
  # check_language(Fortran) takes the first compiler on the search path, which on
  # a ROCm CI image is the system gfortran, and that is how the bindings ended up
  # as the one component of ROCm built by a different toolchain from everything
  # else: installed under fortran/gfortran/ while the documentation promises
  # fortran/amdflang/, and exercising a compiler the product does not ship with.
  #
  # amdflang sits beside the clang already compiling this project, so the answer
  # is next to CMAKE_CXX_COMPILER when there is one; ROCM_PATH covers a
  # standalone configure. Searched with NO_DEFAULT_PATH so a stray amdflang
  # elsewhere on PATH cannot win over the ROCm being built against.
  #
  # An explicit CMAKE_Fortran_COMPILER, or FC in the environment, always wins:
  # building these bindings with gfortran is a supported thing to do, and a .mod
  # is compiler-specific, so whoever asked for a compiler meant it.
  #
  # check_language() is a no-op once CMAKE_Fortran_COMPILER is defined, so this
  # has to run before it, and the probe below never overrides what we set here.
  if(NOT CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
    set(_rocm_fc_hints)
    foreach(_rocm_fc_cc "${CMAKE_CXX_COMPILER}" "${CMAKE_C_COMPILER}"
                        "${CMAKE_HIP_COMPILER}")
      if(_rocm_fc_cc)
        get_filename_component(_rocm_fc_dir "${_rocm_fc_cc}" DIRECTORY)
        list(APPEND _rocm_fc_hints "${_rocm_fc_dir}")
      endif()
    endforeach()
    if(ROCM_PATH)
      list(APPEND _rocm_fc_hints "${ROCM_PATH}/lib/llvm/bin" "${ROCM_PATH}/bin")
    endif()
    if(_rocm_fc_hints)
      find_program(_rocm_amdflang NAMES amdflang HINTS ${_rocm_fc_hints}
                   NO_DEFAULT_PATH)
      # Confirm it runs before committing to it. A ROCm tree can carry a dangling
      # amdflang symlink, and check_language() would not catch that: it does
      # nothing at all once the variable is set, so the failure would surface far
      # later as a broken enable_language with no hint of where the choice was
      # made.
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
    endif()
    unset(_rocm_fc_hints)
    unset(_rocm_fc_cc)
    unset(_rocm_fc_dir)
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

# Pulled in here rather than beside the install() rules below because the install
# layout is also baked into the target's INSTALL_INTERFACE include directory,
# which is set well before them. Nothing must hardcode lib/ or include/: a distro
# that installs to lib64 gets CMAKE_INSTALL_LIBDIR=lib64, and a literal
# lib/fortran then names a directory that does not exist.
include(GNUInstallDirs)

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

# Translate the tri-state into the one macro the generated source is guarded
# with. hip.F90 contains five bare #ifdef USE_ASSUMED_RANK regions and no
# USE_ASSUMED_SHAPE at all, for the reason given beside the option above, so
# defining USE_ASSUMED_SHAPE here would name a macro the source never tests: a
# reader would take it for the switch that selects the F2008 tier, and changing
# it would change nothing. The lower two tiers therefore define nothing, and
# assumed-rank defines the one macro that does select a different module.
if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(hip_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

# No line-length flag: every emitted line fits inside the 132 columns free-form
# Fortran guarantees, so no compiler needs a vendor-specific option to read this
# source. Measured on the committed hip.F90, the longest line is 131 columns.

target_include_directories(hip_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}>")

# Link interface only: needed to link an executable, not to build this archive.
# Three cases, and the order matters.
#
# In-tree under clr the runtime is BEING BUILT here, as the target amdhip64, and
# clr adds this directory right after hipamd so it already exists. Calling
# find_package(hip) there would be circular, and under TheRock it is fatal
# rather than merely pointless: the super-project installs a dependency provider
# that intercepts every find_package and hard-errors on a package it builds
# itself but which the caller has not declared ("Add a BUILD_DEPS or
# RUNTIME_DEPS appropriately"). QUIET does not help, because the provider
# reports before the search is ever attempted.
#
# Standalone the runtime has to be looked for. Testing for the target without
# ever calling find_package would mean it is never found: the archive would
# export no link interface at all, and every consumer would hit undefined
# references to hipMalloc and friends despite linking hip::hip_fortran exactly
# as documented.
#
# hip-config.cmake reaches FindThreads, which hard-errors unless C or CXX is
# enabled, so C is enabled first. QUIET, because a site that has no installed
# ROCm can still legitimately build the bindings, since they compile against nothing.
if(NOT TARGET hip::host AND NOT TARGET amdhip64)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hip QUIET)
endif()

# Whether the INSTALLED link interface ends up naming hip::host is also what the
# config package has to know, and it cannot work that out for itself: it sees
# only the targets file. So record it here and interpolate it into the template.
# The consumer-side find_dependency(hip) exists solely to make hip::host exist
# before the targets file names it, so a binding built with no runtime in sight,
# which names nothing, must not make its consumers find a package the artifact
# does not depend on.
if(TARGET hip::host)
  target_link_libraries(hip_fortran INTERFACE hip::host)
  set(HIP_FORTRAN_NEEDS_C_PACKAGE TRUE)
elseif(TARGET amdhip64)
  # In-tree the runtime is a real target, and clr already exports it from TWO
  # sets (hip-targets and hip-lang-targets). CMake refuses to export a target
  # whose link interface names one that is exported more than once, so naming
  # amdhip64 unconditionally fails the generate step with "not in this export
  # set, but in multiple other export sets".
  #
  # So it is linked for the BUILD only, and the INSTALLED interface names
  # hip::host -- exactly what the standalone branch above exports, which keeps
  # the two packages identical from a consumer's side.
  # hip-fortran-config.cmake calls find_dependency(hip) before reading the
  # targets file, so hip::host exists by the time a consumer resolves it.
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

# The binding tests below are GPU runtime executables, so in-tree they belong to
# the client/test track and must not come along with a plain library build: 29
# test binaries linking a GPU runtime is not what -DCLR_BUILD_HIP=ON asked for,
# and test/CMakeLists.txt additionally shells out to rocm_agent_enumerator AT
# CONFIGURE TIME to pick an offload arch, which a library build has no business
# doing. Every other ROCm Fortran binding gates its tests on its own project's
# client switch; clr has none -- its options are CLR_BUILD_HIP, CLR_BUILD_OCL,
# CLR_HRR_CODEGEN_CHECK, USE_NEW_HOSTCALL_IMPL and BUILD_SHARED_LIBS, and the
# BUILD_TESTS it does carry is declared inside opencl/ and scoped there.
#
# So the condition is BUILD_TESTING, the name CMake's own include(CTest) defines
# and the fallback every sibling's gate already ends in. clr never includes
# CTest, so in-tree it is simply undefined and the tests stay off until someone
# asks for them with -DBUILD_TESTING=ON, while a super-project that has turned
# testing on for the whole build gets them without a HIP-specific flag.
#
# HIP_BUILD_FORTRAN_CLIENTS still wins, because the option block above promises
# that a per-library name overrides the shared one; ignoring it here would make
# that promise false for the one library where it is the only escape hatch.
#
# Standalone, this directory IS the build and the tests are the point, so none
# of this applies. Setting BUILD_FORTRAN_CLIENTS as a plain variable is
# deliberate: the cache entry is one switch shared by every library in the
# monorepo, so turning it off here has to be scoped to this directory and must
# not be written back.
#
# Placed here, immediately before the switch is read, rather than beside the
# option() above: everything between the two can return early -- a
# -DBUILD_FORTRAN_BINDINGS=OFF configure, or a host with no Fortran compiler --
# and a message about skipping tests would then be printed for a directory that
# configures nothing at all, blaming a reason that is not the one that applied.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND BUILD_FORTRAN_CLIENTS
   AND NOT (BUILD_TESTING OR HIP_BUILD_FORTRAN_CLIENTS))
  message(STATUS
    "HIP: testing is not enabled, skipping the Fortran binding tests. Pass "
    "-DBUILD_TESTING=ON or -DHIP_BUILD_FORTRAN_CLIENTS=ON to build them.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

if(BUILD_FORTRAN_CLIENTS)
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

include(CMakePackageConfigHelpers)

# The config package goes in lib/cmake/hip-fortran, the directory
# find_package(hip-fortran) searches natively, so that the documented spelling
# resolves with nothing standing in for it.
#
# The alternative, putting it beside the C package in lib/cmake/hip, only pays
# off if hip-config.cmake grows an include() of it, and that include does not
# exist: hip-config.cmake.in mentions Fortran nowhere. Adding one would mean
# editing the file EVERY HIP consumer loads, and it would have to run after the
# C targets are defined or the Fortran targets file would name hip::host before
# it exists -- rocm_export_targets' INCLUDE hook emits its entries before the
# targets file, so it is not that hook, and growing one belongs in rocm-cmake
# rather than here. rocRAND is the cautionary precedent: its installed
# rocrand-config.cmake carries exactly such an include with no OPTIONAL, so
# find_package(rocrand) fails outright on a prefix where the bindings are not
# installed.
set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/hip-fortran")

# COMPONENT devel on every rule below, because everything this directory
# installs -- a static archive, a set of .mod files, the generated source and
# the CMake package -- is a development artifact: nothing here is needed to RUN
# a program, only to build one against the bindings. devel is also what
# rocm-cmake would have chosen, since rocm_install_targets and rocm_install
# default to it. Without it these files land in the runtime component instead,
# so a user who installs only the runtime package gets an archive and a set of
# .mod files they cannot use, while the one who installs devel to build against
# HIP does not get them at all.
#
# Plain install(), not rocm_install(): this directory must also configure
# standalone, where rocm-cmake was never loaded and those commands do not exist.
install(TARGETS hip_fortran
        EXPORT hip-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}"
                COMPONENT devel)
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.mod")
install(EXPORT hip-fortran-targets
        FILE hip-fortran-targets.cmake
        NAMESPACE hip::
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

# The two directories the config package has to name, kept out of the template
# so that neither side spells lib/ or include/ literally. A literal lib/fortran
# is wrong the moment CMAKE_INSTALL_LIBDIR is lib64, and the config then hands
# the consumer an IMPORTED_LOCATION that does not exist.
set(HIP_FORTRAN_LIBDIR     "${CMAKE_INSTALL_LIBDIR}/fortran")
set(HIP_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/hip-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
  INSTALL_DESTINATION "${_fortran_pkgdir}"
  # PATH_VARS is exactly what these two want: @PACKAGE_INIT@ rewrites each into
  # an absolute path computed from where the config file was FOUND, so the
  # package survives being relocated and gets lib64 right without the template
  # knowing which layout it was built for. HIP_FORTRAN_COMPILER_DIR stays a
  # plain @-substitution on purpose -- it is a bare directory NAME, not a path,
  # and PATH_VARS would resolve it as one, under the install prefix.
  PATH_VARS HIP_FORTRAN_LIBDIR HIP_FORTRAN_INCLUDEDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

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
  # Beside the config it belongs to, and only there: CMake looks for
  # <pkg>-config-version.cmake NEXT TO the <pkg>-config.cmake it resolved, and
  # with the package in one directory there is only one place to look.
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT devel)
else()
  # Better no version file than a wrong one: without it an unversioned
  # find_package still succeeds, whereas a version file claiming the wrong
  # number makes a versioned one succeed and link the wrong .mod.
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
        DESTINATION "${CMAKE_INSTALL_DATADIR}/hip/fortran"
        COMPONENT devel)
