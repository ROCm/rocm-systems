# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Install hipFile unit-test assets so the suite can be run from an installed,
# relocatable tree (e.g. a packaged TheRock test artifact), not just the build
# tree.
#
# The problem:
#   gtest_discover_tests() writes ctest fragments that reference the test
#   executables (and the CMake GoogleTest helper) by ABSOLUTE build-tree paths,
#   so they are not relocatable and cannot be shipped in an artifact.
#
# The approach (mirrors rocm-systems/projects/hip-tests):
#   Install the unit-test binaries to a known share/ location and ship a small,
#   self-locating CTestTestfile pair:
#     share/hipfile/test/CTestTestfile.cmake          -> include(script/hipfile_discover.cmake)
#     share/hipfile/test/script/hipfile_discover.cmake -> discovery + add_test
#   ctest evaluates an include()'d file in full scripting mode, so the discover
#   script locates ITSELF via file(REAL_PATH ${CMAKE_CURRENT_LIST_FILE}),
#   computes the runtime library path RELATIVE to that location, enumerates each
#   binary's GoogleTest cases at ctest-run time, and emits one add_test per case.
#
# Why discovery is deferred to ctest-run time (not build/install time):
#   The test machine is the one place the binary is guaranteed to run. Enumerating
#   at build or install time couples this to whether that environment can launch
#   the binary (and resolve its shared libs) -- which is not guaranteed in TheRock
#   (libs come from sibling subprojects' staged artifacts). At ctest-run time the
#   tree is fully assembled and the relative library path resolves.
#
# Why one add_test PER CASE (not per binary):
#   Some hipFile tests share process-global state (the Context<DriverState>
#   singleton) and leak across cases; they only pass when each case runs in its
#   own process. This preserves the per-case isolation gtest_discover_tests gives.
#
# Run with:
#   ctest --test-dir <install>/share/hipfile/test -L unit
#
# Scope: UNIT tests only. The "system" tests require a HIP-capable GPU and the
# "stress" tests are gdb-wrapped concurrency testers; both are intentionally
# excluded and continue to run from the build tree on dedicated runners.
#
# NOTE: This module deliberately does NOT use CPack / rocm_package_setup_component.
# In TheRock, hipFile is consumed through the stage/dist/artifact-toml flow. The
# plain install(COMPONENT tests) below is honored by both that flow and the
# standalone build.

include_guard(GLOBAL)

# Relative install destination for the unit-test binaries + ctest files.
set(HIPFILE_TEST_INSTALL_DIR "share/hipfile/test")

# Collect the unit-test targets that exist in this configuration, with the label
# suffix to attach to each binary's cases.
#   - hipfile_tests : defined only when UNIT_TEST_SOURCE_FILES is non-empty,
#                     which today is the NVIDIA build (cufile-api-compat); on
#                     AMD the unit tests live entirely in internal_tests (see
#                     test/CMakeLists.txt).
#   - internal_tests: only defined when building the AMD detail (see
#                     test/amd_detail/CMakeLists.txt).
set(_unit_targets)
set(_unit_suffixes)
if(TARGET hipfile_tests)
    list(APPEND _unit_targets hipfile_tests)
    list(APPEND _unit_suffixes hipfile)
endif()
if(TARGET internal_tests)
    list(APPEND _unit_targets internal_tests)
    list(APPEND _unit_suffixes internal)
endif()

if(NOT _unit_targets)
    message(STATUS "hipFile: no unit-test targets to install")
    return()
endif()

# Install the unit-test executables. RPATH is intentionally left to the normal
# install machinery (and, under TheRock, its RPATH fixup): the discover script
# also sets LD_LIBRARY_PATH relative to the installed location, which is the
# authoritative mechanism and keeps the tree relocatable regardless of RPATH.
install(
    TARGETS ${_unit_targets}
    RUNTIME DESTINATION "${HIPFILE_TEST_INSTALL_DIR}"
    COMPONENT tests
)

# Build the per-binary discovery blocks for the discover script. Each binary's
# cases are listed at ctest-run time and turned into isolated add_test entries.
# We bake in only the binary NAME and label SUFFIX; all paths are resolved at
# run time relative to the script's own location.
set(_discover_blocks "")
list(LENGTH _unit_targets _n)
math(EXPR _last "${_n} - 1")
foreach(_i RANGE ${_last})
    list(GET _unit_targets ${_i} _name)
    list(GET _unit_suffixes ${_i} _suffix)
    string(APPEND _discover_blocks
        "_hipfile_discover_binary(\"${_name}\" \"${_suffix}\")\n")
endforeach()

# Generate the discover script (configure time; contains NO absolute paths).
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/hipfile_discover.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hipfile_discover.cmake"
    @ONLY
)

# Install the discover script under script/ and a tiny top-level CTestTestfile
# that includes it via a relative path (ctest resolves include() relative to the
# test directory when invoked with --test-dir).
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/hipfile_discover.cmake"
    DESTINATION "${HIPFILE_TEST_INSTALL_DIR}/script"
    COMPONENT tests
)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/hipfile_test_CTestTestfile.cmake"
    "# hipFile unit tests -- relocatable. Generated by AISInstallTests.cmake.\n"
    "include(script/hipfile_discover.cmake)\n"
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/hipfile_test_CTestTestfile.cmake"
    DESTINATION "${HIPFILE_TEST_INSTALL_DIR}"
    COMPONENT tests
    RENAME "CTestTestfile.cmake"
)

message(STATUS "hipFile: will install relocatable unit tests to ${HIPFILE_TEST_INSTALL_DIR}: ${_unit_targets}")
