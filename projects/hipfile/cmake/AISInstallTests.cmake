# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Install hipFile unit-test assets so they can be run from an installed tree
# (e.g. a packaged TheRock test artifact) rather than only from the build tree.
#
# Why this exists:
#   gtest_discover_tests() generates ctest fragments that reference the test
#   executables by ABSOLUTE build-tree paths, so those fragments are not
#   relocatable and cannot be shipped in an artifact. Here we instead install
#   the unit-test binaries to a known location and generate a relocatable
#   CTestTestfile.cmake that invokes them by RELATIVE path. ctest sets the
#   working directory to the directory containing CTestTestfile.cmake, so the
#   binaries (installed alongside it) are reached via "./<binary>".
#
#   The CTestTestfile.cmake is generated at INSTALL time (not configure time)
#   by AISGenerateInstallCTest.cmake, which enumerates each binary's GoogleTest
#   cases and emits one ctest entry per case. Per-case isolation matters: some
#   internal tests leak state across cases and only pass when each runs in its
#   own process -- exactly the behavior gtest_discover_tests provides.
#
# Scope: UNIT tests only. The "system" tests require a HIP-capable GPU and the
# "stress" tests are gdb-wrapped concurrency testers; both are intentionally
# excluded from the packaged unit suite. They continue to run from the build
# tree and on dedicated runners.

include_guard(GLOBAL)

# rocm_package_setup_component() lives here. Safe to include early -- it has its
# own include guard and is also pulled in by AISInstall.
include(ROCMCreatePackage)

# Relative install destination for the unit-test binaries + CTestTestfile.cmake.
# Sits under share/ alongside the rest of hipFile's installed data.
set(HIPFILE_TEST_INSTALL_DIR "share/hipfile/test")

# Collect the unit-test targets that actually exist in this configuration, along
# with the label suffix to attach to each binary's cases.
#   - hipfile_tests : defined whenever BUILD_TESTING is on.
#   - internal_tests: only defined when building the AMD detail (see
#                     test/amd_detail/CMakeLists.txt).
set(_hipfile_unit_test_targets)
set(_hipfile_unit_test_suffixes)
if(TARGET hipfile_tests)
    list(APPEND _hipfile_unit_test_targets hipfile_tests)
    list(APPEND _hipfile_unit_test_suffixes hipfile)
endif()
if(TARGET internal_tests)
    list(APPEND _hipfile_unit_test_targets internal_tests)
    list(APPEND _hipfile_unit_test_suffixes internal)
endif()

if(NOT _hipfile_unit_test_targets)
    message(STATUS "hipFile: no unit-test targets to install")
    return()
endif()

# Ship the test assets in their own CPack component so they land in a separate
# "hipfile-tests" package instead of the runtime/devel packages -- matching the
# convention used by sibling rocm-systems projects (e.g. rccl). This requires
# AISInstallTests to be included BEFORE rocm_create_package() (in AISInstall)
# so the component is registered when the package is created.
rocm_package_setup_component(tests)

# Point the installed binaries at libhipfile.so (and the ROCm libs) in the
# packaged tree's lib/ directory. GTest/GMock are linked statically, so the
# library is the only project-private runtime dependency to resolve.
#
# The binaries install to share/hipfile/test, so lib/ is three levels up:
#   test -> hipfile -> share -> <prefix> -> lib
# "$ORIGIN" is also included so the suite still runs if the library is dropped
# alongside the binaries. --enable-new-dtags emits RUNPATH (not RPATH) so an
# explicit LD_LIBRARY_PATH can still override it, matching the rest of ROCm.
set_target_properties(${_hipfile_unit_test_targets} PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../../../lib"
)
foreach(_tgt IN LISTS _hipfile_unit_test_targets)
    target_link_options(${_tgt} PRIVATE "-Wl,--enable-new-dtags")
endforeach()

# Install the unit-test executables.
install(
    TARGETS ${_hipfile_unit_test_targets}
    RUNTIME DESTINATION "${HIPFILE_TEST_INSTALL_DIR}"
    COMPONENT tests
)

# Generate CTestTestfile.cmake at install time, then install it alongside the
# binaries. The generator script enumerates each binary's cases (the binaries
# must exist, hence install time) and writes relative-path ctest entries.
#
# The generator enumerates the BUILD-tree binaries, not the staged/installed
# copies. Under CPack component packaging the 'tests' component is staged in
# isolation -- without the 'runtime' component's libhipfile.so and (when GTest
# is a shared library) without its libgtest/libgmock -- so a staged binary
# cannot be launched at generate time to list its cases. The build-tree binary
# keeps its build RPATH and resolves every shared dependency, so we query it
# instead. The cases are identical; only the launch path differs from the
# installed binary that the generated relative-path entries ultimately invoke.
#
# $<TARGET_FILE:...> is expanded here at generate time -- install(CODE) supports
# generator expressions (CMake >= 3.14) -- yielding concrete build-tree paths in
# the install script.
set(_hipfile_unit_test_files)
foreach(_tgt IN LISTS _hipfile_unit_test_targets)
    list(APPEND _hipfile_unit_test_files "$<TARGET_FILE:${_tgt}>")
endforeach()

set(_hipfile_install_ctest "${CMAKE_CURRENT_BINARY_DIR}/hipfile_install_CTestTestfile.cmake")
install(CODE "
    set(HIPFILE_TEST_INSTALL_DIR \"${HIPFILE_TEST_INSTALL_DIR}\")
    set(HIPFILE_TEST_NAMES \"${_hipfile_unit_test_targets}\")
    set(HIPFILE_TEST_FILES \"${_hipfile_unit_test_files}\")
    set(HIPFILE_TEST_SUFFIXES \"${_hipfile_unit_test_suffixes}\")
    set(HIPFILE_INSTALL_CTEST_FILE \"${_hipfile_install_ctest}\")
    include(\"${CMAKE_CURRENT_LIST_DIR}/AISGenerateInstallCTest.cmake\")
    # file(INSTALL ... RENAME ...) is DESTDIR-aware, so the staged file lands at
    # the correct path under \$ENV{DESTDIR}. (A separate file(RENAME) would not
    # honor DESTDIR and could miss the staged file in packaging/CI installs.)
    file(INSTALL \"${_hipfile_install_ctest}\"
        DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${HIPFILE_TEST_INSTALL_DIR}\"
        RENAME \"CTestTestfile.cmake\")
" COMPONENT tests)

message(STATUS "hipFile: will install unit tests to ${HIPFILE_TEST_INSTALL_DIR}: ${_hipfile_unit_test_targets}")
