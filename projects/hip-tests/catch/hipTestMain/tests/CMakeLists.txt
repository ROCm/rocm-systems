# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Host-only (no HIP, no GPU) tests for the level-resolution logic.
# hip_test_level.cc is compiled here as CXX: the LANGUAGE property set in the
# parent directory is directory-scoped and does not apply to this target.
add_executable(HipTestLevelUnitTest EXCLUDE_FROM_ALL
    hip_test_level_test.cc
    ../hip_test_level.cc)
set_target_properties(HipTestLevelUnitTest PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(HipTestLevelUnitTest PRIVATE Catch2::Catch2WithMain)
add_dependencies(build_tests HipTestLevelUnitTest)
add_test(NAME HipTestLevelUnitTest COMMAND HipTestLevelUnitTest)
set_tests_properties(HipTestLevelUnitTest PROPERTIES LABELS "hosttest")
