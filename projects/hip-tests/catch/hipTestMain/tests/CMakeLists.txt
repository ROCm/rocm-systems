# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(LEVEL_TEST_SRC hip_test_level_test.cc ../hip_test_level.cc)
set_source_files_properties(${LEVEL_TEST_SRC} PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})

add_executable(HipTestLevelUnitTest EXCLUDE_FROM_ALL ${LEVEL_TEST_SRC})
target_link_libraries(HipTestLevelUnitTest PRIVATE Catch2::Catch2WithMain)
add_dependencies(build_tests HipTestLevelUnitTest)
add_test(NAME HipTestLevelUnitTest COMMAND HipTestLevelUnitTest)
set_tests_properties(HipTestLevelUnitTest PROPERTIES LABELS "hosttest")
