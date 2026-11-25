# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

function(hip_add_host_test TARGET)
  add_dependencies(build_tests ${TARGET})
  catch_discover_tests(${TARGET}
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS hosttest ${ARGN}
  )
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS ${TARGET})
  file(GLOB _CTEST_INC_FILES "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}-*_include.cmake")
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CTEST_INCLUDE_FILES ${_CTEST_INC_FILES})
endfunction()

set(LEVEL_TEST_SRC hip_test_level_test.cc ../hip_test_level.cc)
set_source_files_properties(${LEVEL_TEST_SRC} PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
add_executable(HipTestLevelUnitTest EXCLUDE_FROM_ALL ${LEVEL_TEST_SRC})
target_link_libraries(HipTestLevelUnitTest PRIVATE Catch2::Catch2WithMain)
# hip_test_level.cc resolves test category names through the generated
# hip_test_parameters.hh.
target_include_directories(HipTestLevelUnitTest PRIVATE ${CATCH_INCLUDE_BINARY_DIR})
add_dependencies(HipTestLevelUnitTest hip_tests_config)
hip_add_host_test(HipTestLevelUnitTest)

set_source_files_properties(hip_test_level_abort_test.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
add_executable(HipTestLevelAbortTest EXCLUDE_FROM_ALL
               hip_test_level_abort_test.cc $<TARGET_OBJECTS:Main_Object>)
set_target_properties(HipTestLevelAbortTest PROPERTIES LINKER_LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(HipTestLevelAbortTest PRIVATE Catch2::Catch2 ${GPGPU_LINKER_LIBRARIES})
if(UNIX)
  target_link_libraries(HipTestLevelAbortTest PRIVATE stdc++fs dl pthread rt)
endif()
hip_add_host_test(HipTestLevelAbortTest
  ENVIRONMENT HIP_TEST_LEVEL=not_a_level
  PASS_REGULAR_EXPRESSION "is not a supported level"
  FAIL_REGULAR_EXPRESSION "LevelAbort_MustNotRun ran"
)
