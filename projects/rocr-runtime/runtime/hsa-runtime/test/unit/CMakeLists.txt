# Unit tests for pieces of the runtime that can be exercised without a device.
#
# Off by default and skipped entirely when GoogleTest is not present, so this
# neither adds a build dependency nor changes what an existing build produces.
# Enable with -DBUILD_UNIT_TESTS=ON, then run with ctest.

find_package(GTest QUIET)

if(NOT GTest_FOUND)
  message(STATUS "BUILD_UNIT_TESTS is ON but GoogleTest was not found; skipping unit tests")
  return()
endif()

add_executable(hsa_runtime_unit_tests amd_aie_arg_sync_test.cc)

target_include_directories(hsa_runtime_unit_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../inc)

target_link_libraries(hsa_runtime_unit_tests PRIVATE GTest::gtest GTest::gtest_main)

set_property(TARGET hsa_runtime_unit_tests PROPERTY CXX_STANDARD 17)

add_test(NAME hsa_runtime_unit_tests COMMAND hsa_runtime_unit_tests)
