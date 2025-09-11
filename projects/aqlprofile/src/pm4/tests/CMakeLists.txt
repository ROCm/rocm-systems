include(GoogleTest)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})


# Add a test for gfx9 command builder
add_executable(gfx9-command-builder-test)
SET(AQLPROFILE_COMMAND_BUILDER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmd_builder_tests.cpp
)
target_sources(gfx9-command-builder-test PRIVATE ${AQLPROFILE_COMMAND_BUILDER_SOURCES})
target_include_directories(gfx9-command-builder-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR})
target_link_libraries(
    gfx9-command-builder-test
    PRIVATE 
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main)
gtest_add_tests(
    TARGET gfx9-command-builder-test
    SOURCES ${AQLPROFILE_COMMAND_BUILDER_SOURCES}
    TEST_LIST gfx9-command-builder-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(
    ${gfx9-command-builder-test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")