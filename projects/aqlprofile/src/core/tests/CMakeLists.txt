# write a CMakeLists.txt file for the gfx9_factory_test 
# that includes the necessary libraries and sets the properties for the test
include(GoogleTest)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})


find_library(
    hsa-amd-aqlprofile64
    REQUIRED
    NAMES hsa-amd-aqlprofile64 hsa-amd-aqlprofile
    HINTS /opt/rocm/lib /opt/rocm
    PATHS /opt/rocm/lib /opt/rocm)

# Add test for memory manager 
add_executable(gfx9-memory-manager-test)
SET(AQLPROFILE_MEMORYMANAGER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/memorymanager_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../memorymanager.cpp
)

target_sources(gfx9-memory-manager-test PRIVATE ${AQLPROFILE_MEMORYMANAGER_SOURCES})
target_include_directories(gfx9-memory-manager-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR})
target_link_libraries(
    gfx9-memory-manager-test
    PRIVATE 
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main)


gtest_add_tests(
    TARGET gfx9-memory-manager-test
    SOURCES ${AQLPROFILE_MEMORYMANAGER_SOURCES}
    TEST_LIST gfx9-memory-manager_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${gfx9-memory-manager_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")

# Add test for aql profile 
add_executable(aqlprofile-test)
SET(AQLPROFILE_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/aql_profile_tests.cpp
)

target_sources(aqlprofile-test PRIVATE ${AQLPROFILE_TEST_SOURCES})
target_include_directories(aqlprofile-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR})
target_link_libraries(
    aqlprofile-test
    PRIVATE 
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main)


gtest_add_tests(
    TARGET aqlprofile-test
    SOURCES ${AQLPROFILE_TEST_SOURCES}
    TEST_LIST aqlprofile-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${aqlprofile-test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")


# Add tests for command buffer
add_executable(command-buffer-test)
SET(AQLPROFILE_COMMAND_BUFFER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer_tests.cpp
)
target_sources(command-buffer-test PRIVATE ${AQLPROFILE_COMMAND_BUFFER_SOURCES})
target_include_directories(command-buffer-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR})
target_link_libraries(
    command-buffer-test
    PRIVATE 
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main)

gtest_add_tests(
    TARGET command-buffer-test
    SOURCES ${AQLPROFILE_COMMAND_BUFFER_SOURCES}
    TEST_LIST command-buffer-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(
    ${command-buffer-test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")

# Add tests for counters
add_executable(counters-test)
SET(AQLPROFILE_COUNTERS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/counter_tests.cpp
)
target_sources(counters-test PRIVATE ${AQLPROFILE_COUNTERS_SOURCES})
target_include_directories(counters-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR})
target_link_libraries(
    counters-test
    PRIVATE 
            hsa-runtime64::hsa-runtime64
            ${hsa-amd-aqlprofile64}
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main)
gtest_add_tests(
    TARGET counters-test
    SOURCES ${AQLPROFILE_COUNTERS_SOURCES}
    TEST_LIST counters-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(
    ${counters-test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")