cmake_minimum_required(VERSION 3.16.0)
project(aqlprofile-sdk-tests)

# write a CMakeLists.txt file for the gfx9_factory_test
# that includes the necessary libraries and sets the properties for the test
include(GoogleTest)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

if(NOT DEFINED AQLPROFILE_TARGET)
    find_library(
        AQLPROFILE_TARGET
        REQUIRED
        NAMES hsa-amd-aqlprofile64 hsa-amd-aqlprofile
        HINTS /opt/rocm ${CMAKE_INSTALL_PREFIX}
        PATHS /opt/rocm
        PATH_SUFFIXES lib)
endif()

# Add test for memory manager
add_executable(gfx9-memory-manager-test)
SET(AQLPROFILE_MEMORYMANAGER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/memorymanager_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../memorymanager.cpp
)

target_sources(gfx9-memory-manager-test PRIVATE ${AQLPROFILE_MEMORYMANAGER_SOURCES})
target_include_directories(gfx9-memory-manager-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIB_DIR}
        ${LIB_DIR}/core/include)
target_link_libraries(
    gfx9-memory-manager-test
    PUBLIC
            aqlprofile::headers
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
target_include_directories(aqlprofile-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIB_DIR}
        ${LIB_DIR}/core/include)
target_link_libraries(
    aqlprofile-test
    PRIVATE
            aqlprofile::aqlprofile-lib
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})


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
target_include_directories(command-buffer-test PRIVATE
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIB_DIR}
        ${LIB_DIR}/core/include)
target_link_libraries(
    command-buffer-test
    PUBLIC
            aqlprofile::headers
    PRIVATE
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})

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
target_include_directories(counters-test PRIVATE
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIB_DIR}
        ${LIB_DIR}/core/include
        )
target_link_libraries(
    counters-test
    PRIVATE
            hsa-runtime64::hsa-runtime64
            ${AQLPROFILE_TARGET}
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_add_tests(
    TARGET counters-test
    SOURCES ${AQLPROFILE_COUNTERS_SOURCES}
    TEST_LIST counters-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(
    ${counters-test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")

# Add tests for pm4 factory
add_executable(pm4-factory-test)
SET(AQLPROFILE_PM4_FACTORY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pm4_factory_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../pm4_factory.cpp
)
target_sources(pm4-factory-test PRIVATE ${AQLPROFILE_PM4_FACTORY_SOURCES})
target_include_directories(pm4-factory-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR} ${LIB_DIR}/core/include)
target_link_libraries(
    pm4-factory-test
    PRIVATE
            aqlprofile::aqlprofile-lib
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})

gtest_add_tests(
    TARGET pm4-factory-test
    SOURCES ${AQLPROFILE_PM4_FACTORY_SOURCES}
    TEST_LIST pm4-factory-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(
    ${pm4-factory-test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")

# Add tests for logger
add_executable(logger-test)
SET(AQLPROFILE_LOGGER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_tests.cpp
)
target_sources(logger-test PRIVATE ${AQLPROFILE_LOGGER_SOURCES})
target_include_directories(logger-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR} ${LIB_DIR}/core/include)
target_link_libraries(
    logger-test
    PRIVATE
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})

gtest_add_tests(
    TARGET logger-test
    SOURCES ${AQLPROFILE_LOGGER_SOURCES}
    TEST_LIST logger-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${logger-test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")

# Add tests for aql profile v2 header
add_executable(aql-profile-v2-test)
SET(AQLPROFILE_V2_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/aql_profile_v2_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../counters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../memorymanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../populate_aql.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../pm4_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../ip_offset_table_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../parse_ip_discovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../navi_reg_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../vega20_reg_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../spm_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gfx12_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gfx11_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gfx10_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gfx940_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gfx908_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gfx90a_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gfx9_factory.cpp

)
target_sources(aql-profile-v2-test PRIVATE ${AQLPROFILE_V2_SOURCES})
target_include_directories(aql-profile-v2-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR} ${LIB_DIR}/core/include)
target_link_libraries(
    aql-profile-v2-test
    PRIVATE
            aqlprofile::aqlprofile-lib
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})

gtest_add_tests(
    TARGET aql-profile-v2-test
    SOURCES ${AQLPROFILE_V2_SOURCES}
    TEST_LIST aql-profile-v2-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${aql-profile-v2-test_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")
