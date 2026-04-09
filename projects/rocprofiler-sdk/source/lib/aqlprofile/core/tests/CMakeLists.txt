cmake_minimum_required(VERSION 3.16.0)
project(aqlprofile-sdk-tests)

include(GoogleTest)
find_package(GTest QUIET)
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
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main)
gtest_discover_tests(
    gfx9-memory-manager-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

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
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    aqlprofile-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

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
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    command-buffer-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

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
        ${LIB_DIR}/core/include)
target_link_libraries(
    counters-test
    PRIVATE
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            ${AQLPROFILE_TARGET}
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    counters-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

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
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    pm4-factory-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

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
gtest_discover_tests(
    logger-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../gfx115x_factory.cpp
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
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    aql-profile-v2-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")
