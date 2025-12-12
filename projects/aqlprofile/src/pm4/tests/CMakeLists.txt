cmake_minimum_required(VERSION 3.16.0)

include(GoogleTest)
find_package(GTest QUIET)
include_directories(${GTEST_INCLUDE_DIRS})

# Add a test for gfx9 command builder
add_executable(command-builder-test)
SET(AQLPROFILE_COMMAND_BUILDER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmd_builder_tests.cpp
)
target_sources(command-builder-test PRIVATE ${AQLPROFILE_COMMAND_BUILDER_SOURCES})
target_include_directories(command-builder-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR})
target_link_libraries(
    command-builder-test
    PRIVATE
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    command-builder-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

# Add a test for pmc_builder
add_executable(pmc-builder-test)
SET(AQLPROFILE_PMC_BUILDER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pmc_builder_tests.cpp
)
target_sources(pmc-builder-test PRIVATE ${AQLPROFILE_PMC_BUILDER_SOURCES})
target_include_directories(pmc-builder-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR} ${LIB_DIR}/core/include)
target_link_libraries(
    pmc-builder-test
    PRIVATE
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    pmc-builder-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

# Add a test for gfx9_command_builder
add_executable(gfx9-command-builder-test)
SET(AQLPROFILE_GFX9_COMMAND_BUILDER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gfx9_cmd_builder_tests.cpp
)
target_sources(gfx9-command-builder-test PRIVATE ${AQLPROFILE_GFX9_COMMAND_BUILDER_SOURCES})
target_include_directories(gfx9-command-builder-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR})
target_link_libraries(
    gfx9-command-builder-test
    PRIVATE
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    gfx9-command-builder-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

# Add a SPM builder test
add_executable(spm-builder-test)
SET(AQLPROFILE_SPM_BUILDER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/spm_builder_test.cpp
)
target_sources(spm-builder-test PRIVATE ${AQLPROFILE_SPM_BUILDER_SOURCES})
target_include_directories(spm-builder-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR} ${LIB_DIR}/core/include)
target_link_libraries(
    spm-builder-test
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
    spm-builder-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

# Add a test for trace config
add_executable(trace-config-test)
SET(AQLPROFILE_TRACE_CONFIG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/trace_config_test.cpp
)
target_sources(trace-config-test PRIVATE ${AQLPROFILE_TRACE_CONFIG_SOURCES})
target_include_directories(trace-config-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR} ${LIB_DIR}/core/include)
target_link_libraries(
    trace-config-test
    PRIVATE
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    trace-config-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")

# Add a test for sqtt builder
add_executable(sqtt-builder-test)
SET(AQLPROFILE_SQTT_BUILDER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sqtt_builder_tests.cpp
)
target_sources(sqtt-builder-test PRIVATE ${AQLPROFILE_SQTT_BUILDER_SOURCES})
target_include_directories(sqtt-builder-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR} ${LIB_DIR}/core/include)
target_link_libraries(
    sqtt-builder-test
    PRIVATE
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main
            ${CMAKE_DL_LIBS})
gtest_discover_tests(
    sqtt-builder-test
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests")
