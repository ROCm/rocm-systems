#
#
#
rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

# -------------------------------------------------------------------------------------- #
#
# Link to static rocprofiler library
#
# -------------------------------------------------------------------------------------- #

set(rocprofiler_lib_sources agent.cpp buffer.cpp timestamp.cpp version.cpp)

add_executable(rocprofiler-lib-tests)
target_sources(rocprofiler-lib-tests PRIVATE ${rocprofiler_lib_sources} details/agent.cpp)
target_link_libraries(
    rocprofiler-lib-tests
    PRIVATE rocprofiler::rocprofiler-static-library
            rocprofiler::rocprofiler-common-library GTest::gtest GTest::gtest_main)

gtest_add_tests(
    TARGET rocprofiler-lib-tests
    SOURCES ${rocprofiler_lib_sources}
    TEST_LIST lib_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(${lib_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests")

# -------------------------------------------------------------------------------------- #
#
# Link to shared rocprofiler library
#
# -------------------------------------------------------------------------------------- #

set(rocprofiler_shared_lib_sources external_correlation.cpp registration.cpp status.cpp)

add_executable(rocprofiler-lib-tests-shared)
target_sources(rocprofiler-lib-tests-shared PRIVATE ${rocprofiler_shared_lib_sources})
target_link_libraries(
    rocprofiler-lib-tests-shared
    PRIVATE rocprofiler::rocprofiler-shared-library
            rocprofiler::rocprofiler-common-library GTest::gtest GTest::gtest_main)

gtest_add_tests(
    TARGET rocprofiler-lib-tests-shared
    SOURCES ${rocprofiler_shared_lib_sources}
    TEST_LIST shared_lib_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${shared_lib_TESTS}
    PROPERTIES TIMEOUT 360 LABELS "unittests" ENVIRONMENT
               "HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler-shared-library>")
