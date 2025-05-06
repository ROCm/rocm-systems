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

set(rocprofiler_lib_sources
    agent.cpp
    buffer.cpp
    contexts.cpp
    enum_string.cpp
    hsa.cpp
    naming.cpp
    timestamp.cpp
    version.cpp
    hsa_barrier.cpp
    page_migration.cpp)

add_executable(rocprofiler-sdk-lib-tests)
target_sources(rocprofiler-sdk-lib-tests PRIVATE ${rocprofiler_lib_sources}
                                                 details/agent.cpp)
target_link_libraries(
    rocprofiler-sdk-lib-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::counter-test-constants
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-drm
            GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET rocprofiler-sdk-lib-tests
    SOURCES ${rocprofiler_lib_sources}
    TEST_LIST lib_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${lib_TESTS} PROPERTIES TIMEOUT 30 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                            "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# -------------------------------------------------------------------------------------- #
#
# Link to shared rocprofiler library
#
# -------------------------------------------------------------------------------------- #

set(rocprofiler_shared_lib_sources external_correlation.cpp intercept_table.cpp
                                   registration.cpp roctx.cpp status.cpp)

add_executable(rocprofiler-sdk-lib-tests-shared)
target_sources(rocprofiler-sdk-lib-tests-shared PRIVATE ${rocprofiler_shared_lib_sources})
target_link_libraries(
    rocprofiler-sdk-lib-tests-shared
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-drm
            rocprofiler-sdk::rocprofiler-sdk-roctx-shared-library
            GTest::gtest
            GTest::gtest_main)
set_target_properties(rocprofiler-sdk-lib-tests-shared PROPERTIES BUILD_RPATH
                                                                  "\$ORIGIN/../lib")

gtest_add_tests(
    TARGET rocprofiler-sdk-lib-tests-shared
    SOURCES ${rocprofiler_shared_lib_sources}
    TEST_LIST shared_lib_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set(rocprofiler-sdk-lib-tests-env
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}")

set_tests_properties(
    ${shared_lib_TESTS}
    PROPERTIES TIMEOUT 120 LABELS "unittests" ENVIRONMENT
               "${rocprofiler-sdk-lib-tests-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
