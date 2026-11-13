#
#   Unit tests of the out-of-process collector
#
project(rocprofiler-sdk-tests-collector LANGUAGES C CXX)

rocprofiler_deactivate_clang_tidy()

# The test target writes events with its own lttng-ust tracepoint provider. Without the
# lttng-ust development files, the tests that record a process are skipped.
find_package(PkgConfig REQUIRED)
pkg_check_modules(COLLECTOR_TEST_LTTNG_UST IMPORTED_TARGET lttng-ust>=2.13)

set(collector_test_app_path "")
if(COLLECTOR_TEST_LTTNG_UST_FOUND)
    add_executable(collector-test-app)
    target_sources(collector-test-app PRIVATE test_app.c)
    target_include_directories(collector-test-app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(collector-test-app PRIVATE PkgConfig::COLLECTOR_TEST_LTTNG_UST
                                                     ${CMAKE_DL_LIBS})
    set(collector_test_app_path "$<TARGET_FILE:collector-test-app>")
endif()

set(collector_test_sources collector.cpp)

add_executable(collector-tests)
# c_api.c checks that the public headers compile as C.
target_sources(collector-tests PRIVATE ${collector_test_sources} c_api.c)
target_compile_definitions(
    collector-tests PRIVATE ROCPROFILER_COLLECTOR_TEST_APP="${collector_test_app_path}")
target_link_libraries(
    collector-tests PRIVATE rocprofiler-sdk::rocprofiler-sdk-collector-shared-library
                            GTest::gtest GTest::gtest_main)
if(COLLECTOR_TEST_LTTNG_UST_FOUND)
    add_dependencies(collector-tests collector-test-app)
endif()

rocprofiler_add_unit_test(
    TARGET collector-tests
    SOURCES ${collector_test_sources}
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}"
    )
