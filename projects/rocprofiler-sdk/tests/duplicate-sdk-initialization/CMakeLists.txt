#
# Duplicate rocprofiler-sdk initialization integration test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-duplicate-sdk-initialization LANGUAGES CXX)

add_executable(rocprofiler-sdk-duplicate-initialization-test)
target_sources(rocprofiler-sdk-duplicate-initialization-test PRIVATE main.cpp)
target_link_libraries(rocprofiler-sdk-duplicate-initialization-test
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers ${CMAKE_DL_LIBS})

if(TARGET rocprofiler-sdk-test-duplicate-shared-library)
    set(_duplicate_sdk "$<TARGET_FILE:rocprofiler-sdk-test-duplicate-shared-library>")
    add_dependencies(
        rocprofiler-sdk-duplicate-initialization-test rocprofiler-sdk-shared-library
        rocprofiler-sdk-test-duplicate-shared-library)
else()
    find_package(rocprofiler-sdk-tests REQUIRED)
    set(_duplicate_sdk
        "${rocprofiler-sdk-tests_ROOT_DIR}/duplicate-sdk/librocprofiler-sdk.so")
    if(NOT EXISTS "${_duplicate_sdk}")
        message(
            FATAL_ERROR "Missing installed duplicate rocprofiler-sdk: ${_duplicate_sdk}")
    endif()
endif()

rocprofiler_add_integration_execute_test(
    duplicate-sdk-initialization
    TARGET rocprofiler-sdk-duplicate-initialization-test
    ARGS "${_duplicate_sdk}"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk::rocprofiler-sdk-shared-library>"
    ENVIRONMENT "ROCPROFILER_LIBRARY_CTOR=1"
    TIMEOUT 30
    LABELS "integration-tests;duplicate-sdk"
    PASS_REGULAR_EXPRESSION "PASS: only the preloaded SDK initialized"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}|FAIL:")
