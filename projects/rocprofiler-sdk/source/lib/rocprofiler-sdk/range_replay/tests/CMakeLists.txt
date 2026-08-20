rocprofiler_deactivate_clang_tidy()

project(rocprofiler-sdk-range-replay-tests LANGUAGES CXX)

include(GoogleTest)

# Range eligibility bookkeeping and divergence checking are pure host logic with no GPU or
# HSA runtime dependency, so these are plain CXX unit tests that run unconditionally.
foreach(_TEST range-state digest)
    string(REPLACE "-" "_" _SOURCE_STEM "${_TEST}")
    set(_SOURCE "${_SOURCE_STEM}.cpp")

    add_executable(range-replay-${_TEST}-test)
    target_sources(range-replay-${_TEST}-test PRIVATE ${_SOURCE})

    target_link_libraries(
        range-replay-${_TEST}-test
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
                rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
                GTest::gtest_main)

    rocprofiler_add_unit_test(
        TARGET range-replay-${_TEST}-test
        SOURCES ${_SOURCE}
        TIMEOUT 60
        ENVIRONMENT
            "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
        FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endforeach()
