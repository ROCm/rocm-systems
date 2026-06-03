#
#   Tests for PC sampling reported-PC correction
#
project(rocprofiler-sdk-tests-pc-sampling LANGUAGES C CXX)

set(pc_sampling_sources pc_sampling_pc_correction_test.cpp)

add_executable(pc-sampling-pc-correction-tests)
target_sources(pc-sampling-pc-correction-tests PRIVATE ${pc_sampling_sources})
target_link_libraries(
    pc-sampling-pc-correction-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-output-library
            rocprofiler-sdk::rocprofiler-sdk-cereal
            GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET pc-sampling-pc-correction-tests
    SOURCES ${pc_sampling_sources}
    ENVIRONMENT
        "TEST_LOG_LEVEL=info"
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    )
