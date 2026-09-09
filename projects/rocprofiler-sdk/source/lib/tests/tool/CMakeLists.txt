#
# Tests for the rocprofv3 tool library (GPU-independent logic)
#
project(rocprofiler-sdk-tests-tool LANGUAGES CXX)

include(GoogleTest)

set(tool_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../rocprofiler-sdk-tool)

set(tool_test_sources kernel_filter_range.cpp kernel_iteration_filter.cpp)

# compile the GPU-independent tool logic directly into the test so it can be exercised
# without loading the tool runtime (which requires a GPU)
set(tool_lib_sources ${tool_dir}/kernel_filter_range.cpp
                     ${tool_dir}/kernel_iteration_filter.cpp)

add_executable(tool-tests)
target_sources(tool-tests PRIVATE ${tool_test_sources} ${tool_lib_sources})
target_include_directories(tool-tests PRIVATE ${tool_dir})
target_link_libraries(
    tool-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-common-library GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET tool-tests
    SOURCES ${tool_test_sources}
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    )
