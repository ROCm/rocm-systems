#
#
#
project(rocprofiler-sdk-tests-buffering LANGUAGES C CXX)

include(GoogleTest)

set(buffering_sources buffering-serial.cpp buffering-parallel.cpp buffering-save-load.cpp)

if(ROCPROFILER_EXPERIMENTAL_LTTNG_BUFFER_TRANSPORT)
    list(APPEND buffering_sources lttng-buffer.cpp)
endif()

add_executable(buffering-tests)
target_sources(buffering-tests PRIVATE ${buffering_sources})
target_link_libraries(
    buffering-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-common-library GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET buffering-tests
    SOURCES ${buffering_sources}
    ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:/opt/rocm/lib:/opt/rocm/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    )
