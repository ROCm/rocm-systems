#
# Waitcnt ISA tests
#

project(rocprofiler-sdk-tests-att-decoder-waitcnt LANGUAGES C CXX)

include(GoogleTest)

set(waitcnt_sources att_decoder_waitcnt_test.cpp)

add_executable(att-decoder-waitcnt-test)
target_sources(att-decoder-waitcnt-test PRIVATE ${waitcnt_sources})
target_link_libraries(
    att-decoder-waitcnt-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-att-parser
            rocprofiler-sdk::rocprofiler-sdk-json
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-glog
            rocprofiler-sdk::rocprofiler-sdk-static-library
            GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET att-decoder-waitcnt-test
    SOURCES ${waitcnt_sources}
    TIMEOUT 10
    ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:/opt/rocm/lib:/opt/rocm/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    )
